# Copyright 2023 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of benchmarking a Pallas kernel (fused attention).

Since a common use-case of Pallas may be to write performant tiled kernels with 
certain amount of low-level control, users will often want to benchmark their
Pallas implementation against pure JAX implementations (e.g. lowered by XLA) and 
implementations in external libraries.

Here, we show an example benchmarking fused attention for the following:
1. Pallas implementation
2. Pure-JAX implementation
3. Triton implementation (with PyTorch tensor infra)
4. flash_attn implementation

TODO:
1. cuDNN
2. xformers

We choose the settings to be similar to those in 
https://tridao.me/publications/flash2/flash2.pdf
"""

import functools
import time
import math

import matplotlib.pyplot as plt
import triton 
import triton.language as tl
from triton import cdiv
import torch

from jax import random
import random as pyrand
import jax
import jax.numpy as jnp
from jax.experimental.pallas.ops import attention


DIM = 2048
D_HEAD = 64
N_HEADS = DIM // D_HEAD
BATCH, SEQ_LEN = 8, 2048
SEQ_LENS = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

TOTAL_RUNS = 150
NUM_OUTER_RUNS = 5 # For randomization of benchmark order
NUM_INNER_RUNS = (TOTAL_RUNS + NUM_OUTER_RUNS - 1) // NUM_OUTER_RUNS

BETWEEN_RUN_SLEEP_TIME_MS = 200

DELAYED_SOFTMAX_NORMALIZE = True
SEPARATE_ON_GRAPH = True

"""
Appendix
"""

@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    DELAYED_SOFTMAX_NORMALIZE: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(BLOCK_DMODEL, N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr)
    q = (q * qk_scale).to(K.dtype.element_ty)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        qk += tl.dot(q, k, allow_tf32=True)
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])

        if DELAYED_SOFTMAX_NORMALIZE:
            # -- scale and update acc --
            acc *= alpha[:, None]
            acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new
        else:
            l_i = l_i * alpha
            l_i_new = l_i + tl.sum(p, 1)
            l_rcp = 1. / l_i_new
            p *= l_rcp[:, None]
            # -- scale and update acc --
            acc *= (l_i * l_rcp)[:, None] 
            acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
            # -- update m_i and l_i --
            l_i = l_i_new
            m_i = m_i_new

        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    if DELAYED_SOFTMAX_NORMALIZE:
        acc = acc / l_i[:, None]

    # write back l and m
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, sequence_parallel=False):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        BLOCK_M = 128
        BLOCK_N = 64
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        num_warps = 4 if Lk <= 64 else 8
        compiled = _fwd_kernel[grid](
            q, k, v, sm_scale,
            L,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            DELAYED_SOFTMAX_NORMALIZE=DELAYED_SOFTMAX_NORMALIZE,
            num_warps=num_warps,
            num_stages=4)
        
        from triton.compiler import compiler as tc 
        # print("IR", compiled.asm['ttir'])
        # print("TTGIR", compiled.asm['ttgir'])
        # print("IR", compiled.asm['ptx'])

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.sequence_parallel = sequence_parallel
        return o

triton_attention = _attention.apply

def benchmark_jax(batch=BATCH, heads=N_HEADS, seq_len=SEQ_LEN, d_model=D_HEAD, causal=True, mode="jax", swap_seq_axis=False):
    block_qk_grid = [(128, 64)] #[(64, 32), (128, 32), (64, 64)if mode == "pallas" else [(None, None)] 
    k1, k2, k3 = random.split(random.PRNGKey(0), 3)
    if swap_seq_axis:
        shape = (batch, heads, seq_len, d_model)
    else:
        shape =  (batch, seq_len, heads, d_model)
    q = random.normal(k1, shape, dtype=jnp.float16)
    k = random.normal(k2, shape, dtype=jnp.float16)
    v = random.normal(k3, shape, dtype=jnp.float16)


    min_ms = float("inf")

    # Perform a grid search and choose the best timing
    for block_q, block_k in block_qk_grid:
        if mode == "pallas":
            impl = functools.partial(
                attention.mha, causal=causal, block_q=block_q, block_k=block_k, num_warps=4, segment_ids=None, debug=False, swap_seq_axis=swap_seq_axis)
        elif mode == "jax":
            if seq_len >= 2048: # Handle OOM
                return None
            impl = functools.partial(attention.mha_reference, segment_ids=None)
        else:
            raise ValueError("Invalid JAX benchmark mode")

        # Warm up
        impl(q, k, v).block_until_ready()
        impl(q, k, v).block_until_ready()

        t1 = time.time()
        for _ in range(NUM_INNER_RUNS):
            impl(q, k, v).block_until_ready()
        estimate_ms = 1000 * (time.time() - t1) / NUM_INNER_RUNS
        min_ms = min(estimate_ms, min_ms)
        print(f"{mode} (seq_len={seq_len}, block_q={block_q}, block_k={block_k}): {estimate_ms} ms")
    return min_ms

# Mode is one of {"triton", "flash_attn"}
def bench_torch(batch=BATCH, heads=N_HEADS, seq_len=SEQ_LEN, d_model=D_HEAD, causal=True, mode="triton"):
    import torch
    dtype = torch.float16
    if mode == "triton":
        """
        Triton implementation broken in dep of jax-triton: 
        `RuntimeError: CUDA error: an illegal memory access was encountered`
        """
        # from triton.ops import attention as triton_attention
        # Use a jitted function from triton nightly 28/08/23 as defined below.
        fn = lambda: triton_attention(q, k, v, causal, 1.0)
        shape = (batch, heads, seq_len, d_model)
    elif mode == "flash_attn":
        from flash_attn import flash_attn_func
        fn = lambda: flash_attn_func(q, k, v, causal=causal)
        shape =  (batch, seq_len, heads, d_model)
    else:
        raise ValueError("Invalid JAX benchmark mode")
    q = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    # Warmup
    fn()
    fn()
    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(NUM_INNER_RUNS):
        fn()
        torch.cuda.synchronize()
    estimate_ms = 1000 * (time.time() - t1) / NUM_INNER_RUNS
    return estimate_ms

# TODO: implement this
def test_allclose():
    pass

def tflops_from_ms(timing, seq_len, tokens):
    return (4 * seq_len ** 2 * D_HEAD * N_HEADS * tokens / seq_len) / (timing * 10 ** 9)

def is_zero(a):
    return math.isclose(a, 0.0, abs_tol=0.00001)

def benchmark(causal=True):
    configs = [
        {'name': name, 'timings': [0.0 for _ in range(len(SEQ_LENS))], 'tokens': tokens } 
        for name in ["pallas", "triton", "flash_attn", "jax"] 
        for tokens in [32768]
    ]

    bench_fns = {
        'jax': functools.partial(benchmark_jax, mode="jax"),
        'pallas': functools.partial(benchmark_jax, mode="pallas"),
        # 'pallas_swap': functools.partial(benchmark_jax, mode="pallas", swap_seq_axis=True),
        'flash_attn': functools.partial(bench_torch, mode="flash_attn"),
        'triton': functools.partial(bench_torch, mode="triton"),
    }
    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)

    for _ in range(NUM_OUTER_RUNS):
        shuffled = configs[:] 
        pyrand.shuffle(shuffled)
        print("ORDERING", [cfg['name'] for cfg in shuffled])
        for config in shuffled:
            for s_idx, s in enumerate(SEQ_LENS):
            # Randomize order of configs as the order been shown to matter (esp for small runs) 
                res = bench_fns[config['name']](
                        batch=config['tokens'] // s, heads=N_HEADS, seq_len=s, d_model=D_HEAD, causal=causal)
                if res is not None and config['timings'][s_idx] is not None:
                    config['timings'][s_idx] += res

                time.sleep(BETWEEN_RUN_SLEEP_TIME_MS / 1000)

    # preprocess
    for config in configs:
        config['timings'] = [None if is_zero(t) else t / NUM_OUTER_RUNS for t in config['timings']]

    len_configs = float(len(configs))
    min_timings = [min([config['timings'][pos] for config in configs if config['timings'][pos] is not None]) for pos in range(len(SEQ_LENS))]

    configs.sort(key=lambda c: ([t for t in c['timings'] if t is not None] or [float.max])[-1], reverse=True)
    
    for config_idx, config in enumerate(configs):
        config_pos = (float(len_configs - config_idx) - len_configs / 2) / len_configs
        ax.plot(SEQ_LENS, config['timings'], label=config['name'] + '_' + str(config['tokens'] // 1000) + 'K_tokens' )
        for seq_len, timing, min_timing in zip(SEQ_LENS, config['timings'], min_timings): 
            if timing is not None:
                if SEPARATE_ON_GRAPH:
                    timing_height = timing * (1. + 0.1 * config_pos)
                else:
                    timing_height = timing
                tflops = tflops_from_ms(timing, seq_len, config['tokens'])
                max_tflops = tflops_from_ms(min_timing, seq_len, config['tokens'])
                percentage_of_max = tflops / max_tflops
                plt.text(seq_len, timing_height, f"{config['name']}: TFLOPs={round(tflops, 1)} ({round(percentage_of_max * 100, 2)}% max) - {round(timing, 1)}ms")
    plt.title(f'Fused Attention ({"Causal" if causal else "Non-Causal"})')
    plt.ylabel('time (ms)')
    plt.xlabel('Sequence Length')
    ax.set_xticks(SEQ_LENS, SEQ_LENS)
    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_allclose()
    benchmark()


