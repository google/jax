# Introduction

Self-attention is arguably the most critical operation in transformers and speeding it up on hardware accelerators is key to scaling large language models. Here is a simple implementation of it in JAX:

```python
def attention(
    q: jax.Array,  # [seq_len, head_dim]
    k: jax.Array,  # [seq_len, head_dim]
    v: jax.Array,  # [seq_len, head_dim]
    ) -> jax.Array:
  logits = jnp.einsum('sd,td->st', q, k) / jnp.sqrt(q.shape[1])
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum('st,td->sd', probs, v)
```

Attention operates on sequences of vectors but is $O(n^2)$ in its runtime complexity ($n$ being sequence length) due to the outer product between `q` and `k` long the sequence dimension. This quadratic requirement often makes it a computational bottleneck in transformer training. Furthermore, if naively implemented, self-attention utilizes hardware accelerators very poorly due to inefficient use of memory. In fact, it may even have $O(n^2)$ memory usage, making long sequence training untenable just due to memory requirements.

XLA TPU can often fuse this entire sequence of operations into one kernel but with long enough sequences, the fusion isn't possible. As a result, we will materialize the $O(n^2)$-size `logits` and `probs` matrices, and the individual operations of attention will be memory-bound, not compute bound. Enter [Flash attention](https://arxiv.org/abs/2205.14135) by Dao et al, an algorithm for computing self-attention exactly and greatly improve accelerator utilization. While still remaining $O(n^2)$ in time complexity, it avoids $O(n^2)$ memory capacity usage and improves memory bandwidth usage as well, leading to dramatic speedups in practice. While the original paper and implementation was GPU-specific, the algorithm is works on TPUs as well and using Pallas TPU we can implement our own efficient flash attention kernel.

Flash attention is particularly tricky to learn because it includes both a mathematical rewrite of attention, and knowledge of how to maximally utilize accelerators. In this series of guides, we will go over both the math of flash attention and how to efficiently implement it on TPUs using Pallas.
