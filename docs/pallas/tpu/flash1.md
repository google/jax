---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "_OPHxPDzDAIe"}

# The Math of Flash Attention

+++ {"id": "1NKSJ3qIKrYp"}

In this notebook, we will cover the math of flash attention, specifically for the forward pass. We will cover the backward pass in a future document. We won't dive into writing an optimized kernel *quite* yet, but will finish the guide with a simple vanilla JAX implementation.

```{code-cell}
:cellView: form
:id: jHaKJ7-GDRD_

#@title Imports
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
import numpy as np
```

+++ {"id": "VhHO861GEOqe"}

In JAX, we can implement self attention by doing einsum, divide, softmax, then einsum.

```{code-cell}
:id: heFFbbRCEQla

def attention(
    q: jax.Array,  # [seq_len, head_dim]
    k: jax.Array,  # [seq_len, head_dim]
    v: jax.Array,  # [seq_len, head_dim]
    ) -> jax.Array:
  logits = jnp.einsum('sd,td->st', q, k) / jnp.sqrt(q.shape[1])
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum('st,td->sd', probs, v)
```

+++ {"id": "mL-wR-OqDQh-"}

With flash attention, we are interested in writing a single kernel that does this entire operation without materializing any intermediates in memory (a.k.a. fusion). However, as we will soon see, the softmax makes this operation particularly tricky to fuse. Let's first dive into the softmax to see what exactly it's doing.

## Stable Softmax

Softmax is mathematically defined as (for a vector $x$)

$$
\sigma(x) \triangleq \frac{\exp x}{\sum_i \exp x_i}
$$

and in JAX, naively defined as:

```{code-cell}
:id: O-OzzPzPC-iC

def naive_softmax(x, axis=-1):
  x_exp = jnp.exp(x)
  return x_exp / jnp.sum(x_exp, axis=axis, keepdims=True)
```

+++ {"id": "b6Uruw4bDeTc"}

Softmax takes a vector of real-valued numbers (often called "logits") and normalizes it to make it a probability distribution; the resulting vector only has values between 0 and 1 and sums to 1.

```{code-cell}
---
executionInfo:
  elapsed: 8978
  status: ok
  timestamp: 1709845894749
  user:
    displayName: Sharad Vikram
    userId: '16113763449895047367'
  user_tz: 480
id: kpYVySD4Dd1t
outputId: cdf0118b-d3be-4474-a7f3-6c4914f611e5
---
logits = jnp.array([-1., 2., 3.])
probs = naive_softmax(logits)
print(probs, probs.sum())
```

+++ {"id": "mZdkQXElRPoj"}

In practice, we never implement softmax like this because of the possibility of numerical overflow and underflow. The `exp` function can overflow values to infinity or underflow values to zero, resulting in our `softmax` function outputting `nan`s.

```{code-cell}
---
executionInfo:
  elapsed: 2
  status: ok
  timestamp: 1709845894935
  user:
    displayName: Sharad Vikram
    userId: '16113763449895047367'
  user_tz: 480
id: s2LoLpDERsTh
outputId: 1c754725-e218-47e6-e33a-565a5964a1ca
---
# Overflowing one value
logits = jnp.array([70., 80., 90.])
probs = naive_softmax(logits)
print(probs, probs.sum())

# Underflowing all values
logits = jnp.array([-90., -110., -120.])
probs = naive_softmax(logits)
print(probs, probs.sum())
```

+++ {"id": "XMwfpz-2SD-0"}

Instead, we take advantage of the fact that we can shift our logits by a constant $c$ without changing the result of the softmax, i.e. $\sigma(x - c) = \sigma(x)$. If we set $c = \max_i x_i$, we will subtract off the largest logit and we can avoid over and underflow because the largest logit will always be 0.

```{code-cell}
:id: 49Doqlk2SDCt

@jax.jit
def softmax(x, axis: int = -1):
  x_max = jnp.max(x, axis=axis, keepdims=True)
  x_shifted = x - x_max
  x_numerator = jnp.exp(x_shifted)
  return x_numerator / jnp.sum(x_numerator, axis=axis, keepdims=True)
```

```{code-cell}
---
executionInfo:
  elapsed: 52
  status: ok
  timestamp: 1709845895352
  user:
    displayName: Sharad Vikram
    userId: '16113763449895047367'
  user_tz: 480
id: 5aU7VcmmSfXI
outputId: e12509dc-2c6f-4abe-d9d7-bbbd0485746c
---
# No more overflow!
logits = jnp.array([70., 80., 90.])
probs = softmax(logits)
print(probs, probs.sum())

# No more underflow!
logits = jnp.array([-90., -110., -120.])
probs = softmax(logits)
print(probs, probs.sum())
```

+++ {"id": "cQqVKlL66wzA"}

## Challenges of fusing attention

+++ {"id": "W5JbRV8n60Y8"}

Here's the implementation of attention one more time for us to look at.

```{code-cell}
:id: VeF4vwDt64on

def attention(
    q: jax.Array,  # [seq_len, head_dim]
    k: jax.Array,  # [seq_len, head_dim]
    v: jax.Array,  # [seq_len, head_dim]
    ) -> jax.Array:
  logits = jnp.einsum('sd,td->st', q, k) / jnp.sqrt(q.shape[1])
  probs = jax.nn.softmax(logits, axis=-1)
  return jnp.einsum('st,td->sd', probs, v)
```

+++ {"id": "ElBZberU66Dl"}

If we want to design a fused version of it, we need the operation to do a matrix multiplication, then a softmax, then a matrix multiplication all at the same time. Let's imagine this op but without the softmax.

```{code-cell}
:id: ihrYePt57iBC

def attention_without_softmax(
    q: jax.Array,  # [seq_len, head_dim]
    k: jax.Array,  # [seq_len, head_dim]
    v: jax.Array,  # [seq_len, head_dim]
    ) -> jax.Array:
  logits = jnp.einsum('sd,td->st', q, k) / jnp.sqrt(q.shape[1])
  o = jnp.einsum('st,td->sd', logits, v)
  return o
```

+++ {"id": "xFh5Ao1l7wOa"}

This function is basically two back to back matmuls. It's actually quite easy to write a fusion of this. We just need to notice that each entry of the output
vector `o_i` can be computed as $o_i = (q_i k^T) v$. Furthermore, the `head_dim` is often small (ranging from 32 to 512 usually) and can be comfortably fit into VMEM. Therefore, we don't need to tile the reduction dimension for the `q @ k.T` matmul and can tile the reduction over the `(q @ k.T) @ v` matmul.

```{code-cell}
:id: vmgFElfX8qlD

def attention_without_softmax_kernel(q_ref, k_ref, v_ref, o_ref):
  @pl.when(pl.program_id(1) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref[...])
  q = q_ref[...] / jnp.sqrt(q_ref.shape[1])
  transpose_dims = ((1,), (1,)), ((), ())
  qk = jax.lax.dot_general(q, k_ref[...], transpose_dims)
  o_ref[...] += qk @ v_ref[...]


def attention_without_softmax_pallas(q, k, v, *, bq: int = 128, bkv: int = 128):
  seq_len, head_dim = q.shape
  return pl.pallas_call(
      attention_without_softmax_kernel,
      out_shape=q,
      grid=(seq_len // bq, seq_len // bkv),
      in_specs=[
          pl.BlockSpec(lambda i, j: (i, 0), (bq, head_dim)),
          pl.BlockSpec(lambda i, j: (j, 0), (bkv, head_dim)),
          pl.BlockSpec(lambda i, j: (j, 0), (bkv, head_dim)),
      ],
      out_specs=pl.BlockSpec(lambda i, j: (i, 0), (bq, head_dim)),
  )(q, k, v)


k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
q = jax.random.normal(k1, (1024, 256))
k = jax.random.normal(k2, (1024, 256))
v = jax.random.normal(k3, (1024, 256))
np.testing.assert_allclose(
    attention_without_softmax(q, k, v),
    attention_without_softmax_pallas(q, k, v),
)
```

+++ {"id": "rvsvLCm77nMo"}

What happens if we add the softmax back in? Can we still do the same fusion? Well not exactly. The problem is because the softmax involves a sequence of *reductions* over the rows of `(q @ k.T)`. For a stable softmax, we have to do a maximum over the output to compute `x_max` and a sum over the output to compute the denominator of the softmax. We fiially need to normalize the vector by the denominator.

Softmax requires several passes over a vector to compute the final result, but inside our kernel, we're only computing a chunk of each row of `(q @ k.T)` at any given moment so we never have the full dimension present in VMEM and we only do one pass over the rows of `(q @ k.T)`. If we can fit that dimension in VMEM, then we can do all of those passes over the rows inside the kernel. Let's try a version of attention where we include the softmax, but only if we can fit the entire dimension in VMEM.

```{code-cell}
:id: rTGvAWL3EGwB

def attention_with_softmax_kernel(q_ref, k_ref, v_ref, o_ref):
  q = q_ref[...] / jnp.sqrt(q_ref.shape[1])
  transpose_dims = ((1,), (1,)), ((), ())
  qk = jax.lax.dot_general(q, k_ref[...], transpose_dims)
  # Call our existing stable softmax function
  probs = softmax(qk)
  o_ref[...] = probs @ v_ref[...]


def attention_pallas(q, k, v, *, bq: int = 128):
  seq_len, head_dim = q.shape
  return pl.pallas_call(
      attention_with_softmax_kernel,
      out_shape=q,
      grid=(seq_len // bq,),
      in_specs=[
          pl.BlockSpec(lambda i: (i, 0), (bq, head_dim)),
          pl.BlockSpec(lambda i: (0, 0), (seq_len, head_dim)),
          pl.BlockSpec(lambda i: (0, 0), (seq_len, head_dim)),
      ],
      out_specs=pl.BlockSpec(lambda i: (i, 0), (bq, head_dim)),
  )(q, k, v)


k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
q = jax.random.normal(k1, (1024, 256))
k = jax.random.normal(k2, (1024, 256))
v = jax.random.normal(k3, (1024, 256))
np.testing.assert_allclose(
    attention(q, k, v),
    attention_pallas(q, k, v),
    atol=1e-3,
    rtol=1e-3
)
```

+++ {"id": "fAM2WVZMeXCa"}

This fusion is similar to one XLA performs automatically at smaller sequence lengths. This strategy has the main disadvantage that if we increase the sequence length too much, we run out of VMEM. For example at 16k, our kernel will fail to compile due to VMEM OOM.

```{code-cell}
---
executionInfo:
  elapsed: 3087
  status: ok
  timestamp: 1709845900986
  user:
    displayName: Sharad Vikram
    userId: '16113763449895047367'
  user_tz: 480
id: X8CV28aZE_GK
outputId: 220134c3-f5bb-46d8-ed0c-ea9a629d7089
---
k1, k2, k3 = jax.random.split(jax.random.key(0), 3)
q = jax.random.normal(k1, (16384, 256))
k = jax.random.normal(k2, (16384, 256))
v = jax.random.normal(k3, (16384, 256))
try:
  np.testing.assert_allclose(
      attention(q, k, v),
      attention_pallas(q, k, v),
      atol=1e-3,
      rtol=1e-3
  )
except:
  print("Ran out of memory!")
```

+++ {"id": "h-mr5pdaFarD"}

Unfortunately, this fit-into-VMEM approach isn't scalable, so we'll have to come up with a different strategy to handle longer sequences.

+++ {"id": "aarziaQWFrRh"}

## Chunked softmax: the math

+++ {"id": "cRJQIBuYHJ_w"}

> Warning: this section will go into a lot of the mathematical detail. If you'd like to skip it, we'll outline the equations for flash attention at the beginning of the next section.


Rather than trying to squeeze the entire `seq_len` dimension into VMEM for `k` and `v`, could we operate on it in chunks like we did with the `attention_without_softmax`?

The answer is yes! But how to do it is not as simple and involves some math.

+++ {"id": "BBl_zLw9KHQo"}

To build intuition, let's say that we are doing a chunked softmax, a.k.a. we are computing the softmax of a vector but we would like to compute it by iterating over chunks of the vector. We'll keep some iteration state.

In this chunked softmax, for simplicity, we'll divide our vector $x$ into two chunks $[a; b]$. We'll assume that we are iterating over chunks and that we have *just* finished processing $a$ and are about to process $b$.

$$
\begin{align}
x &\triangleq [a; b]
\end{align}
$$

Let's assume *inductively* that we've computed some state about the softmax up to this point.
Specifically, let's assume we have $m_{\text{prev}}, l_{\text{prev}}$, and $f_{\text{prev}}$.

\begin{align}
m_\text{prev} &\triangleq \max_i a_i \\
f_\text{prev} &\triangleq \exp (a - m_\text{prev}) \\
l_\text{prev} &\triangleq \sum_i f_{\text{prev},i}
\end{align}

The idea is to eventually compute $m_\text{next}, f_\text{next}$ and $l_\text{next}$ given a the new chunk $b$ and the previous terms.

### Computing the updated terms

We'd now like to compute $m_\text{next}, l_\text{next}$ and $f_\text{next}$, a.k.a. the state that we'd hope would allow us to compute the softmax of $[a; b]$.

Let's first start by computing some local terms (i.e. terms computed only on $b$):

\begin{align}
m_\text{curr} &\triangleq \max_i b_i \\
f_\text{curr} &\triangleq \exp (b - m_\text{curr}) \\
l_\text{curr} &\triangleq \sum_i f_{\text{curr},i}
\end{align}

#### Computing $m_\text{next}$
How can we use these terms to compute an updated softmax state? The first thing to notice is that we need to keep track of running maximum. We know that:

$$
\begin{align}
m_\text{next} \triangleq \max(m_\text{prev}, m_\text{curr})
\end{align}
$$

#### Computing $f_\text{next}$
That was the easy part. How do we now compute $f_\text{next}$.

Ideally, we want $f_\text{next}$ to be $\exp(x - m(x))$. However, $m(x)$ could be either $m_\text{curr}$ or $m_\text{prev}$.

It seems like we want a conditional here where $f(x)$ depends on whether our running maximum was updated or not:
$$
f_\text{next} =
   \begin{cases}
      \exp\left(x - m_\text{curr}\right) & m_\text{curr} > m_\text{prev} \\
      \exp\left(x - m_\text{prev}\right) & \text{otherwise} \\
   \end{cases}
$$

However, there are two problems that make this impractical:
1. We don't have access to $a$ anymore (we only "remember" $m_\text{prev}, f_\text{prev}$ and $l_\text{prev}$).
2. If we eventually want to implement a chunked softmax as a GPU or TPU kernel, we want to avoid a conditional. Can we express this without explicitly comparing the $m_\text{curr}$ and $m_\text{prev}$ values?

Let's address the issue #1 first. How do we compute $f_\text{next}$ when we don't directly have access to $a$. Let's break it down into two cases:

##### Case 1: $m_\text{curr} > m_\text{prev}$ (a.k.a. we are updating our maximum).

Observe that
$$
\begin{align*}
f_\text{next} &=  [f_{\text{next},a}; f_{\text{next},b}] \\
&= [\exp(a - m_\text{curr}); \exp(b - m_\text{curr})] \\
&= [\exp(a - m_\text{curr}); f_\text{curr}]
\end{align*}
$$

This means our already computed $f_\text{curr}$ is ready to go!

However, $f_\text{prev} =\exp(a - m_\text{prev}) \neq \exp(a - m_\text{curr})$ so we need to adjust the $a$-part.

We therefore need to multiply $f_\text{prev}$ by a correction factor $\alpha = \exp(m_\text{prev} - m_\text{curr})$
producing:
$$
\begin{align*}
\exp(m_\text{prev} - m_\text{curr}) f_\text{prev} &= \exp(m_\text{prev} - m_\text{curr}) \exp(a - m_\text{prev}) \\
&= \exp(a - m_\text{curr}) = f_{\text{next},a}
\end{align*}
$$

Therefore, we set:
$$
f_\text{next} = [\alpha f_\text{prev}; f_\text{curr}]
$$

##### Case 2: $m_\text{curr} \leq m_\text{prev}$ (a.k.a. we keep our old maximum).

Observe that
$$
\begin{align*}
f_\text{next} &=  [f_{\text{next},a}; f_{\text{next},b}] \\
&= [\exp(a - m_\text{prev}); \exp(b - m_\text{prev})] \\
&= [f_\text{prev}; \exp(b - m_\text{prev})] \\
\end{align*}
$$

In this case, our previously computed $f_\text{prev}$ was correct!

Let's update $f_\text{curr} = \exp(b - m_\text{curr})$ in a similar fashion.
Let $\beta = \exp(m_\text{curr} - m_\text{prev})$ and let
$$
f_{\text{next},b} = \beta f_\text{curr}
$$

Therefore, we set:
$$
f_\text{next} =  [f_\text{prev}; \beta f_\text{curr}]
$$

Okay! We were able adjust our computed quantities to compute the correct $f_\text{next}$ in both cases (new or old max).

##### Computing $f_\text{next}$ without a conditional

Can we address issue #2 now and avoid using a conditional? Well it turns out to be really easy. Let's take a look at our correction terms:
$$
\begin{align*}
\alpha = \exp (m_\text{prev} - m_\text{curr}) \\
\beta = \exp (m_\text{curr} - m_\text{prev})
\end{align*}
$$
Let's use $m_\text{next}$ (which we have already computed) to rewrite these terms without changing their values in their conditionals.
$$
\begin{align}
\alpha \triangleq \exp (m_\text{prev} - m_\text{next}) \\
\beta \triangleq \exp (m_\text{curr} - m_\text{next})
\end{align}
$$

Convince yourself that this is okay and that this falls out from the definition of $m_\text{next}$.

Now, notice that when $m_\text{curr} > m_\text{prev}$ (a.k.a. case 1), $\beta = 1$. Similarly, when $m_\text{curr} \leq m_\text{prev}$, $\alpha = 1$.

That means, that the corrections turn into no-ops automatically if their case isn't triggered. We can therefore avoid the conditional by computing both $\alpha$ and $\beta$ and then computing:

$$
\begin{align}
f_\text{next} \triangleq [\alpha f_\text{prev}; \beta f_\text{curr}]
\end{align}
$$

#### Computing $l_\text{next}$

Well, the hard work is all done now!

We know $l_\text{next} = \sum_i f_{\text{next}, i}$.

Thanks to linearity, we can therefore compute:
$$
\begin{align}
l_\text{next} \triangleq \alpha l_\text{prev} + \beta l_\text{curr}
\end{align}
$$

That's it!

+++ {"id": "yu71mFxlMaLr"}

### Final chunked softmax equations

The equations for chunked softmax are, assuming we have access to $m_\text{prev}$, $l_\text{prev}$, $f_\text{prev}$ and a new chunk $b$, we first compute the local terms:

$$
\begin{align*}
m_\text{curr} &= \max_i b_i \\
f_\text{curr} &= \exp \left(b - m_\text{curr}\right) \\
l_\text{curr} &= \sum_i f_{\text{curr}, i}
\end{align*}
$$
To compute the updated terms for the next iteration, we compute:
$$
\begin{align*}
m_\text{next} &= \max \left(m_\text{prev}, m_\text{curr}\right) \\
\alpha &= \exp \left(m_\text{prev} - m_\text{next}\right) \\
\beta &= \exp \left(m_\text{curr} - m_\text{next}\right) \\
l_\text{next} &= \alpha l_\text{prev} + \beta l_\text{curr} \\
f_\text{next} &= \left[\alpha f_\text{prev}; \beta f_\text{curr}\right]
\end{align*}
$$

+++ {"id": "tQCqqi9viXUP"}

Here is an implementation in JAX as well:

```{code-cell}
:id: 8hlJRzKZMZad

def chunked_softmax(x, *, chunk_size: int = 128):
  assert x.shape[0] % chunk_size == 0
  m_prev = -jnp.inf
  l_prev = 0.
  f_prev = jnp.array([], dtype=np.float32)

  for i in range(x.shape[0] // chunk_size):
    x_chunk = x[i * chunk_size:(i + 1) * chunk_size]

    # Compute terms for current chunk
    m_curr = x_chunk.max()
    f_curr = jnp.exp(x_chunk - m_curr)
    l_curr = f_curr.sum()

    # Compute updated terms with corrections
    m_next = jnp.maximum(m_prev, m_curr)
    alpha = jnp.exp(m_prev - m_next)
    beta = jnp.exp(m_curr - m_next)
    l_next = alpha * l_prev + beta * l_curr
    f_next = jnp.concatenate([alpha * f_prev, beta * f_curr])

    m_prev, l_prev, f_prev = m_next, l_next, f_next
  return f_prev / l_prev

x = jax.random.normal(jax.random.key(0), (1024,), jnp.float32)
np.testing.assert_allclose(chunked_softmax(x), softmax(x), atol=1e-6, rtol=1e-6)
```

+++ {"id": "L77OM90mjfIy"}

## Flash Attention: the math

+++ {"id": "mHvP-j_EjhKj"}

> Again, we will go through some math here to get a deep understanding of the flash attention math. Feel free to skip to the end of this section to see the final equations.

Now that we know softmax can be computed online (in one pass) using the chunking approach, let's apply it to something that looks a bit more like flash attention.

What's the difference between flash attention and a chunked softmax? Well, not much actually. The main difference is that we fuse the softmax with a dot product. (In full attention, we'd be fusing a batched softmax with a matrix multiply).

Let's consider a really simple version of this where after computing $s = \text{softmax}(x)$ we want to compute $o = s \cdot v$ for some vector $v$.

Ideally, we'd do this by iterating over $s$ and $o$ exactly once and compute it online (i.e. fused).

Let's use our induction strategy from before by first partitioning $x = [a; b]$ and $v = [c; d]$.
Let's also assume we have $m_{\text{prev}}, l_{\text{prev}}, f_\text{prev}$, and $o_{\text{prev}}$.

\begin{align}
m_\text{prev} &\triangleq \max_i a_i \\
f_\text{prev} &\triangleq \exp (a - m_\text{prev}) \\
l_\text{prev} &\triangleq \sum_i f_{\text{prev},i} \\
o_\text{prev} &\triangleq \frac{f_\text{prev}}{l_\text{prev}} \cdot c
\end{align}

Let's also compute our local terms:

\begin{align}
m_\text{curr} &\triangleq \max_i b_i \\
f_\text{curr} &\triangleq \exp (b - m_\text{curr}) \\
l_\text{curr} &\triangleq \sum_i f_{\text{curr},i} \\
o_\text{curr} &\triangleq \frac{f_\text{curr}}{l_\text{curr}} \cdot d
\end{align}

We'd now like to compute $m_\text{next}, l_\text{next}, f_\text{next}$ and $o_\text{next}$.

We can exactly compute $m_\text{next}, l_\text{next}$ and $f_\text{next}$ as we did in chunked softmax.

$$
\begin{align*}
m_\text{next} &\triangleq \max(m_\text{prev}, m_\text{curr}) \\
\alpha &\triangleq \exp (m_\text{prev} - m_\text{next}) \\
\beta &\triangleq \exp (m_\text{curr} - m_\text{next}) \\
f_\text{next} &\triangleq [\alpha f_\text{prev}; \beta f_\text{curr}] \\
l_\text{next} &\triangleq \alpha l_\text{prev} + \beta l_\text{curr} \\
\end{align*}
$$

We now need to do a correction for $o_\text{prev}$ and $o_\text{curr}$ now by adopting a similar strategy to before (consider the cases when we do and don't update the maximum).

##### Case 1: $m_\text{curr} > m_\text{prev}$ (a.k.a. we are updating our maximum).

Ideally we want
$$
\begin{align*}
o_\text{next} &=  \frac{f_{\text{next},a}}{l_{\text{next}}} \cdot c + \frac{f_{\text{next},b}}{l_{\text{next}}} \cdot d \\
&=  \frac{\exp(a - m_\text{next})}{l_\text{next}} \cdot c + \frac{\exp(b - m_\text{next})}{l_\text{next}} \cdot d \\
&=  \frac{\exp(a - m_\text{curr})}{l_\text{next}} \cdot c + \frac{\exp(b - m_\text{curr})}{l_\text{next}} \cdot d
\end{align*}
$$

Notice that neither of the terms in the sum match something we have computed already. We'll have to correct both $o_\text{prev}$ and $o_\text{curr}$ to obtain the correct answer. Let's start with $o_\text{prev}$.

$$
\begin{align*}
o_\text{prev} &= \frac{f_\text{prev}}{l_\text{prev}} \cdot c \\
&= \frac{\exp(a - m_\text{prev})}{l_\text{prev}} \cdot c
\end{align*}
$$

Multiplying $o_\text{prev}$ by $l_\text{prev}\exp(m_\text{prev} - m_\text{curr}) / l_\text{next}$ we get:
$$
\begin{align}
\frac{l_\text{prev}\exp(m_\text{prev} - m_\text{curr})}{l_\text{next}} o_\text{prev}
&= \frac{l_\text{prev}\exp(m_\text{prev} - m_\text{curr})}{l_\text{next}}  \frac{\exp(a - m_\text{prev})}{l_\text{prev}} \cdot c\\
&= \frac{\exp(a - m_\text{curr})}{l_\text{next}}  \cdot c\\
\end{align}
$$,
making the terms match.  Notice also this correction is equal to $\frac{l_\text{prev}}{l_\text{next}} \alpha$.

Let's now correct $o_\text{curr}$. Remember that :

$$
\begin{align*}
o_\text{curr} &= \frac{f_\text{curr}}{l_\text{curr}} \cdot d \\
&= \frac{\exp(b - m_\text{curr})}{l_\text{curr}} \cdot d
\end{align*}
$$

Multiplying $o_\text{curr}$ by $l_\text{curr} / l_\text{next}$ we get:
$$
\begin{align}
\frac{l_\text{curr}}{l_\text{next}} \frac{\exp(b - m_\text{curr})}{l_\text{curr}} \cdot d = \frac{\exp(b - m_\text{curr})}{l_\text{next}} \cdot d
\end{align}
$$

Therefore, we set:
$$
\begin{align*}
o_\text{next} &= \frac{l_\text{prev}}{l_\text{next}} \alpha o_\text{prev} + \frac{l_\text{curr}}{l_\text{next}}o_\text{curr} \\
&= \frac{1}{l_\text{next}}\left(l_\text{prev} \alpha o_\text{prev} + l_\text{curr}o_\text{curr}\right) \\
\end{align*}
$$

##### Case 2: $m_\text{curr} \leq m_\text{prev}$ (a.k.a. we keep our old maximum).

Ideally we want
$$
\begin{align*}
o_\text{next} &=  \frac{f_{\text{next},a}}{l_{\text{next}}} \cdot c + \frac{f_{\text{next},b}}{l_{\text{next}}} \cdot d \\
&=  \frac{\exp(a - m_\text{next})}{l_\text{next}} \cdot c + \frac{\exp(b - m_\text{next})}{l_\text{next}} \cdot d \\
&=  \frac{\exp(a - m_\text{prev})}{l_\text{next}} \cdot c + \frac{\exp(b - m_\text{prev})}{l_\text{next}} \cdot d
\end{align*}
$$

Notice that neither of the terms in the sum match something we have computed already. We'll have to correct both $o_\text{prev}$ and $o_\text{curr}$ to obtain the correct answer. Let's start with $o_\text{prev}$.

$$
\begin{align*}
o_\text{prev} &= \frac{f_\text{prev}}{l_\text{prev}} \cdot c \\
&= \frac{\exp(a - m_\text{prev})}{l_\text{prev}} \cdot c
\end{align*}
$$

Multiplying $o_\text{prev}$ by $l_\text{prev} / l_\text{next}$ we get:
$$
\begin{align}
\frac{l_\text{prev}}{l_\text{next}} o_\text{prev}
&= \frac{l_\text{prev}}{l_\text{next}}  \frac{\exp(a - m_\text{prev})}{l_\text{prev}} \cdot c\\
&= \frac{\exp(a - m_\text{prev})}{l_\text{next}}  \cdot c\\
\end{align}
$$,
making the terms match.

Let's now correct $o_\text{curr}$. Remember that:

$$
\begin{align*}
o_\text{curr} &= \frac{f_\text{curr}}{l_\text{curr}} \cdot d \\
&= \frac{\exp(b - m_\text{curr})}{l_\text{curr}} \cdot d
\end{align*}
$$

Multiplying $o_\text{curr}$ by $l_\text{curr}\exp(m_\text{curr} - m_\text{prev}) / l_\text{next}$ we get:
$$
\begin{align}
\frac{l_\text{curr}\exp(m_\text{curr}-  m_\text{prev})}{l_\text{next}} \frac{\exp(b - m_\text{curr})}{l_\text{curr}} \cdot d = \frac{\exp(b - m_\text{prev})}{l_\text{next}} \cdot d
\end{align}
$$

Notice that this correction is equal to $\frac{l_\text{curr}}{l_\text{next}} \beta$.

Therefore, we set:
$$
\begin{align*}
o_\text{next} &= \frac{l_\text{prev}}{l_\text{next}} o_\text{prev} + \frac{l_\text{curr}}{l_\text{next}}\beta o_\text{curr} \\
&= \frac{1}{l_\text{next}}\left(l_\text{prev} o_\text{prev} + l_\text{curr}\beta o_\text{curr}\right)
\end{align*}
$$

Unifying both sides of the conditional, we get the final update:

$$
\begin{align}
o_\text{next} &\triangleq \frac{1}{l_\text{next}}\left(l_\text{prev} \alpha o_\text{prev} + l_\text{curr}\beta o_\text{curr}\right)
\end{align}
$$

+++ {"id": "vseyzYjqrR53"}

## Final flash attention equations and implementation

+++ {"id": "a9b9po7DrOCI"}

Putting everything together:

\begin{align}
m_\text{curr} &\triangleq \max_i b_i \\
f_\text{curr} &\triangleq \exp (b - m_\text{curr}) \\
l_\text{curr} &\triangleq \sum_i f_{\text{curr},i} \\
o_\text{curr} &\triangleq \frac{f_\text{curr}}{l_\text{curr}} \cdot d
\end{align}

$$
\begin{align}
m_\text{next} &\triangleq \max(m_\text{prev}, m_\text{curr}) \\
\alpha &\triangleq \exp (m_\text{prev} - m_\text{next}) \\
\beta &\triangleq \exp (m_\text{curr} - m_\text{next}) \\
f_\text{next} &\triangleq [\alpha f_\text{prev}; \beta f_\text{curr}] \\
l_\text{next} &\triangleq \alpha l_\text{prev} + \beta l_\text{curr} \\
o_\text{next} &\triangleq \frac{1}{l_\text{next}}\left(l_\text{prev} \alpha o_\text{prev} + l_\text{curr}\beta o_\text{curr}\right)
\end{align}
$$

Let's try the JAX version.

```{code-cell}
:id: yvtE0TRBima5

def flash_attention_jax(q, k, v, *, chunk_size: int = 128):
  assert q.shape[0] % chunk_size == 0
  seq_len, head_dim = q.shape

  output = jnp.zeros_like(q)

  for q_index in range(seq_len // chunk_size):
    q_slice = slice(q_index * chunk_size, (q_index + 1) * chunk_size)
    q_chunk = q[q_slice]

    m_prev = jnp.full(chunk_size, -jnp.inf, dtype=jnp.float32)
    l_prev = jnp.zeros(chunk_size, dtype=jnp.float32)
    o_prev = jnp.zeros((chunk_size, head_dim), dtype=jnp.float32)

    for kv_index in range(seq_len // chunk_size):
      kv_slice = slice(kv_index * chunk_size, (kv_index + 1) * chunk_size)
      k_chunk = k[kv_slice]
      # compute logits
      logits = jnp.einsum('sd,td->st', q_chunk, k_chunk) / jnp.sqrt(
          q_chunk.shape[1]
      )

      # compute terms for current chunk
      m_curr = logits.max(axis=-1)
      f_curr = jnp.exp(logits - m_curr[:, None])
      l_curr = f_curr.sum(axis=-1)
      v_chunk = v[kv_slice]
      o_curr = jnp.einsum('st,td->sd', f_curr / l_curr[:, None], v_chunk)

      # compute updated terms with corrections
      m_next = jnp.maximum(m_prev, m_curr)
      alpha = jnp.exp(m_prev - m_next)
      beta = jnp.exp(m_curr - m_next)
      l_next = alpha * l_prev + beta * l_curr
      o_next = (
          1
          / l_next[:, None]
          * (
              (l_prev * alpha)[:, None] * o_prev
              + (l_curr * beta)[:, None] * o_curr
          )
      )
      m_prev, l_prev, o_prev = m_next, l_next, o_next
    output = output.at[q_slice].set(o_prev)
  return output


q = jax.random.normal(jax.random.key(0), (1024, 256), jnp.float32)
k = jax.random.normal(jax.random.key(0), (1024, 256), jnp.float32)
v = jax.random.normal(jax.random.key(0), (1024, 256), jnp.float32)
np.testing.assert_allclose(
    flash_attention_jax(q, k, v), attention(q, k, v), atol=1e-2, rtol=1e-2
)
```

+++ {"id": "TG8ouE4vvZyD"}

Note the increased tolerances on our results check. **Flash attention is a numerics-changing optimization**, so it is always good to validate its results in practice.

+++ {"id": "9BSkntRsvll0"}

## Conclusion

+++ {"id": "elCU5dt6walT"}

In this guide we covered why attention is hard to fuse in general, how to express softmax in a chunked, online fashion, and how to generalize that strategy to compute attention. We completed the guide with a simple JAX implementation of flash attention that we'll use as our basis of writing our flash attention kernel. In a future guide, we will also cover the math of the backward pass of flash attention.

Exercises for the reader:
* We implemented flash attention in such a way that leaves some performance on the table. Can you optimize the softmax corrections to avoid as many operations?
* It's common in practice to use "causal" attention, where each `q` will not attend to `k`s and `v`s ahead of it in the sequence dimension. Try adding that variant to our JAX implementation.
* Try running ahead and implement a basic version of the kernel, you should have the tools!
