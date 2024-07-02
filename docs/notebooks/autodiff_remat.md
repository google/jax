---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  name: python3
---

+++ {"id": "29WqUVkCXjDD"}

## Control autodiff's saved values with `jax.checkpoint` (aka `jax.remat`)

<!--* freshness: { reviewed: '2024-04-08' } *-->

```{code-cell}
import jax
import jax.numpy as jnp
```

+++ {"id": "qaIsQSh1XoKF"}

### TL;DR

Use the `jax.checkpoint` decorator (aliased as `jax.remat`) with `jax.grad` to control which intermediates are saved on the forward pass versus recomputed on the backward pass, trading off memory and FLOPs.

**Don't miss the [practical notes](#practical-notes) for a discussion about how `jax.checkpoint` interacts with `jax.jit`.**

Without using `jax.checkpoint`, the forward pass of `jax.grad(f)(x)` saves, for use on the backward pass, the values of Jacobian coefficients and other intermediates. We call these saved values _residuals_:

```{code-cell}
def g(W, x):
  y = jnp.dot(W, x)
  return jnp.sin(y)

def f(W1, W2, W3, x):
  x = g(W1, x)
  x = g(W2, x)
  x = g(W3, x)
  return x

W1 = jnp.ones((5, 4))
W2 = jnp.ones((6, 5))
W3 = jnp.ones((7, 6))
x = jnp.ones(4)

# Inspect the 'residual' values to be saved on the forward pass
# if we were to evaluate `jax.grad(f)(W1, W2, W3, x)`
from jax.ad_checkpoint import print_saved_residuals
jax.ad_checkpoint.print_saved_residuals(f, W1, W2, W3, x)
```

+++ {"id": "97vvWfI-fSSF"}

By applying `jax.checkpoint` to sub-functions, as a decorator or at specific application sites, we force JAX not to save any of that sub-function's residuals. Instead, only the inputs of a `jax.checkpoint`-decorated function might be saved, and any residuals consumed on the backward pass are re-computed from those inputs as needed:

```{code-cell}
def f2(W1, W2, W3, x):
  x = jax.checkpoint(g)(W1, x)
  x = jax.checkpoint(g)(W2, x)
  x = jax.checkpoint(g)(W3, x)
  return x

jax.ad_checkpoint.print_saved_residuals(f2, W1, W2, W3, x)
```

Here the values of two `sin` applications are saved because they are arguments
in subsequent applications of the `jax.checkpoint`-decorated `g` function, and
inputs to a `jax.checkpoint`-decorated function may be saved. But no values of
`cos` applications are saved.

+++ {"id": "CyRR3mTpjRtl"}

To control which values are saveable without having to edit the definition of the function to be differentiated, you can use a rematerialization _policy_. Here is an example that saves only the results of `dot` operations with no batch dimensions (since they are often FLOP-bound, and hence worth saving rather than recomputing):

```{code-cell}
f3 = jax.checkpoint(f, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
jax.ad_checkpoint.print_saved_residuals(f3, W1, W2, W3, x)
```

+++ {"id": "9fe6W0YxlfKa"}

You can also use policies to refer to intermediate values you name using `jax.ad_checkpoint.checkpoint_name`:

```{code-cell}
from jax.ad_checkpoint import checkpoint_name

def f4(W1, W2, W3, x):
  x = checkpoint_name(g(W1, x), name='a')
  x = checkpoint_name(g(W2, x), name='b')
  x = checkpoint_name(g(W3, x), name='c')
  return x

f4 = jax.checkpoint(f4, policy=jax.checkpoint_policies.save_only_these_names('a'))
jax.ad_checkpoint.print_saved_residuals(f4, W1, W2, W3, x)
```

+++ {"id": "40oy-FbmVkDc"}

When playing around with these toy examples, we can get a closer look at what's going on using the `print_fwd_bwd` utility defined in this notebook:

```{code-cell}
from jax.tree_util import tree_flatten, tree_unflatten

from rich.console import Console
from rich.table import Table
import rich.text

def print_fwd_bwd(f, *args, **kwargs) -> None:
  args, in_tree = tree_flatten((args, kwargs))

  def f_(*args):
    args, kwargs = tree_unflatten(in_tree, args)
    return f(*args, **kwargs)

  fwd = jax.make_jaxpr(lambda *args: jax.vjp(f_, *args))(*args).jaxpr

  y, f_vjp = jax.vjp(f_, *args)
  res, in_tree = tree_flatten(f_vjp)

  def g_(*args):
    *res, y = args
    f_vjp = tree_unflatten(in_tree, res)
    return f_vjp(y)

  bwd = jax.make_jaxpr(g_)(*res, y).jaxpr

  table = Table(show_header=False, show_lines=True, padding=(1, 2, 0, 2), box=None)
  table.add_row("[bold green]forward computation:",
                "[bold green]backward computation:")
  table.add_row(rich.text.Text.from_ansi(str(fwd)),
                rich.text.Text.from_ansi(str(bwd)))
  console = Console(width=240, force_jupyter=True)
  console.print(table)

def _renderable_repr(self):
  return self.html
rich.jupyter.JupyterRenderable._repr_html_ = _renderable_repr
```

```{code-cell}
# no use of jax.checkpoint:
print_fwd_bwd(f, W1, W2, W3, x)
```

```{code-cell}
# using jax.checkpoint with policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable:
print_fwd_bwd(f3, W1, W2, W3, x)
```

+++ {"id": "UsvnQJYomcub"}

### Let's think step by step

You might want to first (re)read [the Autodiff Cookbook Part 1](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html).

+++ {"id": "VMfwm_yinvoZ"}

#### Fundamentals of `jax.checkpoint`



In both `jax.linearize` and `jax.vjp` there is flexibility in how and when some values are computed. Different choices can trade off memory use against FLOPs. JAX provides control over these choices with `jax.checkpoint`.

One such choice is whether to perform Jacobian coefficient computations on the forward pass, as soon as the inputs are available, or on the backward pass, just before the coefficients are needed. Consider the example of `sin_vjp`:

```{code-cell}
def sin_vjp(x):
  y = jnp.sin(x)
  cos_x = jnp.cos(x)
  return y, lambda y_bar: cos_x * y_bar
```

+++ {"id": "7swp_LJcorL6"}

Another valid implementation would compute the value of `jnp.cos(x)` on the backward pass rather than on the forward pass:

```{code-cell}
def sin_vjp2(x):
  y = jnp.sin(x)
  return y, lambda y_bar: jnp.cos(x) * y_bar
```

+++ {"id": "uDaHIXzHo18i"}

For this particular function, the amount of memory used by the two versions is the same, though we've reduced the FLOPs for the primal computation (i.e. the forward pass) and increased the FLOPs for the cotangent computation (i.e. the backward pass).

There's another choice when it comes to function composition. Recall our VJP rule for a composition of two functions:

```{code-cell}
def f(x):
  y = g(x)
  z = h(y)
  return z

def f_vjp(x):
  y, g_vjp = jax.vjp(g, x)
  z, h_vjp = jax.vjp(h, y)
  def f_bwd(z_bar):
    y_bar, = h_vjp(z_bar)
    x_bar, = g_vjp(y_bar)
    return x_bar
  return z, f_bwd
```

+++ {"id": "6pC6Ng-6pigH"}

An alternative is:

```{code-cell}
def f_vjp_checkpoint(x):
  y = g(x)
  z, h_vjp = jax.vjp(h, y)
  def f_bwd2(z_bar):
    y_bar, = h_vjp(z_bar)
    _, g_vjp = jax.vjp(g, x)
    x_bar, = g_vjp(y_bar)
    return x_bar
  return z, f_bwd2
```

+++ {"id": "JYMw6oxtp6SH"}

In words, this alternative implementation doesn't compute `g_vjp`, or the residual values in its closure, on the forward pass. Instead it only computes them in the backward pass `f_bwd2`. That means `f_vjp_checkpoint` requires less memory: if `g` and `h` each required similar amounts of memory for their residuals, each much larger than `x`, then the function produced by `f_vjp_checkpoint(x)` requires half the memory as that of `f_vjp(x)`!

The cost we pay is redundant work: in `f_bwd2` we must re-evaluate `g(x)` as part of `jax.vjp(g, x)` just to discard its value (in the underscore variable on the line `_, g_vjp = jax.vjp(g, x)`).

+++ {"id": "LqTrjPoGqrK7"}

We can get this VJP behavior in autodiff &#151; without having to write VJP functions directly &#151; by instead using `jax.checkpoint` in an alternative definition of the original function `f`:

```{code-cell}
def f_checkpoint(x):
  y = jax.checkpoint(g)(x)
  z = h(y)
  return z
```

+++ {"id": "bjNcEAnLrUwy"}

In other words, we apply `jax.checkpoint` to `g`, the first stage of `f`, rather than to `f` itself. This way, when we evaluate `jax.grad(f_checkpoint)(x)`, we'd get a computation like:
1. run the forward pass of `g`, discarding residual values;
2. run the forward pass of `h`, saving residuals;
3. run the backward pass of `h`, consuming residuals from step 2;
4. re-run the forward pass of `g`, saving residuals;
5. run the backward pass of `g`, consuming residuals from step 4.

That is, by evaluating `jax.grad(f_checkpoint)(x)` we'd get the same computation as:

```{code-cell}
def f_checkpoint_grad(x):
  y = g(x)                  # step 1
  _, h_vjp = jax.vjp(h)(y)  # step 2
  y_bar, = h_vjp(1.0)       # step 3
  _, g_vjp = jax.vjp(g, x)  # step 4
  x_bar, = g_vjp(y_bar)     # step 5
  return x_bar
```

+++ {"id": "tfssPyb2sQgw"}

In general, `jax.checkpoint(foo)` is a new function which has the same input-output behavior as `foo`, but behaves differently under autodiff, particularly under `jax.linearize` and `jax.vjp` (and their wrappers, like `jax.grad`) but not `jax.jvp`. When differentiated, only the input to a `jax.checkpoint`-differentiated function is stored on the forward pass; on the backward pass, residuals (i.e. intermediates from `foo` and its Jacobian coefficient values needed for the backward pass) are recomputed.

+++ {"id": "HVvS_S5zsVZ-"}

Notice that if `f = lambda x: h(g(x))` is the function we want to differentiate, i.e. if we want to apply `jax.grad(f)`, we don't get any memory savings by applying `jax.checkpoint` to `f` itself. That's because evaluating `jax.grad(jax.checkpoint(f))(x)` would lead to a computation like:
1. run the forward pass, discarding all residuals;
2. immediately re-run the forward pass, saving residuals;
3. run the backward pass, consuming residuals from step 2.

That is, in code we'd have something like:

```{code-cell}
def f_grad_bad(x):
  _ = f(x)                  # step 1
  _, f_vjp = jax.vjp(f, x)  # step 2
  x_bar, = f_vjp(1.0)       # step 3
  return x_bar
```

+++ {"id": "yayWMylctuhM"}

We also wouldn't get any memory savings by applying `jax.checkpoint` to `h`, the second stage of `f`. That's because evaluating `jax.grad(lambda x: jax.checkpoint(h)(g(x)))` would lead to a computation like:
1. run the forward pass of `g`, saving residuals;
2. run the forward pass of `h`, discarding residuals;
3. immediately re-run the forward pass of `h`, saving residuals;
4. run the backward pass of `h`, consuming residuals from step 3;
5. run the backward pass of `g`, consuming residuals from step 1.

That is, in code we'd have something like:

```{code-cell}
def f_grad_bad2(x):
  y, g_vjp = jax.vjp(g, x)  # step 1
  z = h(y)                  # step 2
  _, h_vjp = jax.vjp(h, y)  # step 3
  y_bar, = h_vjp(1.0)       # step 3
  x_bar, = g_vjp(y_bar)     # step 5
  return x_bar
```

+++ {"id": "suLiGdSFxUOc"}

Slightly more generally, if we had a chain composition of functions, like `f = lambda x: f3(f2(f1(x)))`, and we were interested in evaluating `jax.grad(f)`, we could say that:
* we shouldn't apply `jax.checkpoint` to the whole function `f`, since that wouldn't save any memory (and will perform wasteful recomputation);
* we shouldn't apply `jax.checkpoint` to the last sub-function `f3`, since that wouldn't save any memory (and will perform wasteful recomputation);
* we could apply `jax.checkpoint` to `f1`, `f2`, or their composition `lambda x: f2(f1(x))`, since any of those might save memory and would express different memory/recompute tradeoffs.

+++ {"id": "s9KXvtlkyBfq"}

#### Custom policies for what's saveable

+++ {"id": "i9cPf56JyO_h"}

As shown so far, using `jax.checkpoint` switches from one extreme to another:
* without `jax.checkpoint`, JAX's autodiff tends to compute everything possible on the forward pass and store it for the backward pass;
* with a `jax.checkpoint` decorator, we instead compute as little as possible on the forward pass and recompute values as needed on the backward pass.

To operate between these two extremes, saving some things and not others, we can carefully place `jax.checkpoint` decorators on sub-functions. But that requires editing the function to be differentiated, e.g. model code, which may be inconvenient. It can also be hard to experiment with variations.

So an alternative is to use the `policy` argument to `jax.checkpoint`. A policy is a callable (i.e. a function) which takes as input a type-level specification of a first order primitive application and returns a boolean indicating whether the corresponding output value(s) are allowed to be saved as residuals (or instead must be recomputed in the (co)tangent computation as needed). To write robust code, a policy should be selected from the attributes on `jax.checkpoint_policies`, like `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`, since the API for writing custom policy callables is considered internal.

For example, consider this function to be differentiated:

```{code-cell}
def loss(params, x, y):
  return jnp.sum((predict(params, x) - y)**2)

def predict(params, x):
  *Ws, Wlast = params
  for W in Ws:
    x = layer(W, x)
  x = jnp.dot(Wlast, x)
  return x

def layer(W, x):
  return jnp.sin(jnp.dot(W, x))
```

```{code-cell}
W1 = W2 = W3 = jnp.ones((4, 4))
params = [W1, W2, W3]
x = jnp.ones(4)
y = jnp.ones(4)
```

```{code-cell}
print_saved_residuals(loss, params, x, y)
```

+++ {"id": "Mep7AReRNHby"}

Instead of saving so many values on the forward pass, perhaps we only want to save the results of matrix multiplications with no batch dimension (since they may be FLOP- rather than memory-bound). We can do that using the policy `jax.checkpoint_policies.dots_with_no_batch_dims_saveable`:

```{code-cell}
loss_checkpoint = jax.checkpoint(loss, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
print_saved_residuals(loss_checkpoint, params, x, y)
```

+++ {"id": "bSS5AgbhOtEO"}

Notice also that by providing a policy, we didn't need to edit the code defining `loss`, `predict`, or `layer`. That is particularly convenient if we want to experiment with policies in calling code (e.g. a training script) without changing library code (e.g. the neural network library).

+++ {"id": "wa8NudsITxNx"}

Some policies can refer to values named with `jax.ad_checkpoint.checkpoint_name`:

```{code-cell}
from jax.ad_checkpoint import checkpoint_name

def predict(params, x):
  *Ws, Wlast = params
  for i, W in enumerate(Ws):
    x = layer(W, x)
    x = checkpoint_name(x, name=f'layer{i}_output')
  x = jnp.dot(Wlast, x)
  return x
```

+++ {"id": "eu88XIW6UXW_"}

By itself, `checkpoint_name` is just an identity function. But because some policy functions know to look for them, we can use the names to control whether certain values output by `checkpoint_name` are considered saveable:

```{code-cell}
print_saved_residuals(loss, params, x, y)
```

```{code-cell}
loss_checkpoint2 = jax.checkpoint(loss, policy=jax.checkpoint_policies.save_any_names_but_these('layer1_output'))
print_saved_residuals(loss_checkpoint2, params, x, y)
```

+++ {"id": "YYBzKTnT6JkL"}

Another policy which refers to names is `jax.checkpoint_policies.save_only_these_names`.

Some of the policies are:
* `everything_saveable` (the default strategy, as if `jax.checkpoint` were not being used at all)
* `nothing_saveable` (i.e. rematerialize everything, as if a custom policy were not being used at all)
* `dots_saveable` or its alias `checkpoint_dots`
* `dots_with_no_batch_dims_saveable` or its alias `checkpoint_dots_with_no_batch_dims`
* `save_anything_but_these_names` (save any values except for the output of
  `checkpoint_name` with any of the names given)
* `save_any_names_but_these` (save only named values, i.e. any outputs of
  `checkpoint_name`, except for those with the names given)
* `save_only_these_names` (save only named values, and only among the names
  given)

Policies only indicate what is saveable; a value is only saved if it's actually needed by the backward pass.

+++ {"id": "lixGsLNwxQo7"}

#### Advanced: recursive `jax.checkpoint`

+++ {"id": "QHz3fQHZyT16"}

By applying `jax.checkpoint` in the right way, there are many tradeoffs between memory usage and (re)computation that can be expressed. One surprising example is _recursive_ checkpointing, where we apply `jax.checkpoint` to a function which itself calls `jax.checkpoint`-decorated functions in a way so that memory usage from the chain composition of $D$ functions scales like $\mathcal{O}(\log_2 D)$ rather than $\mathcal{O}(D)$.

As a toy example, consider the chain composition of multiple `jnp.sin` functions:

```{code-cell}
def chain_compose(funs):
  def f(x):
    for fun in funs:
      x = fun(x)
    return x
  return f

f = chain_compose([jnp.sin] * 8)
print_saved_residuals(f, 3.)
```

+++ {"id": "3SFXo4n6YJQG"}

In general, the number of stored residuals scales linearly with the length of the chain:

```{code-cell}
f = chain_compose([jnp.sin] * 16)
print_saved_residuals(f, 3.)
```

+++ {"id": "RcTuFRwZYXm7"}

But we can apply `jax.checkpoint` recursively to improve the scaling:

```{code-cell}
def recursive_checkpoint(funs):
  if len(funs) == 1:
    return funs[0]
  elif len(funs) == 2:
    f1, f2 = funs
    return lambda x: f1(f2(x))
  else:
    f1 = recursive_checkpoint(funs[:len(funs)//2])
    f2 = recursive_checkpoint(funs[len(funs)//2:])
    return lambda x: f1(jax.checkpoint(f2)(x))
```

```{code-cell}
f = recursive_checkpoint([jnp.sin] * 8)
print_saved_residuals(f, 3.)
```

```{code-cell}
f = recursive_checkpoint([jnp.sin] * 16)
print_saved_residuals(f, 3.)
```

+++ {"id": "D0yhcndHX0yN"}

The cost here, as usual, is recomputation: in particular, we end up performing $\mathcal{O}(\log_2 D)$ times as many FLOPs:

```{code-cell}
f = chain_compose([jnp.sin] * 8)
print_fwd_bwd(f, 3.)
```

```{code-cell}
f = recursive_checkpoint([jnp.sin] * 8)
print_fwd_bwd(f, 3.)
```

+++ {"id": "wvSm1yu0yUtd"}

### Practical notes

+++ {"id": "LSADkBOCyX9X"}

When differentiated functions are staged out to XLA for compilation, for example by applying `jax.jit` to a function which contains a `jax.grad` call, XLA will automatically optimize the computation, including decisions about when to compute or rematerialize values. As a result, **`jax.checkpoint` often isn't needed for differentiated functions under a `jax.jit`**. XLA will optimize things for you.

One exception is when using staged-out control flow, like `jax.lax.scan`. Automatic compiler optimizations across multiple control flow primitives, e.g. across a forward-pass `scan` and the corresponding backward-pass `scan`, typically aren't aren't as thorough. As a result, it's often a good idea to use `jax.checkpoint` on the body function passed to `jax.lax.scan`.

For example, one common pattern in large [Transformer models](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) is to express the architecture as a `jax.lax.scan` over layers so as to reduce compilation times. That is, using a simple fully-connected network as an analogy, instead of writing something like this:

+++ {"id": "BUeqKFRS5yPU"}

```python
LayerParam = tuple[jnp.ndarray, jnp.ndarray]  # weights, bias pair for a layer
ParamsList = list[LayerParam]

def net(params: ParamsList, x: jnp.ndarray):
  for W, b in params:
    x = jnp.maximum(jnp.dot(x, W) + b, 0.)
  return x
```

+++ {"id": "38xDcRwW518P"}

We would instead iterate over the layer application with `jax.lax.scan`:

+++ {"id": "ZU2fwYoG6A4z"}

```python
StackedWeights = jnp.ndarray  # all weight matrices stacked together
StackedBiases = jnp.ndarray   # all bias vectors stacked together

all_weights = jnp.stack([W for W, _ in params])
all_biases = jnp.stack([b for _, b in params])

def layer(x, W_b_pair):
  W, b = W_b_pair
  out = jnp.maximum(jnp.dot(x, W) + b, 0.)
  return out, None

def net(all_weights, all_biases, x):
  x, _ = jax.lax.scan(layer, x, (all_weights, all_biases))
  return x
```

+++ {"id": "A-NbCn8A6TFK"}

This scan-over-layers version reduces compile times, but by foiling some compiler optimizations it can lead to inefficient computation of gradients. To mitigate the issue, we would use `jax.checkpoint` on the scanned function:

+++ {"id": "iHVVNVdO66Dv"}

```python
from functools import partial

@partial(jax.checkpoint,
         policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
def layer(x, W_b_pair):
  W, b = W_b_pair
  out = jnp.maximum(jnp.dot(x, W) + b, 0.)
  return out, None
```

+++ {"id": "QYZlp_-s7D4M"}

By using `jax.checkpoint` this way, we're manually controlling which values JAX's autodiff saves between the forward and backward passes, and hence not relying on XLA optimizations to choose for us.
