---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(advanced-autodiff)=
# Advanced automatic differentiation

<!--* freshness: { reviewed: '2024-05-14' } *-->

In this tutorial, you will learn about complex applications of automatic differentiation (autodiff) in JAX and gain a better understanding of how taking derivatives in JAX can be both easy and powerful.

Make sure to check out the {ref}`automatic-differentiation` tutorial to go over the JAX autodiff basics, if you haven't already.

## Setup

```{code-cell}
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

key = random.key(0)
```

## Taking gradients (part 2)

### Higher-order derivatives

JAX's autodiff makes it easy to compute higher-order derivatives, because the functions that compute derivatives are themselves differentiable. Thus, higher-order derivatives are as easy as stacking transformations.

The single-variable case was covered in the {ref}`automatic-differentiation` tutorial, where the example showed how to use {func}`jax.grad` to compute the the derivative of $f(x) = x^3 + 2x^2 - 3x + 1$.

In the multivariable case, higher-order derivatives are more complicated. The second-order derivative of a function is represented by its [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix), defined according to:

$$(\mathbf{H}f)_{i,j} = \frac{\partial^2 f}{\partial_i\partial_j}.$$

The Hessian of a real-valued function of several variables, $f: \mathbb R^n\to\mathbb R$, can be identified with the [Jacobian](https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant) of its gradient.

JAX provides two transformations for computing the Jacobian of a function, {func}`jax.jacfwd` and {func}`jax.jacrev`, corresponding to forward- and reverse-mode autodiff. They give the same answer, but one can be more efficient than the other in different circumstances – refer to the [video about autodiff](https://www.youtube.com/watch?v=wG_nF1awSSY).

```{code-cell}
def hessian(f):
  return jax.jacfwd(jax.grad(f))
```

Let's double check this is correct on the dot-product $f: \mathbf{x} \mapsto \mathbf{x} ^\top \mathbf{x}$.

if $i=j$, $\frac{\partial^2 f}{\partial_i\partial_j}(\mathbf{x}) = 2$. Otherwise, $\frac{\partial^2 f}{\partial_i\partial_j}(\mathbf{x}) = 0$.

```{code-cell}
def f(x):
  return jnp.dot(x, x)

hessian(f)(jnp.array([1., 2., 3.]))
```

## Higher-order optimization

Some meta-learning techniques, such as Model-Agnostic Meta-Learning ([MAML](https://arxiv.org/abs/1703.03400)), require differentiating through gradient updates. In other frameworks this can be quite cumbersome, but in JAX it's much easier:

```python
def meta_loss_fn(params, data):
  """Computes the loss after one step of SGD."""
  grads = jax.grad(loss_fn)(params, data)
  return loss_fn(params - lr * grads, data)

meta_grads = jax.grad(meta_loss_fn)(params, data)
```


### Stopping gradients

Autodiff enables automatic computation of the gradient of a function with respect to its inputs. Sometimes, however, you might want some additional control: for instance, you might want to avoid backpropagating gradients through some subset of the computational graph.

Consider for instance the TD(0) ([temporal difference](https://en.wikipedia.org/wiki/Temporal_difference_learning)) reinforcement learning update. This is used to learn to estimate the *value* of a state in an environment from experience of interacting with the environment. Let's assume the value estimate $v_{\theta}(s_{t-1}$) in a state $s_{t-1}$ is parameterised by a linear function.

```{code-cell}
# Value function and initial parameters
value_fn = lambda theta, state: jnp.dot(theta, state)
theta = jnp.array([0.1, -0.1, 0.])
```

Consider a transition from a state $s_{t-1}$ to a state $s_t$ during which you observed the reward $r_t$

```{code-cell}
# An example transition.
s_tm1 = jnp.array([1., 2., -1.])
r_t = jnp.array(1.)
s_t = jnp.array([2., 1., 0.])
```

The TD(0) update to the network parameters is:

$$
\Delta \theta = (r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})) \nabla v_{\theta}(s_{t-1})
$$

This update is not the gradient of any loss function.

However, it can be **written** as the gradient of the pseudo loss function

$$
L(\theta) = - \frac{1}{2} [r_t + v_{\theta}(s_t) - v_{\theta}(s_{t-1})]^2
$$

if the dependency of the target $r_t + v_{\theta}(s_t)$ on the parameter $\theta$ is ignored.

How can you implement this in JAX? If you write the pseudo loss naively, you get:

```{code-cell}
def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return -0.5 * ((target - v_tm1) ** 2)

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

delta_theta
```

But `td_update` will **not** compute a TD(0) update, because the gradient computation will include the dependency of `target` on $\theta$.

You can use {func}`jax.lax.stop_gradient` to force JAX to ignore the dependency of the target on $\theta$:

```{code-cell}
def td_loss(theta, s_tm1, r_t, s_t):
  v_tm1 = value_fn(theta, s_tm1)
  target = r_t + value_fn(theta, s_t)
  return -0.5 * ((jax.lax.stop_gradient(target) - v_tm1) ** 2)

td_update = jax.grad(td_loss)
delta_theta = td_update(theta, s_tm1, r_t, s_t)

delta_theta
```

This will treat `target` as if it did **not** depend on the parameters $\theta$ and compute the correct update to the parameters.

Now, let's also calculate $\Delta \theta$ using the original TD(0) update expression, to cross-check our work. You may wish to try and implement this yourself using {func}`jax.grad` and your knowledge so far. Here's our solution:

```{code-cell}
s_grad = jax.grad(value_fn)(theta, s_tm1)
delta_theta_original_calculation = (r_t + value_fn(theta, s_t) - value_fn(theta, s_tm1)) * s_grad

delta_theta_original_calculation # [1.2, 2.4, -1.2], same as `delta_theta`
```

`jax.lax.stop_gradient` may also be useful in other settings, for instance if you want the gradient from some loss to only affect a subset of the parameters of the neural network (because, for instance, the other parameters are trained using a different loss).


### Straight-through estimator using `stop_gradient`

The straight-through estimator is a trick for defining a 'gradient' of a function that is otherwise non-differentiable. Given a non-differentiable function $f : \mathbb{R}^n \to \mathbb{R}^n$ that is used as part of a larger function that we wish to find a gradient of, we simply pretend during the backward pass that $f$ is the identity function. This can be implemented neatly using `jax.lax.stop_gradient`:

```{code-cell}
def f(x):
  return jnp.round(x)  # non-differentiable

def straight_through_f(x):
  # Create an exactly-zero expression with Sterbenz lemma that has
  # an exactly-one gradient.
  zero = x - jax.lax.stop_gradient(x)
  return zero + jax.lax.stop_gradient(f(x))

print("f(x): ", f(3.2))
print("straight_through_f(x):", straight_through_f(3.2))

print("grad(f)(x):", jax.grad(f)(3.2))
print("grad(straight_through_f)(x):", jax.grad(straight_through_f)(3.2))
```

### Per-example gradients

While most ML systems compute gradients and updates from batches of data, for reasons of computational efficiency and/or variance reduction, it is sometimes necessary to have access to the gradient/update associated with each specific sample in the batch.

For instance, this is needed to prioritize data based on gradient magnitude, or to apply clipping / normalisations on a sample by sample basis.

In many frameworks (PyTorch, TF, Theano) it is often not trivial to compute per-example gradients, because the library directly accumulates the gradient over the batch. Naive workarounds, such as computing a separate loss per example and then aggregating the resulting gradients are typically very inefficient.

In JAX, you can define the code to compute the gradient per-sample in an easy but efficient way.

Just combine the {func}`jax.jit`, {func}`jax.vmap` and {func}`jax.grad` transformations together:

```{code-cell}
perex_grads = jax.jit(jax.vmap(jax.grad(td_loss), in_axes=(None, 0, 0, 0)))

# Test it:
batched_s_tm1 = jnp.stack([s_tm1, s_tm1])
batched_r_t = jnp.stack([r_t, r_t])
batched_s_t = jnp.stack([s_t, s_t])

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

Let's go through this one transformation at a time.

First, you apply {func}`jax.grad` to `td_loss` to obtain a function that computes the gradient of the loss w.r.t. the parameters on single (unbatched) inputs:

```{code-cell}
dtdloss_dtheta = jax.grad(td_loss)

dtdloss_dtheta(theta, s_tm1, r_t, s_t)
```

This function computes one row of the array above.

Then, you vectorise this function using {func}`jax.vmap`. This adds a batch dimension to all inputs and outputs. Now, given a batch of inputs, you produce a batch of outputs — each output in the batch corresponds to the gradient for the corresponding member of the input batch.

```{code-cell}
almost_perex_grads = jax.vmap(dtdloss_dtheta)

batched_theta = jnp.stack([theta, theta])
almost_perex_grads(batched_theta, batched_s_tm1, batched_r_t, batched_s_t)
```

This isn't quite what we want, because we have to manually feed this function a batch of `theta`s, whereas we actually want to use a single `theta`. We fix this by adding `in_axes` to the {func}`jax.vmap`, specifying theta as `None`, and the other args as `0`. This makes the resulting function add an extra axis only to the other arguments, leaving `theta` unbatched, as we want:

```{code-cell}
inefficient_perex_grads = jax.vmap(dtdloss_dtheta, in_axes=(None, 0, 0, 0))

inefficient_perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

This does what we want, but is slower than it has to be. Now, you wrap the whole thing in a {func}`jax.jit` to get the compiled, efficient version of the same function:

```{code-cell}
perex_grads = jax.jit(inefficient_perex_grads)

perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t)
```

```{code-cell}
%timeit inefficient_perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t).block_until_ready()
%timeit perex_grads(theta, batched_s_tm1, batched_r_t, batched_s_t).block_until_ready()
```

### Hessian-vector products with `jax.grad`-of-`jax.grad`

One thing you can do with higher-order {func}`jax.vmap` is build a Hessian-vector product function. (Later on you'll write an even more efficient implementation that mixes both forward- and reverse-mode, but this one will use pure reverse-mode.)

A Hessian-vector product function can be useful in a [truncated Newton Conjugate-Gradient algorithm](https://en.wikipedia.org/wiki/Truncated_Newton_method) for minimizing smooth convex functions, or for studying the curvature of neural network training objectives (e.g. [1](https://arxiv.org/abs/1406.2572), [2](https://arxiv.org/abs/1811.07062), [3](https://arxiv.org/abs/1706.04454), [4](https://arxiv.org/abs/1802.03451)).

For a scalar-valued function $f : \mathbb{R}^n \to \mathbb{R}$ with continuous second derivatives (so that the Hessian matrix is symmetric), the Hessian at a point $x \in \mathbb{R}^n$ is written as $\partial^2 f(x)$. A Hessian-vector product function is then able to evaluate

$\qquad v \mapsto \partial^2 f(x) \cdot v$

for any $v \in \mathbb{R}^n$.

The trick is not to instantiate the full Hessian matrix: if $n$ is large, perhaps in the millions or billions in the context of neural networks, then that might be impossible to store.

Luckily, {func}`jax.vmap` already gives us a way to write an efficient Hessian-vector product function. You just have to use the identity:

$\qquad \partial^2 f (x) v = \partial [x \mapsto \partial f(x) \cdot v] = \partial g(x)$,

where $g(x) = \partial f(x) \cdot v$ is a new scalar-valued function that dots the gradient of $f$ at $x$ with the vector $v$. Notice that you're only ever differentiating scalar-valued functions of vector-valued arguments, which is exactly where you know {func}`jax.vmap` is efficient.

In JAX code, you can just write this:

```{code-cell}
def hvp(f, x, v):
    return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)
```

This example shows that you can freely use lexical closure, and JAX will never get perturbed or confused.

You will check this implementation a few cells down, once you learn how to compute dense Hessian matrices. You'll also write an even better version that uses both forward-mode and reverse-mode.


### Jacobians and Hessians using `jax.jacfwd` and `jax.jacrev`

You can compute full Jacobian matrices using the {func}`jax.jacfwd` and {func}`jax.jacrev` functions:

```{code-cell}
from jax import jacfwd, jacrev

# Define a sigmoid function.
def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

# Outputs probability of a label being true.
def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)

# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])

# Initialize random model coefficients
key, W_key, b_key = random.split(key, 3)
W = random.normal(W_key, (3,))
b = random.normal(b_key, ())

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

J = jacfwd(f)(W)
print("jacfwd result, with shape", J.shape)
print(J)

J = jacrev(f)(W)
print("jacrev result, with shape", J.shape)
print(J)
```

These two functions compute the same values (up to machine numerics), but differ in their implementation: {func}`jax.jacfwd` uses forward-mode automatic differentiation, which is more efficient for "tall" Jacobian matrices, while {func}`jax.jacrev` uses reverse-mode, which is more efficient for "wide" Jacobian matrices. For matrices that are near-square, {func}`jax.jacfwd` probably has an edge over {func}`jax.jacrev`.

You can also use {func}`jax.jacfwd` and {func}`jax.jacrev` with container types:

```{code-cell}
def predict_dict(params, inputs):
    return predict(params['W'], params['b'], inputs)

J_dict = jacrev(predict_dict)({'W': W, 'b': b}, inputs)
for k, v in J_dict.items():
    print("Jacobian from {} to logits is".format(k))
    print(v)
```

For more details on forward- and reverse-mode, as well as how to implement {func}`jax.jacfwd` and {func}`jax.jacrev` as efficiently as possible, read on!

Using a composition of two of these functions gives us a way to compute dense Hessian matrices:

```{code-cell}
def hessian(f):
    return jacfwd(jacrev(f))

H = hessian(f)(W)
print("hessian, with shape", H.shape)
print(H)
```

This shape makes sense: if you start with a function $f : \mathbb{R}^n \to \mathbb{R}^m$, then at a point $x \in \mathbb{R}^n$ you expect to get the shapes:

* $f(x) \in \mathbb{R}^m$, the value of $f$ at $x$,
* $\partial f(x) \in \mathbb{R}^{m \times n}$, the Jacobian matrix at $x$,
* $\partial^2 f(x) \in \mathbb{R}^{m \times n \times n}$, the Hessian at $x$,

and so on.

To implement `hessian`, you could have used `jacfwd(jacrev(f))` or `jacrev(jacfwd(f))` or any other composition of the two. But forward-over-reverse is typically the most efficient. That's because in the inner Jacobian computation we're often differentiating a function wide Jacobian (maybe like a loss function $f : \mathbb{R}^n \to \mathbb{R}$), while in the outer Jacobian computation we're differentiating a function with a square Jacobian (since $\nabla f : \mathbb{R}^n \to \mathbb{R}^n$), which is where forward-mode wins out.


## How it's made: Two foundational autodiff functions

### Jacobian-Vector products (JVPs, a.k.a. forward-mode autodiff)

JAX includes efficient and general implementations of both forward- and reverse-mode automatic differentiation. The familiar {func}`jax.vmap` function is built on reverse-mode, but to explain the difference between the two modes, and when each can be useful, you need a bit of math background.


#### JVPs in math

Mathematically, given a function $f : \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian of $f$ evaluated at an input point $x \in \mathbb{R}^n$, denoted $\partial f(x)$, is often thought of as a matrix in $\mathbb{R}^m \times \mathbb{R}^n$:

$\qquad \partial f(x) \in \mathbb{R}^{m \times n}$.

But you can also think of $\partial f(x)$ as a linear map, which maps the tangent space of the domain of $f$ at the point $x$ (which is just another copy of $\mathbb{R}^n$) to the tangent space of the codomain of $f$ at the point $f(x)$ (a copy of $\mathbb{R}^m$):

$\qquad \partial f(x) : \mathbb{R}^n \to \mathbb{R}^m$.

This map is called the [pushforward map](https://en.wikipedia.org/wiki/Pushforward_(differential)) of $f$ at $x$. The Jacobian matrix is just the matrix for this linear map on a standard basis.

If you don't commit to one specific input point $x$, then you can think of the function $\partial f$ as first taking an input point and returning the Jacobian linear map at that input point:

$\qquad \partial f : \mathbb{R}^n \to \mathbb{R}^n \to \mathbb{R}^m$.

In particular, you can uncurry things so that given input point $x \in \mathbb{R}^n$ and a tangent vector $v \in \mathbb{R}^n$, you get back an output tangent vector in $\mathbb{R}^m$. We call that mapping, from $(x, v)$ pairs to output tangent vectors, the *Jacobian-vector product*, and write it as:

$\qquad (x, v) \mapsto \partial f(x) v$


#### JVPs in JAX code

Back in Python code, JAX's {func}`jax.jvp` function models this transformation. Given a Python function that evaluates $f$, JAX's {func}`jax.jvp` is a way to get a Python function for evaluating $(x, v) \mapsto (f(x), \partial f(x) v)$.

```{code-cell}
from jax import jvp

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

key, subkey = random.split(key)
v = random.normal(subkey, W.shape)

# Push forward the vector `v` along `f` evaluated at `W`
y, u = jvp(f, (W,), (v,))
```

In terms of [Haskell-like type signatures](https://wiki.haskell.org/Type_signature), you could write:

```haskell
jvp :: (a -> b) -> a -> T a -> (b, T b)
```

where `T a` is used to denote the type of the tangent space for `a`.

In other words, `jvp` takes as arguments a function of type `a -> b`, a value of type `a`, and a tangent vector value of type `T a`. It gives back a pair consisting of a value of type `b` and an output tangent vector of type `T b`.

The `jvp`-transformed function is evaluated much like the original function, but paired up with each primal value of type `a` it pushes along tangent values of type `T a`. For each primitive numerical operation that the original function would have applied, the `jvp`-transformed function executes a "JVP rule" for that primitive that both evaluates the primitive on the primals and applies the primitive's JVP at those primal values.

That evaluation strategy has some immediate implications about computational complexity. Since we evaluate JVPs as we go, we don't need to store anything for later, and so the memory cost is independent of the depth of the computation. In addition, the FLOP cost of the `jvp`-transformed function is about 3x the cost of just evaluating the function (one unit of work for evaluating the original function, for example `sin(x)`; one unit for linearizing, like `cos(x)`; and one unit for applying the linearized function to a vector, like `cos_x * v`). Put another way, for a fixed primal point $x$, we can evaluate $v \mapsto \partial f(x) \cdot v$ for about the same marginal cost as evaluating $f$.

That memory complexity sounds pretty compelling! So why don't we see forward-mode very often in machine learning?

To answer that, first think about how you could use a JVP to build a full Jacobian matrix. If we apply a JVP to a one-hot tangent vector, it reveals one column of the Jacobian matrix, corresponding to the nonzero entry we fed in. So we can build a full Jacobian one column at a time, and to get each column costs about the same as one function evaluation. That will be efficient for functions with "tall" Jacobians, but inefficient for "wide" Jacobians.

If you're doing gradient-based optimization in machine learning, you probably want to minimize a loss function from parameters in $\mathbb{R}^n$ to a scalar loss value in $\mathbb{R}$. That means the Jacobian of this function is a very wide matrix: $\partial f(x) \in \mathbb{R}^{1 \times n}$, which we often identify with the Gradient vector $\nabla f(x) \in \mathbb{R}^n$. Building that matrix one column at a time, with each call taking a similar number of FLOPs to evaluate the original function, sure seems inefficient! In particular, for training neural networks, where $f$ is a training loss function and $n$ can be in the millions or billions, this approach just won't scale.

To do better for functions like this, you just need to use reverse-mode.


### Vector-Jacobian products (VJPs, a.k.a. reverse-mode autodiff)

Where forward-mode gives us back a function for evaluating Jacobian-vector products, which we can then use to build Jacobian matrices one column at a time, reverse-mode is a way to get back a function for evaluating vector-Jacobian products (equivalently Jacobian-transpose-vector products), which we can use to build Jacobian matrices one row at a time.


#### VJPs in math

Let's again consider a function $f : \mathbb{R}^n \to \mathbb{R}^m$.
Starting from our notation for JVPs, the notation for VJPs is pretty simple:

$\qquad (x, v) \mapsto v \partial f(x)$,

where $v$ is an element of the cotangent space of $f$ at $x$ (isomorphic to another copy of $\mathbb{R}^m$). When being rigorous, we should think of $v$ as a linear map $v : \mathbb{R}^m \to \mathbb{R}$, and when we write $v \partial f(x)$ we mean function composition $v \circ \partial f(x)$, where the types work out because $\partial f(x) : \mathbb{R}^n \to \mathbb{R}^m$. But in the common case we can identify $v$ with a vector in $\mathbb{R}^m$ and use the two almost interchangeably, just like we might sometimes flip between "column vectors" and "row vectors" without much comment.

With that identification, we can alternatively think of the linear part of a VJP as the transpose (or adjoint conjugate) of the linear part of a JVP:

$\qquad (x, v) \mapsto \partial f(x)^\mathsf{T} v$.

For a given point $x$, we can write the signature as

$\qquad \partial f(x)^\mathsf{T} : \mathbb{R}^m \to \mathbb{R}^n$.

The corresponding map on cotangent spaces is often called the [pullback](https://en.wikipedia.org/wiki/Pullback_(differential_geometry))
of $f$ at $x$. The key for our purposes is that it goes from something that looks like the output of $f$ to something that looks like the input of $f$, just like we might expect from a transposed linear function.

#### VJPs in JAX code

Switching from math back to Python, the JAX function `vjp` can take a Python function for evaluating $f$ and give us back a Python function for evaluating the VJP $(x, v) \mapsto (f(x), v^\mathsf{T} \partial f(x))$.

```{code-cell}
from jax import vjp

# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

y, vjp_fun = vjp(f, W)

key, subkey = random.split(key)
u = random.normal(subkey, y.shape)

# Pull back the covector `u` along `f` evaluated at `W`
v = vjp_fun(u)
```

In terms of [Haskell-like type signatures](https://wiki.haskell.org/Type_signature), we could write

```haskell
vjp :: (a -> b) -> a -> (b, CT b -> CT a)
```

where we use `CT a` to denote the type for the cotangent space for `a`. In words, `vjp` takes as arguments a function of type `a -> b` and a point of type `a`, and gives back a pair consisting of a value of type `b` and a linear map of type `CT b -> CT a`.

This is great because it lets us build Jacobian matrices one row at a time, and the FLOP cost for evaluating $(x, v) \mapsto (f(x), v^\mathsf{T} \partial f(x))$ is only about three times the cost of evaluating $f$. In particular, if we want the gradient of a function $f : \mathbb{R}^n \to \mathbb{R}$, we can do it in just one call. That's how {func}`jax.vmap` is efficient for gradient-based optimization, even for objectives like neural network training loss functions on millions or billions of parameters.

There's a cost, though: though the FLOPs are friendly, memory scales with the depth of the computation. Also, the implementation is traditionally more complex than that of forward-mode, though JAX has some tricks up its sleeve (that's a story for a future notebook!).

For more on how reverse-mode works, check out [this tutorial video from the Deep Learning Summer School in 2017](http://videolectures.net/deeplearning2017_johnson_automatic_differentiation/).


### Vector-valued gradients with VJPs

If you're interested in taking vector-valued gradients (like `tf.gradients`):

```{code-cell}
def vgrad(f, x):
  y, vjp_fn = vjp(f, x)
  return vjp_fn(jnp.ones(y.shape))[0]

print(vgrad(lambda x: 3*x**2, jnp.ones((2, 2))))
```

### Hessian-vector products using both forward- and reverse-mode

In a previous section, you implemented a Hessian-vector product function just using reverse-mode (assuming continuous second derivatives):

```{code-cell}
def hvp(f, x, v):
    return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)
```

That's efficient, but you can do even better and save some memory by using forward-mode together with reverse-mode.

Mathematically, given a function $f : \mathbb{R}^n \to \mathbb{R}$ to differentiate, a point $x \in \mathbb{R}^n$ at which to linearize the function, and a vector $v \in \mathbb{R}^n$, the Hessian-vector product function we want is:

$(x, v) \mapsto \partial^2 f(x) v$

Consider the helper function $g : \mathbb{R}^n \to \mathbb{R}^n$ defined to be the derivative (or gradient) of $f$, namely $g(x) = \partial f(x)$. All you need is its JVP, since that will give us:

$(x, v) \mapsto \partial g(x) v = \partial^2 f(x) v$.

We can translate that almost directly into code:

```{code-cell}
# forward-over-reverse
def hvp(f, primals, tangents):
  return jvp(grad(f), primals, tangents)[1]
```

Even better, since you didn't have to call {func}`jnp.dot` directly, this `hvp` function works with arrays of any shape and with arbitrary container types (like vectors stored as nested lists/dicts/tuples), and doesn't even have a dependence on {mod}`jax.numpy`.

Here's an example of how to use it:

```{code-cell}
def f(X):
  return jnp.sum(jnp.tanh(X)**2)

key, subkey1, subkey2 = random.split(key, 3)
X = random.normal(subkey1, (30, 40))
V = random.normal(subkey2, (30, 40))

ans1 = hvp(f, (X,), (V,))
ans2 = jnp.tensordot(hessian(f)(X), V, 2)

print(jnp.allclose(ans1, ans2, 1e-4, 1e-4))
```

Another way you might consider writing this is using reverse-over-forward:

```{code-cell}
# Reverse-over-forward
def hvp_revfwd(f, primals, tangents):
  g = lambda primals: jvp(f, primals, tangents)[1]
  return grad(g)(primals)
```

That's not quite as good, though, because forward-mode has less overhead than reverse-mode, and since the outer differentiation operator here has to differentiate a larger computation than the inner one, keeping forward-mode on the outside works best:

```{code-cell}
# Reverse-over-reverse, only works for single arguments
def hvp_revrev(f, primals, tangents):
  x, = primals
  v, = tangents
  return grad(lambda x: jnp.vdot(grad(f)(x), v))(x)


print("Forward over reverse")
%timeit -n10 -r3 hvp(f, (X,), (V,))
print("Reverse over forward")
%timeit -n10 -r3 hvp_revfwd(f, (X,), (V,))
print("Reverse over reverse")
%timeit -n10 -r3 hvp_revrev(f, (X,), (V,))

print("Naive full Hessian materialization")
%timeit -n10 -r3 jnp.tensordot(hessian(f)(X), V, 2)
```

## Composing VJPs, JVPs, and `jax.vmap`

### Jacobian-Matrix and Matrix-Jacobian products

Now that you have {func}`jax.jvp` and {func}`jax.vjp` transformations that give you functions to push-forward or pull-back single vectors at a time, you can use JAX's {func}`jax.vmap` [transformation](https://github.com/google/jax#auto-vectorization-with-vmap) to push and pull entire bases at once. In particular, you can use that to write fast matrix-Jacobian and Jacobian-matrix products:

```{code-cell}
# Isolate the function from the weight matrix to the predictions
f = lambda W: predict(W, b, inputs)

# Pull back the covectors `m_i` along `f`, evaluated at `W`, for all `i`.
# First, use a list comprehension to loop over rows in the matrix M.
def loop_mjp(f, x, M):
    y, vjp_fun = vjp(f, x)
    return jnp.vstack([vjp_fun(mi) for mi in M])

# Now, use vmap to build a computation that does a single fast matrix-matrix
# multiply, rather than an outer loop over vector-matrix multiplies.
def vmap_mjp(f, x, M):
    y, vjp_fun = vjp(f, x)
    outs, = vmap(vjp_fun)(M)
    return outs

key = random.PRNGKey(0)
num_covecs = 128
U = random.normal(key, (num_covecs,) + y.shape)

loop_vs = loop_mjp(f, W, M=U)
print('Non-vmapped Matrix-Jacobian product')
%timeit -n10 -r3 loop_mjp(f, W, M=U)

print('\nVmapped Matrix-Jacobian product')
vmap_vs = vmap_mjp(f, W, M=U)
%timeit -n10 -r3 vmap_mjp(f, W, M=U)

assert jnp.allclose(loop_vs, vmap_vs), 'Vmap and non-vmapped Matrix-Jacobian Products should be identical'
```

```{code-cell}
def loop_jmp(f, W, M):
    # jvp immediately returns the primal and tangent values as a tuple,
    # so we'll compute and select the tangents in a list comprehension
    return jnp.vstack([jvp(f, (W,), (mi,))[1] for mi in M])

def vmap_jmp(f, W, M):
    _jvp = lambda s: jvp(f, (W,), (s,))[1]
    return vmap(_jvp)(M)

num_vecs = 128
S = random.normal(key, (num_vecs,) + W.shape)

loop_vs = loop_jmp(f, W, M=S)
print('Non-vmapped Jacobian-Matrix product')
%timeit -n10 -r3 loop_jmp(f, W, M=S)
vmap_vs = vmap_jmp(f, W, M=S)
print('\nVmapped Jacobian-Matrix product')
%timeit -n10 -r3 vmap_jmp(f, W, M=S)

assert jnp.allclose(loop_vs, vmap_vs), 'Vmap and non-vmapped Jacobian-Matrix products should be identical'
```

### The implementation of `jax.jacfwd` and `jax.jacrev`

Now that we've seen fast Jacobian-matrix and matrix-Jacobian products, it's not hard to guess how to write {func}`jax.jacfwd` and {func}`jax.jacrev`. We just use the same technique to push-forward or pull-back an entire standard basis (isomorphic to an identity matrix) at once.

```{code-cell}
from jax import jacrev as builtin_jacrev

def our_jacrev(f):
    def jacfun(x):
        y, vjp_fun = vjp(f, x)
        # Use vmap to do a matrix-Jacobian product.
        # Here, the matrix is the Euclidean basis, so we get all
        # entries in the Jacobian at once. 
        J, = vmap(vjp_fun, in_axes=0)(jnp.eye(len(y)))
        return J
    return jacfun

assert jnp.allclose(builtin_jacrev(f)(W), our_jacrev(f)(W)), 'Incorrect reverse-mode Jacobian results!'
```

```{code-cell}
from jax import jacfwd as builtin_jacfwd

def our_jacfwd(f):
    def jacfun(x):
        _jvp = lambda s: jvp(f, (x,), (s,))[1]
        Jt =vmap(_jvp, in_axes=1)(jnp.eye(len(x)))
        return jnp.transpose(Jt)
    return jacfun

assert jnp.allclose(builtin_jacfwd(f)(W), our_jacfwd(f)(W)), 'Incorrect forward-mode Jacobian results!'
```

Interestingly, the [Autograd](https://github.com/hips/autograd) library couldn't do this. The [implementation](https://github.com/HIPS/autograd/blob/96a03f44da43cd7044c61ac945c483955deba957/autograd/differential_operators.py#L60) of reverse-mode `jacobian` in Autograd had to pull back one vector at a time with an outer-loop `map`. Pushing one vector at a time through the computation is much less efficient than batching it all together with {func}`jax.vmap`.

Another thing that Autograd couldn't do is {func}`jax.jit`. Interestingly, no matter how much Python dynamism you use in your function to be differentiated, we could always use {func}`jax.jit` on the linear part of the computation. For example:

```{code-cell}
def f(x):
    try:
        if x < 3:
            return 2 * x ** 3
        else:
            raise ValueError
    except ValueError:
        return jnp.pi * x

y, f_vjp = vjp(f, 4.)
print(jit(f_vjp)(1.))
```

## Complex numbers and differentiation

JAX is great at complex numbers and differentiation. To support both [holomorphic and non-holomorphic differentiation](https://en.wikipedia.org/wiki/Holomorphic_function), it helps to think in terms of JVPs and VJPs.

Consider a complex-to-complex function $f: \mathbb{C} \to \mathbb{C}$ and identify it with a corresponding function $g: \mathbb{R}^2 \to \mathbb{R}^2$,

```{code-cell}
def f(z):
  x, y = jnp.real(z), jnp.imag(z)
  return u(x, y) + v(x, y) * 1j

def g(x, y):
  return (u(x, y), v(x, y))
```

That is, we've decomposed $f(z) = u(x, y) + v(x, y) i$ where $z = x + y i$, and identified $\mathbb{C}$ with $\mathbb{R}^2$ to get $g$.

Since $g$ only involves real inputs and outputs, we already know how to write a Jacobian-vector product for it, say given a tangent vector $(c, d) \in \mathbb{R}^2$, namely:

$\begin{bmatrix} \partial_0 u(x, y) & \partial_1 u(x, y) \\ \partial_0 v(x, y) & \partial_1 v(x, y) \end{bmatrix}
\begin{bmatrix} c \\ d \end{bmatrix}$.

To get a JVP for the original function $f$ applied to a tangent vector $c + di \in \mathbb{C}$, we just use the same definition and identify the result as another complex number, 

$\partial f(x + y i)(c + d i) =
\begin{matrix} \begin{bmatrix} 1 & i \end{bmatrix} \\ ~ \end{matrix}
\begin{bmatrix} \partial_0 u(x, y) & \partial_1 u(x, y) \\ \partial_0 v(x, y) & \partial_1 v(x, y) \end{bmatrix}
\begin{bmatrix} c \\ d \end{bmatrix}$.

That's our definition of the JVP of a $\mathbb{C} \to \mathbb{C}$ function! Notice it doesn't matter whether or not $f$ is holomorphic: the JVP is unambiguous.

Here's a check:

```{code-cell}
def check(seed):
  key = random.PRNGKey(seed)

  # random coeffs for u and v
  key, subkey = random.split(key)
  a, b, c, d = random.uniform(subkey, (4,))

  def fun(z):
    x, y = jnp.real(z), jnp.imag(z)
    return u(x, y) + v(x, y) * 1j

  def u(x, y):
    return a * x + b * y

  def v(x, y):
    return c * x + d * y

  # primal point
  key, subkey = random.split(key)
  x, y = random.uniform(subkey, (2,))
  z = x + y * 1j

  # tangent vector
  key, subkey = random.split(key)
  c, d = random.uniform(subkey, (2,))
  z_dot = c + d * 1j

  # check jvp
  _, ans = jvp(fun, (z,), (z_dot,))
  expected = (grad(u, 0)(x, y) * c +
              grad(u, 1)(x, y) * d +
              grad(v, 0)(x, y) * c * 1j+
              grad(v, 1)(x, y) * d * 1j)
  print(jnp.allclose(ans, expected))
```

```{code-cell}
check(0)
check(1)
check(2)
```

What about VJPs? We do something pretty similar: for a cotangent vector $c + di \in \mathbb{C}$ we define the VJP of $f$ as

$(c + di)^* \; \partial f(x + y i) =
\begin{matrix} \begin{bmatrix} c & -d \end{bmatrix} \\ ~ \end{matrix}
\begin{bmatrix} \partial_0 u(x, y) & \partial_1 u(x, y) \\ \partial_0 v(x, y) & \partial_1 v(x, y) \end{bmatrix}
\begin{bmatrix} 1 \\ -i \end{bmatrix}$.

What's with the negatives? They're just to take care of complex conjugation, and the fact that we're working with covectors.

Here's a check of the VJP rules:

```{code-cell}
def check(seed):
  key = random.PRNGKey(seed)

  # random coeffs for u and v
  key, subkey = random.split(key)
  a, b, c, d = random.uniform(subkey, (4,))

  def fun(z):
    x, y = jnp.real(z), jnp.imag(z)
    return u(x, y) + v(x, y) * 1j

  def u(x, y):
    return a * x + b * y

  def v(x, y):
    return c * x + d * y

  # primal point
  key, subkey = random.split(key)
  x, y = random.uniform(subkey, (2,))
  z = x + y * 1j

  # cotangent vector
  key, subkey = random.split(key)
  c, d = random.uniform(subkey, (2,))
  z_bar = jnp.array(c + d * 1j)  # for dtype control

  # check vjp
  _, fun_vjp = vjp(fun, z)
  ans, = fun_vjp(z_bar)
  expected = (grad(u, 0)(x, y) * c +
              grad(v, 0)(x, y) * (-d) +
              grad(u, 1)(x, y) * c * (-1j) +
              grad(v, 1)(x, y) * (-d) * (-1j))
  assert jnp.allclose(ans, expected, atol=1e-5, rtol=1e-5)
```

```{code-cell}
check(0)
check(1)
check(2)
```

What about convenience wrappers like {func}`jax.grad`, {func}`jax.jacfwd`, and {func}`jax.jacrev`?

For $\mathbb{R} \to \mathbb{R}$ functions, recall we defined `grad(f)(x)` as being `vjp(f, x)[1](1.0)`, which works because applying a VJP to a `1.0` value reveals the gradient (i.e. Jacobian, or derivative). We can do the same thing for $\mathbb{C} \to \mathbb{R}$ functions: we can still use `1.0` as the cotangent vector, and we just get out a complex number result summarizing the full Jacobian:

```{code-cell}
def f(z):
  x, y = jnp.real(z), jnp.imag(z)
  return x**2 + y**2

z = 3. + 4j
grad(f)(z)
```

For general $\mathbb{C} \to \mathbb{C}$ functions, the Jacobian has 4 real-valued degrees of freedom (as in the 2x2 Jacobian matrices above), so we can't hope to represent all of them within a complex number. But we can for holomorphic functions! A holomorphic function is precisely a $\mathbb{C} \to \mathbb{C}$ function with the special property that its derivative can be represented as a single complex number. (The [Cauchy-Riemann equations](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations) ensure that the above 2x2 Jacobians have the special form of a scale-and-rotate matrix in the complex plane, i.e. the action of a single complex number under multiplication.) And we can reveal that one complex number using a single call to `vjp` with a covector of `1.0`.

Because this only works for holomorphic functions, to use this trick we need to promise JAX that our function is holomorphic; otherwise, JAX will raise an error when {func}`jax.grad` is used for a complex-output function:

```{code-cell}
def f(z):
  return jnp.sin(z)

z = 3. + 4j
grad(f, holomorphic=True)(z)
```

All the `holomorphic=True` promise does is disable the error when the output is complex-valued. We can still write `holomorphic=True` when the function isn't holomorphic, but the answer we get out won't represent the full Jacobian. Instead, it'll be the Jacobian of the function where we just discard the imaginary part of the output:

```{code-cell}
def f(z):
  return jnp.conjugate(z)

z = 3. + 4j
grad(f, holomorphic=True)(z)  # f is not actually holomorphic!
```

There are some useful upshots for how {func}`jax.grad` works here:

1. We can use {func}`jax.grad` on holomorphic $\mathbb{C} \to \mathbb{C}$ functions.
2. We can use {func}`jax.grad` to optimize $f : \mathbb{C} \to \mathbb{R}$ functions, like real-valued loss functions of complex parameters `x`, by taking steps in the direction of the conjugate of `grad(f)(x)`.
3. If we have an $\mathbb{R} \to \mathbb{R}$ function that just happens to use some complex-valued operations internally (some of which must be non-holomorphic, e.g. FFTs used in convolutions) then {func}`jax.grad` still works and we get the same result that an implementation using only real values would have given.

In any case, JVPs and VJPs are always unambiguous. And if we wanted to compute the full Jacobian matrix of a non-holomorphic $\mathbb{C} \to \mathbb{C}$ function, we can do it with JVPs or VJPs!


You should expect complex numbers to work everywhere in JAX. Here's differentiating through a Cholesky decomposition of a complex matrix:

```{code-cell}
A = jnp.array([[5.,    2.+3j,    5j],
              [2.-3j,   7.,  1.+7j],
              [-5j,  1.-7j,    12.]])

def f(X):
    L = jnp.linalg.cholesky(X)
    return jnp.sum((L - jnp.sin(L))**2)

grad(f, holomorphic=True)(A)
```

(advanced-autodiff-custom-derivative-rules)=
## Custom derivative rules for JAX-transformable Python functions

There are two ways to define differentiation rules in JAX:

1. Using {func}`jax.custom_jvp` and {func}`jax.custom_vjp` to define custom differentiation rules for Python functions that are already JAX-transformable; and
2. Defining new `core.Primitive` instances along with all their transformation rules, for example to call into functions from other systems like solvers, simulators, or general numerical computing systems.

This notebook is about #1. To read instead about #2, refer to the [notebook on adding primitives](https://jax.readthedocs.io/en/latest/notebooks/How_JAX_primitives_work.html).


### TL;DR: Custom JVPs with {func}`jax.custom_jvp`

```{code-cell}
from jax import custom_jvp

@custom_jvp
def f(x, y):
  return jnp.sin(x) * y

@f.defjvp
def f_jvp(primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = f(x, y)
  tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
  return primal_out, tangent_out
```

```{code-cell}
print(f(2., 3.))
y, y_dot = jvp(f, (2., 3.), (1., 0.))
print(y)
print(y_dot)
print(grad(f)(2., 3.))
```

```{code-cell}
# Equivalent alternative using the `defjvps` convenience wrapper

@custom_jvp
def f(x, y):
  return jnp.sin(x) * y

f.defjvps(lambda x_dot, primal_out, x, y: jnp.cos(x) * x_dot * y,
          lambda y_dot, primal_out, x, y: jnp.sin(x) * y_dot)
```

```{code-cell}
print(f(2., 3.))
y, y_dot = jvp(f, (2., 3.), (1., 0.))
print(y)
print(y_dot)
print(grad(f)(2., 3.))
```

### TL;DR: Custom VJPs with `jax.custom_vjp`

```{code-cell}
from jax import custom_vjp

@custom_vjp
def f(x, y):
  return jnp.sin(x) * y

def f_fwd(x, y):
# Returns primal output and residuals to be used in backward pass by `f_bwd`.
  return f(x, y), (jnp.cos(x), jnp.sin(x), y)

def f_bwd(res, g):
  cos_x, sin_x, y = res # Gets residuals computed in `f_fwd`
  return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)
```

```{code-cell}
print(grad(f)(2., 3.))
```

### Example problems

To get an idea of what problems {func}`jax.custom_jvp` and {func}`jax.custom_vjp` are meant to solve, let's go over a few examples. A more thorough introduction to the {func}`jax.custom_jvp` and {func}`jax.custom_vjp` APIs is in the next section.


#### Example: Numerical stability

One application of {func}`jax.custom_jvp` is to improve the numerical stability of differentiation.

Say we want to write a function called `log1pexp`, which computes $x \mapsto \log ( 1 + e^x )$. We can write that using `jax.numpy`:

```{code-cell}
def log1pexp(x):
  return jnp.log(1. + jnp.exp(x))

log1pexp(3.)
```

Since it's written in terms of `jax.numpy`, it's JAX-transformable:

```{code-cell}
print(jit(log1pexp)(3.))
print(jit(grad(log1pexp))(3.))
print(vmap(jit(grad(log1pexp)))(jnp.arange(3.)))
```

But there's a numerical stability problem lurking here:

```{code-cell}
print(grad(log1pexp)(100.))
```

That doesn't seem right! After all, the derivative of $x \mapsto \log (1 + e^x)$ is $x \mapsto \frac{e^x}{1 + e^x}$, and so for large values of $x$ we'd expect the value to be about 1.

We can get a bit more insight into what's going on by looking at the jaxpr for the gradient computation:

```{code-cell}
from jax import make_jaxpr

make_jaxpr(grad(log1pexp))(100.)
```

Stepping through how the jaxpr would be evaluated, notice that the last line would involve multiplying values that floating point math will round to 0 and $\infty$, respectively, which is never a good idea. That is, we're effectively evaluating `lambda x: (1 / (1 + jnp.exp(x))) * jnp.exp(x)` for large `x`, which effectively turns into `0. * jnp.inf`.

Instead of generating such large and small values, hoping for a cancellation that floats can't always provide, we'd rather just express the derivative function as a more numerically stable program. In particular, we can write a program that more closely evaluates the equal mathematical expression $1 - \frac{1}{1 + e^x}$, with no cancellation in sight.

This problem is interesting because even though our definition of `log1pexp` could already be JAX-differentiated (and transformed with {func}`jax.jit`, {func}`jax.vmap`, ...), we're not happy with the result of applying standard autodiff rules to the primitives comprising `log1pexp` and composing the result. Instead, we'd like to specify how the whole function `log1pexp` should be differentiated, as a unit, and thus arrange those exponentials better.

This is one application of custom derivative rules for Python functions that are already JAX transformable: specifying how a composite function should be differentiated, while still using its original Python definition for other transformations (like {func}`jax.jit`, {func}`jax.vmap`, ...).

Here's a solution using {func}`jax.custom_jvp`:

```{code-cell}
@custom_jvp
def log1pexp(x):
  return jnp.log(1. + jnp.exp(x))

@log1pexp.defjvp
def log1pexp_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  ans = log1pexp(x)
  ans_dot = (1 - 1/(1 + jnp.exp(x))) * x_dot
  return ans, ans_dot
```

```{code-cell}
print(grad(log1pexp)(100.))
```

```{code-cell}
print(jit(log1pexp)(3.))
print(jit(grad(log1pexp))(3.))
print(vmap(jit(grad(log1pexp)))(jnp.arange(3.)))
```

Here's a `defjvps` convenience wrapper to express the same thing:

```{code-cell}
@custom_jvp
def log1pexp(x):
  return jnp.log(1. + jnp.exp(x))

log1pexp.defjvps(lambda t, ans, x: (1 - 1/(1 + jnp.exp(x))) * t)
```

```{code-cell}
print(grad(log1pexp)(100.))
print(jit(log1pexp)(3.))
print(jit(grad(log1pexp))(3.))
print(vmap(jit(grad(log1pexp)))(jnp.arange(3.)))
```

#### Example: Enforcing a differentiation convention

A related application is to enforce a differentiation convention, perhaps at a boundary.

Consider the function $f : \mathbb{R}_+ \to \mathbb{R}_+$ with $f(x) = \frac{x}{1 + \sqrt{x}}$, where we take $\mathbb{R}_+ = [0, \infty)$. We might implement $f$ as a program like this:

```{code-cell}
def f(x):
  return x / (1 + jnp.sqrt(x))
```

As a mathematical function on $\mathbb{R}$ (the full real line), $f$ is not differentiable at zero (because the limit defining the derivative doesn't exist from the left). Correspondingly, autodiff produces a `nan` value:

```{code-cell}
print(grad(f)(0.))
```

But mathematically if we think of $f$ as a function on $\mathbb{R}_+$ then it is differentiable at 0 [Rudin's Principles of Mathematical Analysis Definition 5.1, or Tao's Analysis I 3rd ed. Definition 10.1.1 and Example 10.1.6]. Alternatively, we might say as a convention we want to consider the directional derivative from the right. So there is a sensible value for the Python function `grad(f)` to return at `0.0`, namely `1.0`. By default, JAX's machinery for differentiation assumes all functions are defined over $\mathbb{R}$ and thus doesn't produce `1.0` here.

We can use a custom JVP rule! In particular, we can define the JVP rule in terms of the derivative function $x \mapsto \frac{\sqrt{x} + 2}{2(\sqrt{x} + 1)^2}$ on $\mathbb{R}_+$,

```{code-cell}
@custom_jvp
def f(x):
  return x / (1 + jnp.sqrt(x))

@f.defjvp
def f_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  ans = f(x)
  ans_dot = ((jnp.sqrt(x) + 2) / (2 * (jnp.sqrt(x) + 1)**2)) * x_dot
  return ans, ans_dot
```

```{code-cell}
print(grad(f)(0.))
```

Here's the convenience wrapper version:

```{code-cell}
@custom_jvp
def f(x):
  return x / (1 + jnp.sqrt(x))

f.defjvps(lambda t, ans, x: ((jnp.sqrt(x) + 2) / (2 * (jnp.sqrt(x) + 1)**2)) * t)
```

```{code-cell}
print(grad(f)(0.))
```

#### Example: Gradient clipping

While in some cases we want to express a mathematical differentiation computation, in other cases we may even want to take a step away from mathematics to adjust the computation autodiff performs. One canonical example is reverse-mode gradient clipping.

For gradient clipping, we can use {func}`jnp.clip` together with a {func}`jax.custom_vjp` reverse-mode-only rule:

```{code-cell}
from functools import partial

@custom_vjp
def clip_gradient(lo, hi, x):
  return x  # identity function

def clip_gradient_fwd(lo, hi, x):
  return x, (lo, hi)  # save bounds as residuals

def clip_gradient_bwd(res, g):
  lo, hi = res
  return (None, None, jnp.clip(g, lo, hi))  # use None to indicate zero cotangents for lo and hi

clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)
```

```{code-cell}
import matplotlib.pyplot as plt

t = jnp.linspace(0, 10, 1000)

plt.plot(jnp.sin(t))
plt.plot(vmap(grad(jnp.sin))(t))
```

```{code-cell}
def clip_sin(x):
  x = clip_gradient(-0.75, 0.75, x)
  return jnp.sin(x)

plt.plot(clip_sin(t))
plt.plot(vmap(grad(clip_sin))(t))
```

#### Example: Python debugging

Another application that is motivated by development workflow rather than numerics is to set a `pdb` debugger trace in the backward pass of reverse-mode autodiff.

When trying to track down the source of a `nan` runtime error, or just examine carefully the cotangent (gradient) values being propagated, it can be useful to insert a debugger at a point in the backward pass that corresponds to a specific point in the primal computation. You can do that with {func}`jax.custom_vjp`.

We'll defer an example until the next section.



#### Example: Implicit function differentiation of iterative implementations

This example gets pretty deep in the mathematical weeds!

Another application for {func}`jax.custom_vjp` is reverse-mode differentiation of functions that are JAX-transformable (by {func}`jax.jit`, {func}`jax.vmap`, ...) but not efficiently JAX-differentiable for some reason, perhaps because they involve {func}`jax.lax.while_loop`. (It's not possible to produce an XLA HLO program that efficiently computes the reverse-mode derivative of an XLA HLO While loop because that would require a program with unbounded memory use, which isn't possible to express in XLA HLO, at least without "side-effecting" interactions through infeed/outfeed.)

For example, consider this `fixed_point` routine which computes a fixed point by iteratively applying a function in a `while_loop`:

```{code-cell}
from jax.lax import while_loop

def fixed_point(f, a, x_guess):
  def cond_fun(carry):
    x_prev, x = carry
    return jnp.abs(x_prev - x) > 1e-6

  def body_fun(carry):
    _, x = carry
    return x, f(a, x)

  _, x_star = while_loop(cond_fun, body_fun, (x_guess, f(a, x_guess)))
  return x_star
```

This is an iterative procedure for numerically solving the equation $x = f(a, x)$ for $x$, by iterating $x_{t+1} = f(a, x_t)$ until $x_{t+1}$ is sufficiently close to $x_t$. The result $x^*$ depends on the parameters $a$, and so we can think of there being a function $a \mapsto x^*(a)$ that is implicitly defined by equation $x = f(a, x)$.

We can use `fixed_point` to run iterative procedures to convergence, for example running Newton's method to calculate square roots while only executing adds, multiplies, and divides:

```{code-cell}
def newton_sqrt(a):
  update = lambda a, x: 0.5 * (x + a / x)
  return fixed_point(update, a, a)
```

```{code-cell}
print(newton_sqrt(2.))
```

We can {func}`jax.vmap` or {func}`jax.jit` the function as well:

```{code-cell}
print(jit(vmap(newton_sqrt))(jnp.array([1., 2., 3., 4.])))
```

We can't apply reverse-mode automatic differentiation because of the `while_loop`, but it turns out we wouldn't want to anyway: instead of differentiating through the implementation of `fixed_point` and all its iterations, we can exploit the mathematical structure to do something that is much more memory-efficient (and FLOP-efficient in this case, too!). We can instead use the implicit function theorem [Prop A.25 of Bertsekas's Nonlinear Programming, 2nd ed.], which guarantees (under some conditions) the existence of the mathematical objects we're about to use. In essence, we linearize the solution and solve those linear equations iteratively to compute the derivatives we want.

Consider again the equation $x = f(a, x)$ and the function $x^*$. We want to evaluate vector-Jacobian products like $v^\mathsf{T} \mapsto v^\mathsf{T} \partial x^*(a_0)$.

At least in an open neighborhood around the point $a_0$ at which we want to differentiate, let's assume that the equation $x^*(a) = f(a, x^*(a))$ holds for all $a$. Since the two sides are equal as functions of $a$, their derivatives must be equal as well, so let's differentiate both sides:

$\qquad \partial x^*(a) = \partial_0 f(a, x^*(a)) + \partial_1 f(a, x^*(a))  \partial x^*(a)$.

Setting $A = \partial_1 f(a_0, x^*(a_0))$ and $B = \partial_0 f(a_0, x^*(a_0))$, we can write the quantity we're after more simply as:

$\qquad \partial x^*(a_0) = B + A \partial x^*(a_0)$,

or, by rearranging,

$\qquad \partial x^*(a_0) = (I - A)^{-1} B$.

That means we can evaluate vector-Jacobian products, such as:

$\qquad v^\mathsf{T} \partial x^*(a_0) = v^\mathsf{T} (I - A)^{-1} B = w^\mathsf{T} B$,

where $w^\mathsf{T} = v^\mathsf{T} (I - A)^{-1}$, or equivalently $w^\mathsf{T} = v^\mathsf{T} + w^\mathsf{T} A$, or equivalently $w^\mathsf{T}$ is the fixed point of the map $u^\mathsf{T} \mapsto v^\mathsf{T} + u^\mathsf{T} A$. That last characterization gives us a way to write the VJP for `fixed_point` in terms of a call to `fixed_point`! Moreover, after expanding $A$ and $B$ back out, you can conclude you need only to evaluate VJPs of $f$ at $(a_0, x^*(a_0))$.

Here's the upshot:

```{code-cell}
@partial(custom_vjp, nondiff_argnums=(0,))
def fixed_point(f, a, x_guess):
  def cond_fun(carry):
    x_prev, x = carry
    return jnp.abs(x_prev - x) > 1e-6

  def body_fun(carry):
    _, x = carry
    return x, f(a, x)

  _, x_star = while_loop(cond_fun, body_fun, (x_guess, f(a, x_guess)))
  return x_star

def fixed_point_fwd(f, a, x_init):
  x_star = fixed_point(f, a, x_init)
  return x_star, (a, x_star)

def fixed_point_rev(f, res, x_star_bar):
  a, x_star = res
  _, vjp_a = vjp(lambda a: f(a, x_star), a)
  a_bar, = vjp_a(fixed_point(partial(rev_iter, f),
                             (a, x_star, x_star_bar),
                             x_star_bar))
  return a_bar, jnp.zeros_like(x_star)
  
def rev_iter(f, packed, u):
  a, x_star, x_star_bar = packed
  _, vjp_x = vjp(lambda x: f(a, x), x_star)
  return x_star_bar + vjp_x(u)[0]

fixed_point.defvjp(fixed_point_fwd, fixed_point_rev)
```

```{code-cell}
print(newton_sqrt(2.))
```

```{code-cell}
print(grad(newton_sqrt)(2.))
print(grad(grad(newton_sqrt))(2.))
```

We can check our answers by differentiating {func}`jnp.sqrt`, which uses a totally different implementation:

```{code-cell}
print(grad(jnp.sqrt)(2.))
print(grad(grad(jnp.sqrt))(2.))
```

A limitation to this approach is that the argument `f` can't close over any values involved in differentiation. That is, you might notice that we kept the parameter `a` explicit in the argument list of `fixed_point`. For this use case, consider using the low-level primitive `lax.custom_root`, which allows for derivatives in closed-over variables with custom root-finding functions.


### Basic usage of `jax.custom_jvp` and `jax.custom_vjp` APIs

#### Use `jax.custom_jvp` to define forward-mode (and, indirectly, reverse-mode) rules

Here's a canonical basic example of using {func}`jax.custom_jvp`, where the comments use
[Haskell-like type signatures](https://wiki.haskell.org/Type_signature):

```{code-cell}
# f :: a -> b
@custom_jvp
def f(x):
  return jnp.sin(x)

# f_jvp :: (a, T a) -> (b, T b)
def f_jvp(primals, tangents):
  x, = primals
  t, = tangents
  return f(x), jnp.cos(x) * t

f.defjvp(f_jvp)
```

```{code-cell}
print(f(3.))

y, y_dot = jvp(f, (3.,), (1.,))
print(y)
print(y_dot)
```

In other words, we start with a primal function `f` that takes inputs of type `a` and produces outputs of type `b`. We associate with it a JVP rule function `f_jvp` that takes a pair of inputs representing the primal inputs of type `a` and the corresponding tangent inputs of type `T a`, and produces a pair of outputs representing the primal outputs of type `b` and tangent outputs of type `T b`. The tangent outputs should be a linear function of the tangent inputs.

You can also use `f.defjvp` as a decorator, as in

```python
@custom_jvp
def f(x):
  ...

@f.defjvp
def f_jvp(primals, tangents):
  ...
```

Even though we defined only a JVP rule and no VJP rule, we can use both forward- and reverse-mode differentiation on `f`. JAX will automatically transpose the linear computation on tangent values from our custom JVP rule, computing the VJP as efficiently as if we had written the rule by hand:

```{code-cell}
print(grad(f)(3.))
print(grad(grad(f))(3.))
```

For automatic transposition to work, the JVP rule's output tangents must be linear as a function of the input tangents. Otherwise a transposition error is raised.

Multiple arguments work like this:

```{code-cell}
@custom_jvp
def f(x, y):
  return x ** 2 * y

@f.defjvp
def f_jvp(primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = f(x, y)
  tangent_out = 2 * x * y * x_dot + x ** 2 * y_dot
  return primal_out, tangent_out
```

```{code-cell}
print(grad(f)(2., 3.))
```

The `defjvps` convenience wrapper lets us define a JVP for each argument separately, and the results are computed separately then summed:

```{code-cell}
@custom_jvp
def f(x):
  return jnp.sin(x)

f.defjvps(lambda t, ans, x: jnp.cos(x) * t)
```

```{code-cell}
print(grad(f)(3.))
```

Here's a `defjvps` example with multiple arguments:

```{code-cell}
@custom_jvp
def f(x, y):
  return x ** 2 * y

f.defjvps(lambda x_dot, primal_out, x, y: 2 * x * y * x_dot,
          lambda y_dot, primal_out, x, y: x ** 2 * y_dot)
```

```{code-cell}
print(grad(f)(2., 3.))
print(grad(f, 0)(2., 3.))  # same as above
print(grad(f, 1)(2., 3.))
```

As a shorthand, with `defjvps` you can pass a `None` value to indicate that the JVP for a particular argument is zero:

```{code-cell}
@custom_jvp
def f(x, y):
  return x ** 2 * y

f.defjvps(lambda x_dot, primal_out, x, y: 2 * x * y * x_dot,
          None)
```

```{code-cell}
print(grad(f)(2., 3.))
print(grad(f, 0)(2., 3.))  # same as above
print(grad(f, 1)(2., 3.))
```

Calling a {func}`jax.custom_jvp` function with keyword arguments, or writing a {func}`jax.custom_jvp` function definition with default arguments, are both allowed so long as they can be unambiguously mapped to positional arguments based on the function signature retrieved by the standard library `inspect.signature` mechanism.

When you're not performing differentiation, the function `f` is called just as if it weren't decorated by {func}`jax.custom_jvp`:

```{code-cell}
@custom_jvp
def f(x):
  print('called f!')  # a harmless side-effect
  return jnp.sin(x)

@f.defjvp
def f_jvp(primals, tangents):
  print('called f_jvp!')  # a harmless side-effect
  x, = primals
  t, = tangents
  return f(x), jnp.cos(x) * t
```

```{code-cell}
print(f(3.))
```

```{code-cell}
print(vmap(f)(jnp.arange(3.)))
print(jit(f)(3.))
```

The custom JVP rule is invoked during differentiation, whether forward or reverse:

```{code-cell}
y, y_dot = jvp(f, (3.,), (1.,))
print(y_dot)
```

```{code-cell}
print(grad(f)(3.))
```

Notice that `f_jvp` calls `f` to compute the primal outputs. In the context of higher-order differentiation, each application of a differentiation transform will use the custom JVP rule if and only if the rule calls the original `f` to compute the primal outputs. (This represents a kind of fundamental tradeoff, where we can't make use of intermediate values from the evaluation of `f` in our rule _and also_ have the rule apply in all orders of higher-order differentiation.)

```{code-cell}
grad(grad(f))(3.)
```

You can use Python control flow with {func}`jax.custom_jvp`:

```{code-cell}
@custom_jvp
def f(x):
  if x > 0:
    return jnp.sin(x)
  else:
    return jnp.cos(x)

@f.defjvp
def f_jvp(primals, tangents):
  x, = primals
  x_dot, = tangents
  ans = f(x)
  if x > 0:
    return ans, 2 * x_dot
  else:
    return ans, 3 * x_dot
```

```{code-cell}
print(grad(f)(1.))
print(grad(f)(-1.))
```

#### Use `jax.custom_vjp` to define custom reverse-mode-only rules

While {func}`jax.custom_jvp` suffices for controlling both forward- and, via JAX's automatic transposition, reverse-mode differentiation behavior, in some cases we may want to directly control a VJP rule, for example in the latter two example problems presented above. We can do that with {func}`jax.custom_vjp`:

```{code-cell}
from jax import custom_vjp

# f :: a -> b
@custom_vjp
def f(x):
  return jnp.sin(x)

# f_fwd :: a -> (b, c)
def f_fwd(x):
  return f(x), jnp.cos(x)

# f_bwd :: (c, CT b) -> CT a
def f_bwd(cos_x, y_bar):
  return (cos_x * y_bar,)

f.defvjp(f_fwd, f_bwd)
```

```{code-cell}
print(f(3.))
print(grad(f)(3.))
```

In other words, we again start with a primal function `f` that takes inputs of type `a` and produces outputs of type `b`. We associate with it two functions, `f_fwd` and `f_bwd`, which describe how to perform the forward- and backward-passes of reverse-mode autodiff, respectively.

The function `f_fwd` describes the forward pass, not only the primal computation but also what values to save for use on the backward pass. Its input signature is just like that of the primal function `f`, in that it takes a primal input of type `a`. But as output it produces a pair, where the first element is the primal output `b` and the second element is any "residual" data of type `c` to be stored for use by the backward pass. (This second output is analogous to [PyTorch's save_for_backward mechanism](https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html).)

The function `f_bwd` describes the backward pass. It takes two inputs, where the first is the residual data of type `c` produced by `f_fwd` and the second is the output cotangents of type `CT b` corresponding to the output of the primal function. It produces an output of type `CT a` representing the cotangents corresponding to the input of the primal function. In particular, the output of `f_bwd` must be a sequence (e.g. a tuple) of length equal to the number of arguments to the primal function.

So multiple arguments work like this:

```{code-cell}
@custom_vjp
def f(x, y):
  return jnp.sin(x) * y

def f_fwd(x, y):
  return f(x, y), (jnp.cos(x), jnp.sin(x), y)

def f_bwd(res, g):
  cos_x, sin_x, y = res
  return (cos_x * g * y, sin_x * g)

f.defvjp(f_fwd, f_bwd)
```

```{code-cell}
print(grad(f)(2., 3.))
```

Calling a {func}`jax.custom_vjp` function with keyword arguments, or writing a {func}`jax.custom_vjp` function definition with default arguments, are both allowed so long as they can be unambiguously mapped to positional arguments based on the function signature retrieved by the standard library `inspect.signature` mechanism.

As with {func}`jax.custom_jvp`, the custom VJP rule composed of `f_fwd` and `f_bwd` is not invoked if differentiation is not applied. If the function is evaluated, or transformed with {func}`jax.jit`, {func}`jax.vmap`, or other non-differentiation transformations, then only `f` is called.

```{code-cell}
@custom_vjp
def f(x):
  print("called f!")
  return jnp.sin(x)

def f_fwd(x):
  print("called f_fwd!")
  return f(x), jnp.cos(x)

def f_bwd(cos_x, y_bar):
  print("called f_bwd!")
  return (cos_x * y_bar,)

f.defvjp(f_fwd, f_bwd)
```

```{code-cell}
print(f(3.))
```

```{code-cell}
print(grad(f)(3.))
```

```{code-cell}
y, f_vjp = vjp(f, 3.)
print(y)
```

```{code-cell}
print(f_vjp(1.))
```

**Forward-mode autodiff cannot be used on the** {func}`jax.custom_vjp` **function** and will raise an error:

```{code-cell}
:tags: [raises-exception]

from jax import jvp

try:
  jvp(f, (3.,), (1.,))
except TypeError as e:
  print('ERROR! {}'.format(e))
```

If you want to use both forward- and reverse-mode, use {func}`jax.custom_jvp` instead.

We can use {func}`jax.custom_vjp` together with `pdb` to insert a debugger trace in the backward pass:

```{code-cell}
import pdb

@custom_vjp
def debug(x):
  return x  # acts like identity

def debug_fwd(x):
  return x, x

def debug_bwd(x, g):
  import pdb; pdb.set_trace()
  return g

debug.defvjp(debug_fwd, debug_bwd)
```

```{code-cell}
def foo(x):
  y = x ** 2
  y = debug(y)  # insert pdb in corresponding backward pass step
  return jnp.sin(y)
```

```python
jax.grad(foo)(3.)

> <ipython-input-113-b19a2dc1abf7>(12)debug_bwd()
-> return g
(Pdb) p x
Array(9., dtype=float32)
(Pdb) p g
Array(-0.91113025, dtype=float32)
(Pdb) q
```


### More features and details

#### Working with `list` / `tuple` / `dict` containers (and other pytrees)

You should expect standard Python containers like lists, tuples, namedtuples, and dicts to just work, along with nested versions of those. In general, any [pytrees](https://jax.readthedocs.io/en/latest/pytrees.html) are permissible, so long as their structures are consistent according to the type constraints. 

Here's a contrived example with {func}`jax.custom_jvp`:

```{code-cell}
from collections import namedtuple
Point = namedtuple("Point", ["x", "y"])

@custom_jvp
def f(pt):
  x, y = pt.x, pt.y
  return {'a': x ** 2,
          'b': (jnp.sin(x), jnp.cos(y))}

@f.defjvp
def f_jvp(primals, tangents):
  pt, = primals
  pt_dot, =  tangents
  ans = f(pt)
  ans_dot = {'a': 2 * pt.x * pt_dot.x,
             'b': (jnp.cos(pt.x) * pt_dot.x, -jnp.sin(pt.y) * pt_dot.y)}
  return ans, ans_dot

def fun(pt):
  dct = f(pt)
  return dct['a'] + dct['b'][0]
```

```{code-cell}
pt = Point(1., 2.)

print(f(pt))
```

```{code-cell}
print(grad(fun)(pt))
```

And an analogous contrived example with {func}`jax.custom_vjp`:

```{code-cell}
@custom_vjp
def f(pt):
  x, y = pt.x, pt.y
  return {'a': x ** 2,
          'b': (jnp.sin(x), jnp.cos(y))}

def f_fwd(pt):
  return f(pt), pt

def f_bwd(pt, g):
  a_bar, (b0_bar, b1_bar) = g['a'], g['b']
  x_bar = 2 * pt.x * a_bar + jnp.cos(pt.x) * b0_bar
  y_bar = -jnp.sin(pt.y) * b1_bar
  return (Point(x_bar, y_bar),)

f.defvjp(f_fwd, f_bwd)

def fun(pt):
  dct = f(pt)
  return dct['a'] + dct['b'][0]
```

```{code-cell}
pt = Point(1., 2.)

print(f(pt))
```

```{code-cell}
print(grad(fun)(pt))
```

#### Handling  non-differentiable arguments

Some use cases, like the final example problem, call for non-differentiable arguments like function-valued arguments to be passed to functions with custom differentiation rules, and for those arguments to also be passed to the rules themselves. In the case of `fixed_point`, the function argument `f` was such a non-differentiable argument. A similar situation arises with `jax.experimental.odeint`.

##### `jax.custom_jvp` with `nondiff_argnums`

Use the optional `nondiff_argnums` parameter to {func}`jax.custom_jvp` to indicate arguments like these. Here's an example with {func}`jax.custom_jvp`:

```{code-cell}
from functools import partial

@partial(custom_jvp, nondiff_argnums=(0,))
def app(f, x):
  return f(x)

@app.defjvp
def app_jvp(f, primals, tangents):
  x, = primals
  x_dot, = tangents
  return f(x), 2. * x_dot
```

```{code-cell}
print(app(lambda x: x ** 3, 3.))
```

```{code-cell}
print(grad(app, 1)(lambda x: x ** 3, 3.))
```

Notice the gotcha here: no matter where in the argument list these parameters appear, they're placed at the *start* of the signature of the corresponding JVP rule. Here's another example:

```{code-cell}
@partial(custom_jvp, nondiff_argnums=(0, 2))
def app2(f, x, g):
  return f(g((x)))

@app2.defjvp
def app2_jvp(f, g, primals, tangents):
  x, = primals
  x_dot, = tangents
  return f(g(x)), 3. * x_dot
```

```{code-cell}
print(app2(lambda x: x ** 3, 3., lambda y: 5 * y))
```

```{code-cell}
print(grad(app2, 1)(lambda x: x ** 3, 3., lambda y: 5 * y))
```

##### `jax.custom_vjp` with `nondiff_argnums`

A similar option exists for {func}`jax.custom_vjp`, and, similarly, the convention is that the non-differentiable arguments are passed as the first arguments to the `_bwd` rule, no matter where they appear in the signature of the original function. The signature of the `_fwd` rule remains unchanged - it is the same as the signature of the primal function. Here's an example:

```{code-cell}
@partial(custom_vjp, nondiff_argnums=(0,))
def app(f, x):
  return f(x)

def app_fwd(f, x):
  return f(x), x

def app_bwd(f, x, g):
  return (5 * g,)

app.defvjp(app_fwd, app_bwd)
```

```{code-cell}
print(app(lambda x: x ** 2, 4.))
```

```{code-cell}
print(grad(app, 1)(lambda x: x ** 2, 4.))
```

Refer to `fixed_point` above for another usage example.

**You don't need to use** `nondiff_argnums` **with array-valued arguments**, such as, for example, ones with the integer dtype. Instead, `nondiff_argnums` should only be used for argument values that don't correspond to JAX types (essentially don't correspond to array types), like Python callables or strings. If JAX detects that an argument indicated by `nondiff_argnums` contains a JAX Tracer, then an error is raised. The `clip_gradient` function above is a good example of not using `nondiff_argnums` for integer-dtype array arguments.
s

## Next steps

There's a whole world of other autodiff tricks and functionality out there. Topics that weren't covered in this tutorial but can be worth pursuing include:

 - Gauss-Newton Vector Products, linearizing once
 - Custom VJPs and JVPs
 - Efficient derivatives at fixed-points
 - Estimating the trace of a Hessian using random Hessian-vector products
 - Forward-mode autodiff using only reverse-mode autodiff
 - Taking derivatives with respect to custom data types
 - Checkpointing (binomial checkpointing for efficient reverse-mode, not model snapshotting)
 - Optimizing VJPs with Jacobian pre-accumulation
