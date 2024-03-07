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

(automatic-differentiation)=
# Automatic differentiation

<!--* freshness: { reviewed: '2024-05-03' } *-->

In this section, you will learn about fundamental applications of automatic differentiation (autodiff) in JAX. JAX has a pretty general autodiff system.
Computing gradients is a critical part of modern machine learning methods, and this tutorial will walk you through a few introductory autodiff topics, such as:

- {ref}`automatic-differentiation-taking-gradients`
- {ref}`automatic-differentiation-linear logistic regression`
- {ref}`automatic-differentiation-nested-lists-tuples-and-dicts`
- {ref}`automatic-differentiation-evaluating-using-jax-value_and_grad`
- {ref}`automatic-differentiation-checking-against-numerical-differences`

Make sure to also check out the {ref}`advanced-autodiff` tutorial for more advanced topics.

While understanding how automatic differentiation works "under the hood" isn't crucial for using JAX in most contexts, you are encouraged to check out this quite accessible [video](https://www.youtube.com/watch?v=wG_nF1awSSY) to get a deeper sense of what's going on.

(automatic-differentiation-taking-gradients)=
## 1. Taking gradients with `jax.grad`

In JAX, you can differentiate a scalar-valued function with the {func}`jax.grad` transformation:

```{code-cell}
import jax
import jax.numpy as jnp
from jax import grad

grad_tanh = grad(jnp.tanh)
print(grad_tanh(2.0))
```

{func}`jax.grad` takes a function and returns a function. If you have a Python function `f` that evaluates the mathematical function $f$, then `jax.grad(f)` is a Python function that evaluates the mathematical function $\nabla f$. That means `grad(f)(x)` represents the value $\nabla f(x)$.

Since {func}`jax.grad` operates on functions, you can apply it to its own output to differentiate as many times as you like:

```{code-cell}
print(grad(grad(jnp.tanh))(2.0))
print(grad(grad(grad(jnp.tanh)))(2.0))
```

JAX's autodiff makes it easy to compute higher-order derivatives, because the functions that compute derivatives are themselves differentiable. Thus, higher-order derivatives are as easy as stacking transformations. This can be illustrated in the single-variable case:

The derivative of $f(x) = x^3 + 2x^2 - 3x + 1$ can be computed as:

```{code-cell}
f = lambda x: x**3 + 2*x**2 - 3*x + 1

dfdx = jax.grad(f)
```

The higher-order derivatives of $f$ are:

$$
\begin{array}{l}
f'(x) = 3x^2 + 4x -3\\
f''(x) = 6x + 4\\
f'''(x) = 6\\
f^{iv}(x) = 0
\end{array}
$$

Computing any of these in JAX is as easy as chaining the {func}`jax.grad` function:

```{code-cell}
d2fdx = jax.grad(dfdx)
d3fdx = jax.grad(d2fdx)
d4fdx = jax.grad(d3fdx)
```

Evaluating the above in $x=1$ would give you:

$$
\begin{array}{l}
f'(1) = 4\\
f''(1) = 10\\
f'''(1) = 6\\
f^{iv}(1) = 0
\end{array}
$$

Using JAX:

```{code-cell}
print(dfdx(1.))
print(d2fdx(1.))
print(d3fdx(1.))
print(d4fdx(1.))
```

(automatic-differentiation-linear logistic regression)=
## 2. Computing gradients in a linear logistic regression

The next example shows how to compute gradients with {func}`jax.grad` in a linear logistic regression model. First, the setup:

```{code-cell}
key = jax.random.key(0)

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
targets = jnp.array([True, True, False, True])

# Training loss is the negative log-likelihood of the training examples.
def loss(W, b):
  preds = predict(W, b, inputs)
  label_probs = preds * targets + (1 - preds) * (1 - targets)
  return -jnp.sum(jnp.log(label_probs))

# Initialize random model coefficients
key, W_key, b_key = jax.random.split(key, 3)
W = jax.random.normal(W_key, (3,))
b = jax.random.normal(b_key, ())
```

Use the {func}`jax.grad` function with its `argnums` argument to differentiate a function with respect to positional arguments.

```{code-cell}
# Differentiate `loss` with respect to the first positional argument:
W_grad = grad(loss, argnums=0)(W, b)
print(f'{W_grad=}')

# Since argnums=0 is the default, this does the same thing:
W_grad = grad(loss)(W, b)
print(f'{W_grad=}')

# But you can choose different values too, and drop the keyword:
b_grad = grad(loss, 1)(W, b)
print(f'{b_grad=}')

# Including tuple values
W_grad, b_grad = grad(loss, (0, 1))(W, b)
print(f'{W_grad=}')
print(f'{b_grad=}')
```

The {func}`jax.grad` API has a direct correspondence to the excellent notation in Spivak's classic *Calculus on Manifolds* (1965), also used in Sussman and Wisdom's [*Structure and Interpretation of Classical Mechanics*](https://mitpress.mit.edu/9780262028967/structure-and-interpretation-of-classical-mechanics) (2015) and their [*Functional Differential Geometry*](https://mitpress.mit.edu/9780262019347/functional-differential-geometry) (2013). Both books are open-access. See in particular the "Prologue" section of *Functional Differential Geometry* for a defense of this notation.

Essentially, when using the `argnums` argument, if `f` is a Python function for evaluating the mathematical function $f$, then the Python expression `jax.grad(f, i)` evaluates to a Python function for evaluating $\partial_i f$.


(automatic-differentiation-nested-lists-tuples-and-dicts)=
## 3. Differentiating with respect to nested lists, tuples, and dicts

Due to JAX's PyTree abstraction (see {ref}`working-with-pytrees`), differentiating with
respect to standard Python containers just works, so use tuples, lists, and dicts (and arbitrary nesting) however you like.

Continuing the previous example:

```{code-cell}
def loss2(params_dict):
    preds = predict(params_dict['W'], params_dict['b'], inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))

print(grad(loss2)({'W': W, 'b': b}))
```

You can create {ref}`pytrees-custom-pytree-nodes` to work with not just {func}`jax.grad` but other JAX transformations ({func}`jax.jit`, {func}`jax.vmap`, and so on).


(automatic-differentiation-evaluating-using-jax-value_and_grad)=
## 4. Evaluating a function and its gradient using `jax.value_and_grad`

Another convenient function is {func}`jax.value_and_grad` for efficiently computing both a function's value as well as its gradient's value in one pass.

Continuing the previous examples:

```{code-cell}
loss_value, Wb_grad = jax.value_and_grad(loss, (0, 1))(W, b)
print('loss value', loss_value)
print('loss value', loss(W, b))
```

(automatic-differentiation-checking-against-numerical-differences)=
## 5. Checking against numerical differences

A great thing about derivatives is that they're straightforward to check with finite differences.

Continuing the previous examples:

```{code-cell}
# Set a step size for finite differences calculations
eps = 1e-4

# Check b_grad with scalar finite differences
b_grad_numerical = (loss(W, b + eps / 2.) - loss(W, b - eps / 2.)) / eps
print('b_grad_numerical', b_grad_numerical)
print('b_grad_autodiff', grad(loss, 1)(W, b))

# Check W_grad with finite differences in a random direction
key, subkey = jax.random.split(key)
vec = jax.random.normal(subkey, W.shape)
unitvec = vec / jnp.sqrt(jnp.vdot(vec, vec))
W_grad_numerical = (loss(W + eps / 2. * unitvec, b) - loss(W - eps / 2. * unitvec, b)) / eps
print('W_dirderiv_numerical', W_grad_numerical)
print('W_dirderiv_autodiff', jnp.vdot(grad(loss)(W, b), unitvec))
```

JAX provides a simple convenience function that does essentially the same thing, but checks up to any order of differentiation that you like:

```{code-cell}
from jax.test_util import check_grads

check_grads(loss, (W, b), order=2)  # check up to 2nd order derivatives
```

## Next steps

The {ref}`advanced-autodiff` tutorial provides more advanced and detailed explanations of how the ideas covered in this document are implemented in the JAX backend. Some features, such as {ref}`advanced-autodiff-custom-derivative-rules`, depend on understanding advanced automatic differentiation, so do check out that section in the {ref}`advanced-autodiff` tutorial if you are interested.
