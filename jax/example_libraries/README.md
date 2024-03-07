## Mini-libraries

JAX provides some small, experimental libraries for machine learning. These
libraries are in part about providing tools and in part about serving as
examples for how to build such libraries using JAX. Each one is only <300 source
lines of code, so take a look inside and adapt them as you need!

#### 👉 **Note**: each mini-library is meant to be an _inspiration_, but not a prescription.

To serve that purpose, it is best to keep their code samples minimal; so we
generally **will not merge PRs** adding new features. Instead, please send your
lovely pull requests and design ideas to more fully-featured libraries like
[Haiku](https://github.com/deepmind/dm-haiku),
[Flax](https://github.com/google/flax), or
[Trax](https://github.com/google/trax).


### Neural-net building with Stax

**Stax** is a functional neural network building library. The basic idea is that
a single layer or an entire network can be modeled as an `(init_fun, apply_fun)`
pair. The `init_fun` is used to initialize network parameters and the
`apply_fun` takes parameters and inputs to produce outputs. There are
constructor functions for common basic pairs, like `Conv` and `Relu`, and these
pairs can be composed in series using `stax.serial` or in parallel using
`stax.parallel`.

Here's an example:

```python
import jax.numpy as jnp
from jax import random
from jax.example_libraries import stax
from jax.example_libraries.stax import (
    Conv, Dense, MaxPool, Relu, Flatten, LogSoftmax)

# Use stax to set up network initialization and evaluation functions
net_init, net_apply = stax.serial(
    Conv(32, (3, 3), padding='SAME'), Relu,
    Conv(64, (3, 3), padding='SAME'), Relu,
    MaxPool((2, 2)), Flatten,
    Dense(128), Relu,
    Dense(10), LogSoftmax,
)

# Initialize parameters, not committing to a batch shape
rng = random.key(0)
in_shape = (-1, 28, 28, 1)
out_shape, net_params = net_init(rng, in_shape)

# Apply network to dummy inputs
inputs = jnp.zeros((128, 28, 28, 1))
predictions = net_apply(net_params, inputs)
```

### First-order optimization

The file `optimizers.py` contains a minimal optimization library focused on
stochastic first-order optimizers. Every optimizer is modeled as an
`(init_fun, update_fun, get_params)` triple of functions. The `init_fun` is used
to initialize the optimizer state, which could include things like momentum
variables, and the `update_fun` accepts a gradient and an optimizer state to
produce a new optimizer state. The `get_params` function extracts the current
iterate (i.e. the current parameters) from the optimizer state. The parameters
being optimized can be ndarrays or arbitrarily-nested list/tuple/dict
structures, so you can store your parameters however you'd like.

Here's an example, using `jit` to compile the whole update end-to-end:

```python
from jax.example_libraries import optimizers
from jax import jit, grad

# Define a simple squared-error loss
def loss(params, batch):
  inputs, targets = batch
  predictions = net_apply(params, inputs)
  return jnp.sum((predictions - targets)**2)

# Use optimizers to set optimizer initialization and update functions
opt_init, opt_update, get_params = optimizers.momentum(step_size=1e-3, mass=0.9)

# Define a compiled update step
@jit
def step(i, opt_state, batch):
  params = get_params(opt_state)
  g = grad(loss)(params, batch)
  return opt_update(i, g, opt_state)

# Dummy input data stream
data_generator = ((jnp.zeros((128, 28, 28, 1)), jnp.zeros((128, 10)))
                  for _ in range(10))

# Optimize parameters in a loop
opt_state = opt_init(net_params)
for i in range(10):
  opt_state = step(i, opt_state, next(data_generator))
net_params = get_params(opt_state)
```
