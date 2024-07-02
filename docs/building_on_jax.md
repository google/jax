# Building on JAX

<!--* freshness: { reviewed: '2024-05-03' } *-->

A great way to learn advanced JAX usage is to see how other libraries are using JAX,
both how they integrate the library into their API,
what functionality it adds mathematically,
and how it's used for computational speedup in other libraries.


Below are examples of how JAX's features can be used to define accelerated
computation across numerous domains and software packages.

## Gradient Computation
Easy gradient calculation is a key feature of JAX.
In the [JaxOpt library](https://github.com/google/jaxopt) value and grad is directly utilized for users in multiple optimization algorithms in [its source code](https://github.com/google/jaxopt/blob/main/jaxopt/_src/base.py#LL87C30-L87C44).

Similarly the same Dynamax Optax pairing mentioned above is an example of
gradients enabling estimation methods that were challenging historically
[Maximum Likelihood Expectation using Optax](https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/lgssm_learning.html).

## Computational Speedup on a Single Core across Multiple Devices
Models defined in JAX can then be compiled to enable single computation speedup through JIT compiling.
The same compiled code can then be sent to a CPU device,
to a GPU or TPU device for additional speedup,
typically with no additional changes needed.
This allows for a smooth workflow from development into production.
In Dynamax the computationally expensive portion of a Linear State Space Model solver has been [jitted](https://github.com/probml/dynamax/blob/main/dynamax/linear_gaussian_ssm/models.py#L579).
A more complex example comes from PyTensor which compiles a JAX function dynamically and then [jits the constructed function](https://github.com/pymc-devs/pytensor/blob/main/pytensor/link/jax/linker.py#L64).

## Single and Multi Computer Speedup Using Parallelization
Another benefit of JAX is the simplicity of parallelizing computation using
`pmap` and `vmap` function calls or decorators.
In Dynamax state space models are parallelized with a [VMAP decorator](https://github.com/probml/dynamax/blob/main/dynamax/linear_gaussian_ssm/parallel_inference.py#L89)
a practical example of this use case being multi object tracking.

## Incorporating JAX code into your, or your users, workflows
JAX is quite composable and can be used in multiple ways.
JAX can be used with a standalone pattern, where the user defines all the calculations themselves.
However other patterns, such as using libraries built on jax that provide specific functionality.
These can be libraries that define specific types of models,
such as Neural Networks or State Space models or others,
or provide specific functionality such as optimization.
Here are more specific examples of each pattern.

### Direct Usage
Jax can be directly imported and utilized to build models “from scratch” as shown across this website,
for example in [JAX Tutorials](https://jax.readthedocs.io/en/latest/tutorials.html)
or [Neural Network with JAX](https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html).
This may be the best option if you are unable to find prebuilt code
for your particular challenge, or if you're looking to reduce the number
of dependencies in your codebase.

### Composable Domain Specific Libraries with JAX exposed
Another common approach are packages that provide prebuilt functionality,
whether it be model definition, or computation of some type.
Combinations of these packages can then be mixed and matched for a full
end to end workflow where a model is defined and its parameters are estimated.

One example is [Flax](https://github.com/google/flax) which simplifies the construction of Neural Networks.
Flax is then typically paired with [Optax](https://github.com/deepmind/optax)
where Flax defines the neural network architecture
and Optax supplies the optimization & model-fitting capabilities.

Another is [Dynamax](https://github.com/probml/dynamax) which allows easy
definition of state space models.
With Dynamax parameters can be estimated using
[Maximum Likelihood using Optax](https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/lgssm_learning.html)
or full Bayesian Posterior can be estimating using [MCMC from Blackjax](https://probml.github.io/dynamax/notebooks/linear_gaussian_ssm/lgssm_hmc.html)

### JAX Totally Hidden from Users
Other libraries opt to completely wrap JAX in their model specific API.
An example is PyMC and [Pytensor](https://github.com/pymc-devs/pytensor),
in which a user may never “see” JAX directly
but instead wrapping [JAX functions](https://pytensor.readthedocs.io/en/latest/extending/creating_a_numba_jax_op.html)
with a PyMC specific API.
