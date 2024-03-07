# Copyright 2018 The JAX Authors.
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

"""Automatic differentiation variational inference in Numpy and JAX.

This demo fits a Gaussian approximation to an intractable, unnormalized
density, by differentiating through a Monte Carlo estimate of the
variational evidence lower bound (ELBO)."""


from functools import partial
import matplotlib.pyplot as plt

from jax import jit, grad, vmap
from jax import random
from jax.example_libraries import optimizers
import jax.numpy as jnp
import jax.scipy.stats.norm as norm


# ========= Functions to define the evidence lower bound. =========

def diag_gaussian_sample(rng, mean, log_std):
    # Take a single sample from a diagonal multivariate Gaussian.
    return mean + jnp.exp(log_std) * random.normal(rng, mean.shape)

def diag_gaussian_logpdf(x, mean, log_std):
    # Evaluate a single point on a diagonal multivariate Gaussian.
    return jnp.sum(vmap(norm.logpdf)(x, mean, jnp.exp(log_std)))

def elbo(logprob, rng, mean, log_std):
    # Single-sample Monte Carlo estimate of the variational lower bound.
    sample = diag_gaussian_sample(rng, mean, log_std)
    return logprob(sample) - diag_gaussian_logpdf(sample, mean, log_std)

def batch_elbo(logprob, rng, params, num_samples):
    # Average over a batch of random samples.
    rngs = random.split(rng, num_samples)
    vectorized_elbo = vmap(partial(elbo, logprob), in_axes=(0, None, None))
    return jnp.mean(vectorized_elbo(rngs, *params))


# ========= Helper function for plotting. =========

@partial(jit, static_argnums=(0, 1, 2, 4))
def _mesh_eval(func, x_limits, y_limits, params, num_ticks):
    # Evaluate func on a 2D grid defined by x_limits and y_limits.
    x = jnp.linspace(*x_limits, num=num_ticks)
    y = jnp.linspace(*y_limits, num=num_ticks)
    X, Y = jnp.meshgrid(x, y)
    xy_vec = jnp.stack([X.ravel(), Y.ravel()]).T
    zs = vmap(func, in_axes=(0, None))(xy_vec, params)
    return X, Y, zs.reshape(X.shape)

def mesh_eval(func, x_limits, y_limits, params, num_ticks=101):
    return _mesh_eval(func, x_limits, y_limits, params, num_ticks)

# ========= Define an intractable unnormalized density =========

def funnel_log_density(params):
    return norm.logpdf(params[0], 0, jnp.exp(params[1])) + \
           norm.logpdf(params[1], 0, 1.35)


if __name__ == "__main__":
    num_samples = 40

    @jit
    def objective(params, t):
        rng = random.key(t)
        return -batch_elbo(funnel_log_density, rng, params, num_samples)

    # Set up figure.
    fig = plt.figure(figsize=(8,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.ion()
    plt.show(block=False)
    x_limits = (-2, 2)
    y_limits = (-4, 2)
    target_dist = lambda x, _: jnp.exp(funnel_log_density(x))
    approx_dist = lambda x, params: jnp.exp(diag_gaussian_logpdf(x, *params))

    def callback(params, t):
        print(f"Iteration {t} lower bound {objective(params, t)}")

        plt.cla()
        X, Y, Z = mesh_eval(target_dist, x_limits, y_limits, 1)
        ax.contour(X, Y, Z, cmap='summer')
        X, Y, Z = mesh_eval(approx_dist, x_limits, y_limits, params)
        ax.contour(X, Y, Z, cmap='winter')
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_yticks([])
        ax.set_xticks([])

        # Plot random samples from variational distribution.
        # Here we clone the rng used in computing the objective
        # so that we can show exactly the same samples.
        rngs = random.split(random.key(t), num_samples)
        samples = vmap(diag_gaussian_sample, in_axes=(0, None, None))(rngs, *params)
        ax.plot(samples[:, 0], samples[:, 1], 'b.')

        plt.draw()
        plt.pause(1.0/60.0)


    # Set up optimizer.
    D = 2
    init_mean = jnp.zeros(D)
    init_std  = jnp.zeros(D)
    init_params = (init_mean, init_std)
    opt_init, opt_update, get_params = optimizers.momentum(step_size=0.1, mass=0.9)
    opt_state = opt_init(init_params)

    @jit
    def update(i, opt_state):
        params = get_params(opt_state)
        gradient = grad(objective)(params, i)
        return opt_update(i, gradient, opt_state)


    # Main loop.
    print("Optimizing variational parameters...")
    for t in range(100):
        opt_state = update(t, opt_state)
        params = get_params(opt_state)
        callback(params, t)
    plt.show(block=True)
