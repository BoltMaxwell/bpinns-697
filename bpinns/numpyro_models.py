"""
Numpyro models for Bayesian Physics-Informed Neural Networks (BPINNs).

Author: Maxwell Bolt
"""
__all__ = ["sample_weights", "bnn", "bpinn", "run_NUTS"]

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, HMC, NUTS
from jax import vmap, jit, lax
from typing import List, Tuple


def nonlin(x):
    """Applies a tanh nonlinearity to input x."""
    return jax.nn.tanh(x)

def sample_weights(layer_sizes: List[int], net_std: float = 2.0):
    """Samples weights and biases for a neural network with the given layer sizes."""
    weights = []
    biases = []

    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]

        w = numpyro.sample(f"w{i+1}", dist.Normal(jnp.zeros((in_size, out_size)), 
                                                  net_std * jnp.ones((in_size, out_size))))
        b = numpyro.sample(f"b{i+1}", dist.Normal(0., net_std), sample_shape=(out_size,))
        weights.append(w)
        biases.append(b)

    return weights, biases

def bnn(X, weights, biases):
    """
    BNN using numpyro. Takes a list of weights and biases.
    Follows the architecture of the paper from the innit docstring.
    """
    z = jnp.expand_dims(X, 0)  # Add an extra dimension to X

    for w, b in zip(weights[:-1], biases[:-1]):
        z = nonlin(jnp.matmul(z, w) + b)
    
    z = jnp.matmul(z, weights[-1]) + biases[-1]
    z = jnp.squeeze(z)
    return z

def bpinn(X, 
          Y, 
          num_collocation, 
          dynamics, 
          layers, 
          prior_params, 
          likelihood_params,
          key):

    N = X.shape[0]
    c_priorMean, k_priorMean, x0_priorMean, params_std, net_std = prior_params
    data_std, phys_std = likelihood_params

    weights, biases = sample_weights(layer_sizes=layers, net_std=net_std)

    X = X.squeeze()
    bnn_partial = jit(partial(bnn, weights=weights, biases=biases))
    bnn_vmap = vmap(bnn_partial, in_axes=0)
    data_pred = bnn_vmap(X)

    # log-normal priors on physics parameters
    log_c = numpyro.sample("log_c", dist.Normal(c_priorMean, params_std))
    log_k = numpyro.sample("log_k", dist.Normal(k_priorMean, params_std))
    log_x0 = numpyro.sample("log_x0", dist.Normal(x0_priorMean, params_std))

    c = jnp.exp(log_c)
    k = jnp.exp(log_k)
    x0 = jnp.exp(log_x0)
    min_value, max_value = jnp.min(X), jnp.max(X)
    collocation_pts = jr.uniform(key, (num_collocation,), minval=min_value, maxval=max_value)
    # The BNN is the function that is the 2nd argument to the dynamics function
    phys_pred = dynamics(collocation_pts, bnn_partial, (c, k, x0))
    
    # add dimension to data_pred
    data_pred = jnp.expand_dims(data_pred, 1)
    phys_pred = jnp.expand_dims(phys_pred, 1)
    
    # observe data
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(data_pred, data_std).to_event(1), obs=Y)

    # observe physics 
    with numpyro.plate("physics", num_collocation):
        numpyro.sample("phys", dist.Normal(phys_pred, phys_std).to_event(1), obs=0)


# helper function for HMC inference
def run_NUTS(model, 
             rng_key, 
             X, 
             Y, 
             num_collocation, 
             dynamics, 
             layers, 
             prior_params, 
             likelihood_params, 
             num_chains=1, num_warmup=1000, num_samples=1000):
    """
    runs NUTS on the numpyro model.

    Args:
        model: the numpyro model to run
        rng_key: the random key
        X: the input data
        Y: the output data
        num_collocation: the collocation points for the physics
        dynamics: the dynamics function
        layers: the size of the neural network
        prior_params: the prior parameters
        likelihood_params: the likelihood parameters
        num_chains: the number of chains to run
        num_warmup: the number of warmup steps
        num_samples: the number of samples to take
    """
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc_key, colloc_key = jr.split(rng_key)
    mcmc.run(mcmc_key, 
             X, 
             Y, 
             num_collocation, 
             dynamics, 
             layers, 
             prior_params, 
             likelihood_params,
             colloc_key)
    
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()
