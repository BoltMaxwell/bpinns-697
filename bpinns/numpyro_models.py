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
from jax import vmap


def nonlin(x):
    """Applies a tanh nonlinearity to input x."""
    return jax.nn.tanh(x)

def sample_weights(width=32, net_std=2.0):

    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((1, width)), 
                                          net_std*jnp.ones((1, width))))
    b1 = numpyro.sample("b1", dist.Normal(0., net_std), sample_shape=(width,))
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((width, width)), 
                                          net_std*jnp.ones((width, width))))
    b2 = numpyro.sample("b2", dist.Normal(0., net_std), sample_shape=(width,))
    wf = numpyro.sample("wf", dist.Normal(jnp.zeros((width, 1)), 
                                          net_std*jnp.ones((width, 1))))
    bf = numpyro.sample("bf", dist.Normal(0., net_std), sample_shape=(1,))

    return w1, b1, w2, b2, wf, bf

def bnn(X, net_params):
    """
    BNN using numpyro. 
    Follows the architecture of the paper from the innit docstring.
    """
    w1, b1, w2, b2, wf, bf = net_params

    # first layer of activations
    X = jnp.expand_dims(X, 0)  # Add an extra dimension to X
    z1 = nonlin(jnp.matmul(X, w1) + b1)
    z2 = nonlin(jnp.matmul(z1, w2) + b2)
    zf = jnp.matmul(z2, wf) + bf
    zf = jnp.squeeze(zf)

    return zf

def bpinn(X, 
          Y, 
          num_collocation, 
          dynamics, 
          width, 
          prior_params, 
          likelihood_params,
          key):

    N = X.shape[0]
    c_priorMean, k_priorMean, x0_priorMean, params_std, net_std = prior_params
    data_std, phys_std = likelihood_params

    w1, b1, w2, b2, wf, bf = sample_weights(width=width, net_std=net_std)
    net_params = (w1, b1, w2, b2, wf, bf)

    X = X.squeeze()
    min_value, max_value = jnp.min(X), jnp.max(X)
    collocation_pts = jr.uniform(key, (num_collocation,), minval=min_value, maxval=max_value)
    # collocation_pts = collocation_pts.squeeze()


    bnn_partial = partial(bnn, net_params=net_params)
    bnn_vmap = vmap(bnn_partial, in_axes=0)
    data_pred = bnn_vmap(X)

    # log-normal priors on physics parameters
    log_c = numpyro.sample("log_c", dist.Normal(c_priorMean, params_std))
    log_k = numpyro.sample("log_k", dist.Normal(k_priorMean, params_std))
    log_x0 = numpyro.sample("log_x0", dist.Normal(x0_priorMean, params_std))
    c = jnp.exp(log_c)
    k = jnp.exp(log_k)
    x0 = jnp.exp(log_x0)
    # The BNN is the function that is the 2nd argument to the dynamics function
    phys_pred = dynamics(collocation_pts, bnn_partial, (c, k, x0))
    
    # add dimension to data_pred
    data_pred = jnp.expand_dims(data_pred, 1)
    phys_pred = jnp.expand_dims(phys_pred, 1)
    
    # observe data
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(data_pred, data_std).to_event(1), obs=Y)

    # I think the plate is wrong.
    # observe physics use numpyro.factor, which is equivalent to multiplying the likelihood
    with numpyro.plate("physics", num_collocation):
        numpyro.sample("phys", dist.Normal(phys_pred, phys_std).to_event(1), obs=0)


# helper function for HMC inference
def run_NUTS(model, 
             rng_key, 
             X, 
             Y, 
             num_collocation, 
             dynamics, 
             width, 
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
        width: the width of the neural network
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
             width, 
             prior_params, 
             likelihood_params,
             colloc_key)
    
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()
