"""
Numpyro models for Bayesian Physics-Informed Neural Networks (BPINNs).

Author: Maxwell Bolt
"""
__all__ = ["bnn", "sample_weights", "run_NUTS", "bnn_predict"]

import os
import time

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, HMC, NUTS

def nonlin(x):
    return jax.nn.tanh(x)

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

# helper function for HMC inference
def run_NUTS(model, rng_key, X, Y, width, net_std, data_std, phys_std, num_collocation,
             num_chains=1, num_warmup=1000, num_samples=1000):
    """
    runs NUTS on the numpyro model
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
    mcmc.run(rng_key, X, Y, width, net_std, data_std, phys_std, num_collocation)
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

# helper function for prediction
def bnn_predict(model, rng_key, samples, X, width, net_std, data_std, phys_std, num_collocation):
    """
    Predicts the output of the model given samples from the posterior.
    """
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, 
                                                  width=width, 
                                                  net_std=net_std, 
                                                  data_std=data_std,
                                                  phys_std=phys_std,
                                                  num_collocation=num_collocation)

    ## I THNK WE WILL ADD LINES HERE TO SAMPLE THE PHYSICS PARAMETERS

    return model_trace["Y"]["value"]