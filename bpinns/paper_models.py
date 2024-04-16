"""
Numpyro models for Bayesian Physics-Informed Neural Networks (BPINNs).

Author: Maxwell Bolt
"""
__all__ = ["bnn", "run_NUTS", "bnn_predict"]

import os
import time

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, HMC, NUTS

# the non-linearity we use in our neural network
def nonlin(x):
    return jax.nn.tanh(x)

def bnn(X, Y, width=32, net_std=2.0, D_Y=1,):
    """
    BNN using numpyro. 
    Follows the architecture of the paper from the innit docstring.
    """
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, width)), 
                                          net_std*jnp.ones((D_X, width))))
    b1 = numpyro.sample("b1", dist.Normal(0., net_std), sample_shape=(width,))
    assert w1.shape == (D_X, width)
    z1 = nonlin(jnp.matmul(X, w1) + b1)  # <= first layer of activations
    assert z1.shape == (N, width)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((width, width)), 
                                          net_std*jnp.ones((width, width))))
    b2 = numpyro.sample("b2", dist.Normal(0., net_std), sample_shape=(width,))
    assert w2.shape == (width, width)
    z2 = nonlin(jnp.matmul(z1, w2) + b2)  # <= second layer of activations
    assert z2.shape == (N, width)

    # sample final layer of weights and neural network output
    wf = numpyro.sample("wf", dist.Normal(jnp.zeros((width, D_Y)), 
                                          net_std*jnp.ones((width, D_Y))))
    bf = numpyro.sample("bf", dist.Normal(0., net_std), sample_shape=(D_Y,))
    assert wf.shape == (width, D_Y)
    zf = jnp.matmul(z2, wf) + bf  # <= output of the neural network
    assert zf.shape == (N, D_Y)

    if Y is not None:
        assert zf.shape == Y.shape

    raw_prior_obs = numpyro.sample("prior_obs", dist.Normal(0.0, 1.0))
    prec_obs = raw_prior_obs ** 2
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(zf, sigma_obs).to_event(1), obs=Y)


# helper function for HMC inference
def run_NUTS(model, rng_key, X, Y, width, net_std, num_chains=1, num_warmup=1000, num_samples=1000):
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
    mcmc.run(rng_key, X, Y, width, net_std)
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

# helper function for prediction
def bnn_predict(model, rng_key, samples, X, width, net_std):
    """
    Predicts the output of the model given samples from the posterior.
    """
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, width=width, net_std=net_std)
    return model_trace["Y"]["value"]