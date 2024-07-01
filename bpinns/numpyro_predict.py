"""
Perform inference on the BPINN model using Numpyro.

Author: Maxwell Bolt
"""
__all__ = ["bpinn_predict", "bpinn_inferPhysics"]

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, HMC, NUTS
from jax import vmap


# helper function for prediction
def bpinn_predict(model, 
                  rng_key, 
                  samples,
                  X, 
                  num_collocation, 
                  dynamics, 
                  layers, 
                  prior_params, 
                  likelihood_params):
    """
    Predicts the output of the model given samples from the posterior.

    Args:
        model: the model to predict from
        rng_key: random number generator key
        samples: samples from the posterior
        X: the input data
        collocation_pts: the collocation points
        dynamics: the dynamics function
        layers: the size of the neural network
        prior_params: the prior parameters
        likelihood_params: the likelihood parameters

    Returns:
        Y: the predicted output
    """
    predict_key, colloc_key = jax.random.split(rng_key)
    model = handlers.substitute(handlers.seed(model, predict_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None,
                                                  dynamics=dynamics, 
                                                  layers=layers,
                                                  num_collocation=num_collocation,
                                                  prior_params=prior_params,
                                                  likelihood_params=likelihood_params,
                                                  key=colloc_key
                                                  )

    ## I THNK WE WILL ADD LINES HERE TO SAMPLE THE PHYSICS PARAMETERS
    return model_trace["Y"]["value"]

# helper function for prediction
def bpinn_inferPhysics(model, 
                  rng_key, 
                  samples,
                  X, 
                  collocation_pts, 
                  dynamics, 
                  layers, 
                  prior_params, 
                  likelihood_params,
                  colloc_key):
    """
    Returns the inferred physics parameters from the posterior.

    Args:
        model: the model to predict from
        rng_key: random number generator key
        samples: samples from the posterior
        X: the input data
        collocation_pts: the collocation points
        dynamics: the dynamics function
        layers: the size of the neural network
        prior_params: the prior parameters
        likelihood_params: the likelihood parameters

    Returns:
        c: the inferred c parameter
        k: the inferred k parameter
        x0: the inferred x0 parameter
    """
    predict_key, colloc_key = jax.random.split(rng_key)
    model = handlers.substitute(handlers.seed(model, predict_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, 
                                                  Y=None,
                                                  dynamics=dynamics, 
                                                  layers=layers,
                                                  collocation_pts=collocation_pts,
                                                  prior_params=prior_params,
                                                  likelihood_params=likelihood_params,
                                                  key=colloc_key)

    return model_trace["log_c"]["value"]