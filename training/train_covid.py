"""
Trains a Bayesian Physics Informed Neural Network on COVID-19 data.
The physics governing the dynamics of the system is a spring-mass-damper system.

Author: Maxwell Bolt
"""
import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import jax.flatten_util as jfu
from jax import grad, jit, vmap
import numpyro
import numpyro.distributions as dist
import blackjax

from bpinns.dynamics import smd_dynamics
from preprocessing.process_covid import process_covid_data
import bpinns.paper_models as pm

## Hyperparameters
width = 32
data_std = 0.1
phys_std = 0.1
obs_std = 0.05
num_collocation = 1000
num_chains = 1
num_warmup = 300
num_samples = 100

# priors
c_priorMean = jnp.log(2.2)
k_priorMean = jnp.log(350.0)
x0_priorMean = jnp.log(0.56)
params_std = 0.5
net_std = 2.0
prior_params = (c_priorMean, k_priorMean, x0_priorMean, params_std)

## Process Data
data = np.loadtxt('data/covid_world.dat')
# This is the data that the original code uses for 2021
start_day = 350
end_day = 700
train_x, train_y, smooth_cases = process_covid_data(data, start_day, end_day)
train_x = jnp.array(train_x).reshape(-1, 1)
train_y = jnp.array(train_y).reshape(-1, 1)
smooth_cases = jnp.array(smooth_cases).reshape(-1, 1)

def bpinn(X, Y, width, net_std):

    N, D_X = X.shape
    # traditional BNN prediction
    zf = pm.bnn(X, Y, width, net_std)

    # log-normal priors on physics parameters
    log_c = numpyro.sample("log_c", dist.Normal(c_priorMean, params_std))
    log_k = numpyro.sample("log_k", dist.Normal(k_priorMean, params_std))
    log_x0 = numpyro.sample("log_x0", dist.Normal(x0_priorMean, params_std))
    c = jnp.exp(log_c)
    k = jnp.exp(log_k)
    x0 = jnp.exp(log_x0)

    ## Holdover from original code, may need it
    raw_prior_obs = numpyro.sample("prior_obs", dist.Normal(0.0, 1.0))
    prec_obs = raw_prior_obs ** 2
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(zf, sigma_obs).to_event(1), obs=Y)


rng_key, rng_key_predict = jr.split(jr.PRNGKey(0))
print(train_x.shape, train_y.shape)
samples = pm.run_NUTS(bpinn, 
                      rng_key, 
                      train_x, 
                      train_y, 
                      width, 
                      net_std, 
                      num_chains=num_chains, 
                      num_warmup=num_warmup, 
                      num_samples=num_samples)
print('Samples taken!')
vmap_args = (samples, jr.split(rng_key_predict, num_samples * num_chains))
predictions = vmap(lambda samples, key: pm.bnn_predict(bpinn, 
                                                       key, 
                                                       samples, 
                                                       train_x, 
                                                       width, 
                                                       net_std
                                                       ))(*vmap_args)
mean_pred = jnp.mean(predictions, axis=0)
