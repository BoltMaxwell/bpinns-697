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
param_priors_dict = {
    "c_prior": dist.Normal(c_priorMean, params_std),
    "k_prior": dist.Normal(k_priorMean, params_std),
    "x0_prior": dist.Normal(x0_priorMean, params_std),
}

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

    return pm.bnn(X, Y, width, net_std)
#     # # output of physics
#     # phys_pred = smd_dynamics(X, Y, [c_priorMean, k_priorMean, x0_priorMean])
#     # numpyro.sample("obs", dist.Normal(bnn_pred, obs_std*jnp.ones_like(bnn_pred)), obs=Y)
#     # numpyro.sample("phys", dist.Normal(jnp.zeros_like([num_collocation, 1]), phys_std), obs=Y)

rng_key, rng_key_predict = jr.split(jr.PRNGKey(0))
print(train_x.shape, train_y.shape)
samples = pm.run_NUTS(bpinn, rng_key, train_x, train_y, width, net_std,
                                    num_chains=num_chains, num_warmup=num_warmup, num_samples=num_samples)
print('Samples taken!')
vmap_args = (samples, jr.split(rng_key_predict, num_samples * num_chains))
predictions = vmap(lambda samples, key: pm.bnn_predict(bpinn, key, samples, train_x, width, net_std))(*vmap_args)
mean_pred = jnp.mean(predictions, axis=0)
