"""
Trains a Bayesian Physics Informed Neural Network on COVID-19 data.
The model is built using Numpyro and JAX.
The physics governing the dynamics of the system is a spring-mass-damper system.

Author: Maxwell Bolt
"""
import sys
sys.path.append('.')
import numpy as np
import seaborn as sns
import pickle

import jax
import jax.numpy as jnp
import jax.random as jr

from bpinns.dynamics import smd_dynamics
from preprocessing.process_covid import process_covid_data
import bpinns.numpyro_models as models

## Hyperparameters
width = 32
num_collocation = 1000
num_chains = 1
num_warmup = 1000
num_samples = 1000

# Model Parameters
phys_std = 0.05
data_std = 0.05
c_priorMean = jnp.log(2.2)
k_priorMean = jnp.log(350.0)
x0_priorMean = jnp.log(0.56)
params_std = 0.5
net_std = 2.0
prior_params = (c_priorMean, k_priorMean, x0_priorMean, params_std, net_std)
likelihood_params = (data_std, phys_std)

## Process Data
data = np.loadtxt('data/covid_world.dat')
# This is the data that the original code uses for 2021
start_day = 350
end_day = 700
time, cases, smooth_cases = process_covid_data(data, start_day, end_day)

# ONCE OPERATIONAL: WE WILL SPLIT TO TRAIN AND TEST
train_t = time
train_x = cases
collocation_pts = jnp.linspace(min(train_t), max(train_t), num_collocation)

# Normalize 
s_train_t = train_t / jnp.max(train_t)
train_x_mean = jnp.mean(train_x)
train_x_std = jnp.std(train_x)
s_train_x = (train_x - train_x_mean) / train_x_std
collocation_pts = collocation_pts / jnp.max(train_t)

rng_key, rng_key_predict = jr.split(jr.PRNGKey(0))

## PARAMETRIZE FUNCTION

## FREEZE PARAMETERS

## DEFINE LOG PROBABILITIES

## DEFINE SAMPLING FUNCTION

# # place the hyperparameters in a dictionary
# hyperparams = {'width': width,
#                'depth': depth}

# # save hyperparams and samples
# np.save('results/blackjax_hyperparams', hyperparams)
# with open('results/blackjax_samples.pkl', 'wb') as f:
#     pickle.dump(samples, f)

# print('Saved hyperparameters and samples to results/')