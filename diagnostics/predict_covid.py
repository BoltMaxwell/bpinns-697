"""
Performs diagnostics on a Bayesian Physics Informed Neural Network COVID-19 data.
The physics governing the dynamics of the system is a spring-mass-damper system.

Author: Maxwell Bolt
"""
import sys
sys.path.append('.')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import pickle

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import grad, jit, vmap
import numpyro
import numpyro.distributions as dist
import blackjax

from bpinns.dynamics import smd_dynamics
from preprocessing.process_covid import process_covid_data
import bpinns.numpyro_models as models
import bpinns.numpyro_infer as infer

## Hyperparameters
hyperparams = np.load('results/model_hyperparams.npy', allow_pickle=True)

with open('results/samples.pkl', 'rb') as f:
    samples = pickle.load(f)

print(samples.keys())
log_c = samples['log_c']
log_k = samples['log_k']
log_x0 = samples['log_x0']

# unpack hyperparameters with dictionary keys
width = hyperparams[()]['width']
num_collocation = hyperparams[()]['num_collocation']
num_chains = hyperparams[()]['num_chains']
num_warmup = hyperparams[()]['num_warmup']
num_samples = hyperparams[()]['num_samples']
prior_params = hyperparams[()]['prior_params']
likelihood_params = hyperparams[()]['likelihood_params']

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
# normalize 
s_train_t = train_t / jnp.max(train_t)
train_x_mean = jnp.mean(train_x)
train_x_std = jnp.std(train_x)
s_train_x = (train_x - train_x_mean) / train_x_std
collocation_pts = collocation_pts / jnp.max(train_t)

rng_key_predict, rng_key_infer = jr.split(jr.PRNGKey(0))

vmap_args = (samples, jr.split(rng_key_predict, num_samples * num_chains))
predictions = vmap(lambda samples, key: infer.bpinn_predict(models.bpinn, 
                                                       key, 
                                                       samples, 
                                                       s_train_t,
                                                       collocation_pts,
                                                       smd_dynamics, 
                                                       width,
                                                       prior_params,
                                                       likelihood_params
                                                       ))(*vmap_args)

mean_pred = jnp.mean(predictions, axis=0)
pred_y = mean_pred * train_x_std + train_x_mean


# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(train_t, train_x, 'ro', label='Data')
plt.plot(train_t, smooth_cases, 'g-', label='Smoothed Data')
plt.plot(train_t, pred_y, 'b-', label='Prediction')
plt.legend()
sns.despine(trim=True)
plt.show()

params = [log_k, log_c, log_x0]
titles = ['Spring Constant', 'Damping Constant', 'Initial Position']

plt.figure(figsize=(12, 6))
# Loop over the parameters and their titles
for i, (param, title) in enumerate(zip(params, titles)):
    plt.subplot(1, 3, i + 1)
    exp_param = jnp.exp(param)
    plt.hist(exp_param, bins=50, density=True)
    plt.axvline(jnp.exp(jnp.mean(param)), color='r', linestyle='--')
    plt.title(f'{title}: {jnp.exp(jnp.mean(param)):.2f}')

plt.tight_layout()
plt.show()





