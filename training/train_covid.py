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
num_collocation = 1000
num_chains = 1
num_warmup = 100
num_samples = 100

# priors
phys_std = 0.05
data_std = 0.05
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

def bpinn(X, Y, width, net_std, data_std, phys_std, collocation_pts):

    N, D_X = X.shape
    num_collocation = collocation_pts.shape[0]

    print(collocation_pts.shape)

    w1, b1, w2, b2, wf, bf = pm.sample_weights(width=width, net_std=net_std)
    net_params = (w1, b1, w2, b2, wf, bf)

    X = X.squeeze()
    collocation_pts = collocation_pts.squeeze()

    bnn = partial(pm.bnn, net_params=net_params)
    bnn_vmap = vmap(bnn, in_axes=0)
    data_pred = bnn_vmap(X)

    # log-normal priors on physics parameters
    log_c = numpyro.sample("log_c", dist.Normal(c_priorMean, params_std))
    log_k = numpyro.sample("log_k", dist.Normal(k_priorMean, params_std))
    log_x0 = numpyro.sample("log_x0", dist.Normal(x0_priorMean, params_std))
    c = jnp.exp(log_c)
    k = jnp.exp(log_k)
    x0 = jnp.exp(log_x0)
    # The BNN is the function that is the second argument to the dynamics function
    phys_pred = smd_dynamics(collocation_pts, bnn, (c, k, x0))
    
    # add dimension to data_pred
    data_pred = jnp.expand_dims(data_pred, 1)
    phys_pred = jnp.expand_dims(phys_pred, 1)
    
    # observe data
    with numpyro.plate("data", N):
        numpyro.sample("Y", dist.Normal(data_pred, data_std).to_event(1), obs=Y)

    # observe physics
    with numpyro.plate("physics", num_collocation):
        numpyro.sample("phys", dist.Normal(phys_pred, phys_std).to_event(1), obs=0)

    # observe physics use numpyro.factor?


rng_key, rng_key_predict = jr.split(jr.PRNGKey(0))

samples = pm.run_NUTS(bpinn, 
                      rng_key, 
                      s_train_t, 
                      s_train_x, 
                      width, 
                      net_std,
                      data_std,
                      phys_std,
                      collocation_pts,
                      num_chains=num_chains, 
                      num_warmup=num_warmup, 
                      num_samples=num_samples)
print('Samples taken!')
vmap_args = (samples, jr.split(rng_key_predict, num_samples * num_chains))
predictions = vmap(lambda samples, key: pm.bnn_predict(bpinn, 
                                                       key, 
                                                       samples, 
                                                       s_train_t, 
                                                       width, 
                                                       net_std,
                                                       data_std,
                                                       phys_std,
                                                       collocation_pts
                                                       ))(*vmap_args)
# save the predictions
np.save('results/predictions.npy', predictions)
# jnp.save('results/samples.npz', samples)
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
