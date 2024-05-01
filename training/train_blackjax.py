"""
Trains a Bayesian Physics Informed Neural Network on COVID-19 data.
The physics governing the dynamics of the system is a spring-mass-damper system.

The model is built using Equinox and Blackjax.
Adapted from Wesley's Jifty example seen here (private repo):
https://github.com/PredictiveScienceLab/jifty/blob/wesleyjholt/sgld-1d-diffusion/examples/basics/forward-sgld-1d-heat-equation.ipynb
Which recreates in jax example 1 from the paper:
https://www.sciencedirect.com/science/article/pii/S002199912300195X


Author: Maxwell Bolt
"""
import sys
sys.path.append('.')
import numpy as np
import seaborn as sns
import pickle
from functools import partial

import jax
from jax import grad, vmap, jit, lax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
from collections import namedtuple
import blackjax


from bpinns.dynamics import smd_dynamics
from preprocessing.process_covid import process_covid_data
import bpinns.numpyro_models as models

## Hyperparameters
width = 32
depth = 2
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
phys_params = (c_priorMean, k_priorMean, x0_priorMean, params_std)
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

# Normalize 
s_train_t = train_t / jnp.max(train_t)
train_x_mean = jnp.mean(train_x)
train_x_std = jnp.std(train_x)
s_train_x = (train_x - train_x_mean) / train_x_std
bounds = (min(s_train_t), max(s_train_t))

Batch = namedtuple("Batch", ["data", "collocation"])

def sample_collocation(batch_size, key):
    return jr.uniform(key, (batch_size, 1), minval=bounds[0], maxval=bounds[1])

def sample_data(batch_size, key):
    indices = jr.randit(key, (batch_size,), minval=bounds[0], maxval=bounds[1])
    return jtu.tree_map(lambda x: x[indices], (s_train_t, s_train_x))

def dataloader(data_batch_size, collocation_batch_size, key):
    key, subkey = jr.split(key)
    return Batch(
        data=sample_data(data_batch_size, subkey),
        collocation=sample_collocation(collocation_batch_size, key))


rng_key, rng_key_predict = jr.split(jr.PRNGKey(0))

# PARAMETRIZE FUNCTION
mlp = eqx.nn.MLP(in_size=s_train_t.shape[1], 
                 out_size='scalar', 
                 width_size=width, 
                 depth=depth, 
                 activation=jax.nn.tanh, 
                 key=rng_key)

# here is where we would add Fourier features
model = mlp

# model_diff = eqx.filter(model, eqx.is_array)
init_theta, static = eqx.partition(model, eqx.is_array)

## DEFINE LOG PROBABILITIES
# mlp prior
def mlp_prior(position):
    # this thetas needs to be checked to make sure its what we think it is
    thetas = position[0]
    flattree = jnp.array(jax.flatten_util.ravel_pytree(thetas)[0])
    return - 0.5 * jnp.linalg.norm(flattree) ** 2

# physics param prior
def param_prior(position):

    phys_params = position[1]
    # need to verify phys params are being called correctly
    phys_prior = jax.scipy.stats.norm.pdf(phys_params) # defaults to mean 1

    return phys_prior

# data likelihood
def data_like(model, batch):
    t, x = batch
    v_model = vmap(eqx.combine(init_theta, static))
# physcs likelihood

# Bayes' rule (additive)
@eqx.filter_jit
def log_prob(model_dff, batch):
    return param_prior(key) + data_like(init_theta, batch.data)


## DEFINE SAMPLING FUNCTION - SGLD?
def run_sgld(log_prob, dataloader, key):

    key, subkey = jr.split(key)

    def learning_rate(i):
        return 1e-3 / (i + 0.1)**0.51
    
    def grad_estimate(model, batch):
        g = jtu.tree_map(lambda x: -x, grad(log_prob)(model, batch))
        return jtu.tree_map(partial(jnp.clip, a_min=-1e4, a_max=1e4), g)
    
    sgld = blackjax.sgld(grad_estimate)
    position = sgld.init(init_theta)
    temperature = 1.0
    sgld_step = jit(sgld.step)
    init = (0, position, subkey)

    def sgld_update(carry, x):
        """One step of SGLD."""
        i, position, key = carry
        key, subkey = jr.split(key)
        batch = dataloader(key=subkey)
        lr = learning_rate(i)
        key, subkey = jr.split(key)
        return (i + 1, sgld_step(subkey, position, batch, lr, temperature), key), position
    
    return sgld_update, init

key, subkey = jr.split(jr.PRNGKey(0))
print(subkey.shape)
max_iter = 1000
thinning_factor = 10

posterior_sgld_scan, init = run_sgld(log_prob, 
                                     partial(sample_collocation, batch_size=1),
                                     subkey)
_, posterior_samples = lax.scan(posterior_sgld_scan, init, None, length=max_iter)
posterior_samples = jtu.tree_map(lambda x: x[::thinning_factor], posterior_samples)

print('Posterior Samples Taken!')

# place the hyperparameters in a dictionary
hyperparams = {'width': width,
               'depth': depth}

# save hyperparams and samples
np.save('results/blackjax_hyperparams', hyperparams)
# with open('results/blackjax_samples.pkl', 'wb') as f:
#     pickle.dump(samples, f)

# print('Saved hyperparameters and samples to results/')