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

## Hyperparameters
width = 32
depth = 2
num_collocation = 1000
num_chains = 1
burn = 1000
max_iter = 2000
thinning_factor = 10

# Model Parameters
phys_std = 0.05
data_std = 0.05
c_priorMean = jnp.log(2.2)
k_priorMean = jnp.log(350.0)
x0_priorMean = jnp.log(0.56)
params_std = 0.5
net_std = 2.0
phys_init = (c_priorMean, k_priorMean, x0_priorMean, params_std)
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
    data = (s_train_t, s_train_x)
    indices = jr.randint(key, (batch_size,), minval=bounds[0], maxval=bounds[1])
    return jtu.tree_map(lambda x: x[indices], data)

def dataloader(data_size, colloc_size, key):
    key, subkey = jr.split(key)
    data=sample_data(data_size, subkey),
    collocation=sample_collocation(colloc_size, key)
    return Batch(data, collocation)


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
print(model)

theta_init, static = eqx.partition(model, eqx.is_array)

## DEFINE LOG PROBABILITIES
# mlp prior
def mlp_prior(position):
    theta = position[0]
    flattree = jnp.array(jax.flatten_util.ravel_pytree(theta)[0])
    return - 0.5 * jnp.linalg.norm(flattree) ** 2

# physics param prior
def phys_prior(position):
    phys_params = position[1]
    # this is ugly
    log_c = phys_params[0]
    log_k = phys_params[1]
    log_x0 = phys_params[2]
    params_std = phys_params[3]
    c_prior = jax.scipy.stats.norm.pdf(log_c, params_std)
    k_prior = jax.scipy.stats.norm.pdf(log_k, params_std)
    x0_prior = jax.scipy.stats.norm.pdf(log_x0, params_std)

    return c_prior + k_prior + x0_prior

# data likelihood
def data_like(position, batch):
    theta = position[0]
    vphi = vmap(eqx.combine(theta, static))
    # Not really sure why I have to specify batch like this, fix this
    t, x = batch[0]
    N = len(x)
    # square residual loss (0.05 is the data std)
    return -0.5*N*jnp.mean((vphi(t)-x)**2, axis=(0, 1))/(2*0.05)

# physcs likelihood - this is where my current bug is
def phys_like(position, batch):
    theta = position[0]
    phys_params = position[1]
    N = batch.shape[0]
    c = jnp.exp(phys_params[0])
    k = jnp.exp(phys_params[1])
    x0 = jnp.exp(phys_params[2])
    psi = eqx.combine(theta, static)

    phys_pred = smd_dynamics(batch, psi, (c, k, x0))
    
    return - 0.5*(N**2)*(jnp.mean(phys_pred**2)/(0.05**2))

# Bayes' rule (additive)
@eqx.filter_jit
def log_prob(position, batch):
    bayes = (mlp_prior(position)
            + phys_prior(position)
            + data_like(position, batch.data)
            + phys_like(position, batch.collocation)
            )
    
    return bayes


## DEFINE SAMPLING FUNCTION - SGLD
def run_sgld(log_prob, dataloader, key):

    key, subkey = jr.split(key)

    def learning_rate(i):
        return 1e-3 / (i + 0.1)**0.51
    
    def grad_estimate(model, batch):
        g = jtu.tree_map(lambda x: -x, grad(log_prob)(model, batch))
        return jtu.tree_map(partial(jnp.clip, a_min=-1e4, a_max=1e4), g)
    
    sgld = blackjax.sgld(grad_estimate)
    init_state = (theta_init, phys_init)
    position = sgld.init(init_state)
    temperature = 1.0
    sgld_step = eqx.filter_jit(sgld.step)
    init = (0, position, subkey)

    @eqx.filter_jit
    def sgld_update(carry, x):
        """One step of SGLD."""
        i, position, key = carry
        key, subkey = jr.split(key)
        batch = dataloader(key=subkey)
        lr = learning_rate(i)
        key, subkey = jr.split(key)
        return (i + 1, sgld_step(subkey, position, batch, lr, temperature), key), position
    
    return sgld_update, init

# Set SGLD hyperparams
key, subkey = jr.split(jr.PRNGKey(0))

# Run SGLD
post_sgld_scan, init = run_sgld(log_prob, 
                                partial(dataloader, data_size=10, colloc_size=10), 
                                subkey)
_, sgld_samples = lax.scan(post_sgld_scan, init, None, length=max_iter)
posterior_samples = jtu.tree_map(lambda x: x[burn::thinning_factor], sgld_samples)

print('Posterior Samples Taken!')

# place the hyperparameters in a dictionary
hyperparams = {'width': width,
               'depth': depth}

# save hyperparams and samples
np.save('results/blackjax_hyperparams', hyperparams)
with open('results/blackjax_samples.pkl', 'wb') as f:
    pickle.dump(posterior_samples, f)

print('Saved hyperparameters and samples to results/')

## Temporary Testing
import matplotlib.pyplot as plt
import seaborn as sns

net_post, phys_post = posterior_samples
def eval_ensemble(diff, static_model, t):
    pinn = eqx.combine(diff, static_model)
    return pinn(t)

# Need to separate the samples from each other
# print(net_post)
# Only the initial physics params are showing, rest are NaNs
# print(phys_post)
print(jnp.shape(s_train_t))
print(net_post)
post_predictive = vmap(eval_ensemble, (None, None, 0), 1)(net_post, static, s_train_t)
# posterior quantiles
pinn_05, pinn_50, pinn_95 = \
    jnp.quantile(
        post_predictive, 
        jnp.array([0.05, 0.5, 0.95]), 
        axis=0
    )

fig, ax = plt.subplots(figsize=(3, 2), dpi=150)
ax.plot(post_predictive, alpha=0.4, lw=0.2)
ax.set(xlabel=r'Iteration $\times 10^{}$'.format(int(jnp.round(jnp.log10(thinning_factor)))), ylabel=r'$\theta_i$', title='Posterior samples (trace plot)')
ax.axvline(burn, ls='--', color='k', lw=1)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
ax.annotate('End of warmup period', xy=(burn + 0.01*(x1 - x0), y0 + 0.9*(y1 - y0)), xycoords='data', fontsize=6)
sns.despine(trim=True);