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
import equinox as eqx
import blackjax

from bpinns.dynamics import smd_dynamics
from preprocessing.process_covid import process_covid_data

data = np.loadtxt('data/covid_world.dat')
# This is the data that the original code uses for 2021
start_day = 350
end_day = 700

time, cases, smooth_cases = process_covid_data(data, start_day, end_day)

print("Time shape:", time.shape)
print("Cases shape:", cases.shape)
print("Smooth cases shape:", smooth_cases.shape)
