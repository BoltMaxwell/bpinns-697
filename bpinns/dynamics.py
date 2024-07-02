"""
Functions for different dynamical systems.

Currently implemented:
- Spring-mass-damper system

Author: Maxwell Bolt
"""
__all__ = ['smd_dynamics']

# Import required packages
import jax.numpy as jnp
from jax import vmap, grad, jacfwd, jacrev

def smd_dynamics(t, fn, params):
    """ODE function for the spring-mass-damper system.
    m x'' + c x' + k x = 0.
    Used for Covid cases model.
    Args:
        t: time
        fn: function is BNN
        params: constants governing the system
    Returns:
        f: function

    """
    c, k, b = params

    x = vmap(fn, in_axes=0)(t)
    x_t = vmap(grad(fn), in_axes=0)(t)
    x_tt = vmap(jacfwd(jacrev(fn)), in_axes=0)(t)

    x_t = jnp.squeeze(x_t)
    x_tt = jnp.squeeze(x_tt)

    y = 1/k * x_tt + c/k * x_t + x - b
    return y