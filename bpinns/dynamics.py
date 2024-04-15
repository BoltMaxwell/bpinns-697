"""
Functions for different dynamical systems.

Currently implemented:
- Spring-mass-damper system

Author: Maxwell Bolt
"""
__all__ = ['smd_dynamics']

# Import required packages
from jax import grad

def smd_dynamics(t, fn, params):
    """ODE function for the spring-mass-damper system.
    m x'' + c x' + k x = 0.
    Used for Covid cases model.
    Args:
        t: time
        fn: function
        params: constants governing the system
    Returns:
        f: function

    """
    c, k, b = params
    x = fn(t)
    x_t = grad(fn)(t)
    x_tt = grad(grad(fn))(t)
    f = 1/k * x_tt + c/k * x_t + x - b

    return f