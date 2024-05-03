"""
Functions for different dynamical systems.

Currently implemented:
- Spring-mass-damper system

Author: Maxwell Bolt
"""
__all__ = ['smd_dynamics']

# Import required packages
from jax import grad, vmap

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
    fn_vmap = vmap(fn, in_axes=0)
    fn_grad = vmap(grad(fn), in_axes=0)
    fn_hess = vmap(grad(grad(fn)), in_axes=0)
    print(t.shape)
    x = fn_vmap(t)
    print(x.shape)
    x_t = fn_grad(t)
    print(x_t.shape)
    x_tt = fn_hess(t)
    f = 1/k * x_tt + c/k * x_t + x - b

    return f