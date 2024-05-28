"""
Functions for different dynamical systems.

Currently implemented:
- Spring-mass-damper system

Author: Maxwell Bolt
"""
__all__ = ['smd_dynamics']

# Import required packages
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, jacfwd, jacrev

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
    ## MAKE SURE WE'RE TAKING GRADIENTS WRT t NOT PARAMS OF NN
    ## BIG QUESTION: 
    # does this mean I should be usingjax.grad or Jacobian/Hessian?
    # https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
    # The trick may lie in jnp.vdot to bring 1d vector to scalar

    c, k, b = params
    # x = vmap(fn, in_axes=0)(t)
    # print(x.shape)
    # fn_t = vmap(grad(fn), in_axes=0)
    # x_t = fn_t(t)
    # print(x_t.shape)
    # fn_tt = vmap(grad(grad(fn)), in_axes=0)
    # x_tt = fn_tt(t)
    # print(x_tt.shape)
    # y = 1/k * x_tt + c/k * x_t + x - b

    x = vmap(fn, in_axes=0)(t)
    x_t = vmap(grad(fn), in_axes=0)(t)
    x_tt = vmap(jacfwd(jacrev(fn)), in_axes=0)(t)

    x_t = jnp.squeeze(x_t)
    x_tt = jnp.squeeze(x_tt)
    # print(jnp.shape(x))
    # print(jnp.shape(x_t))
    # print(jnp.shape(x_tt))

    y = 1/k * x_tt + c/k * x_t + x - b

    return y