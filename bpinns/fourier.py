__all__ = ["FourierEncoding"]


import equinox as eqx
import jax


class FourierEncoding(eqx.Module):
    r"""COPIED FROM JIFTY, USE JIFTY VERSION WHEN STABLE!
    
    Makes a Fourier encoding to be used in a neural network.
    
    It enables neural networks to learn high-frequency patterns in the input
    space.
    The idea follows [Tancik et al. 2020](https://arxiv.org/abs/2006.10739).
    Let $d$ be the dimension of the input space. The Fourier encoding is a
    function $f: \mathbb{R}^d \to \mathbb{R}^D$ that maps the input to a
    higher-dimensional space. The function is defined as

    $$
    f(\mathbf{x}) = \begin{bmatrix}
        \cos(\mathbf{B}\mathbf{x}) \\
        \sin(\mathbf{B}\mathbf{x})
    \end{bmatrix},
    $$

    where $\mathbf{B} \in \mathbb{R}^{D \times d}$ is a matrix of angular
    frequencies and the trigonometric functions are applied element-wise.
    Typically, the matrix $\mathbf{B}$ is initialized with random values
    sampled from a normal distribution with zero mean and variance $\sigma^2$.

    :param in_size: The dimension of the input space.
    :type in_size: int
    :param num_fourier_features: The number of Fourier features.
    :type num_fourier_features: int
    :param key: A PRNG key.
    :type key: jax.random.PRNGKey
    """
    B: jax.Array

    @property
    def num_fourier_features(self) -> int:
        """The number of Fourier features."""
        return self.B.shape[0]

    @property
    def in_size(self) -> int:
        """The dimension of the input space."""
        return self.B.shape[1]
    
    @property
    def out_size(self) -> int:
        """The dimension of the output space."""
        return self.B.shape[0] * 2

    def __init__(self, 
                 in_size: int, 
                 num_fourier_features: int, 
                 key: jax.random.PRNGKey, 
                 sigma: float = 1.0):
        """Initializes the Fourier encoding."""
        self.B = jax.random.normal(
            key, shape=(num_fourier_features, in_size),
            dtype=jax.numpy.float32) * sigma
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """Applies the Fourier encoding.
        
        :param x: The input. It has to be a 1D array of shape (in_size,).
            The function is not vectorized. You should vectorize
            explicitly if you want to apply the function to multiple inputs.
        :type x: jax.numpy.ndarray
        :return: The Fourier encoding of the input. This an array of shape
            (2 * num_fourier_features,).
        :rtype: jax.numpy.ndarray
        
        """
        return jax.numpy.concatenate(
            [jax.numpy.cos(jax.numpy.dot(self.B, x)),
             jax.numpy.sin(jax.numpy.dot(self.B, x))],
            axis=0)