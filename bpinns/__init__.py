"""
Replicates B-PINNs from paper:
Bayesian Physics Informed Neural Networks for real-world nonlinear dynamical systems
https://www.sciencedirect.com/science/article/pii/S0045782522004327?ref=pdf_download&fr=RR-2&rr=874e5c0f9c1c68d0

This code is a reimplementation in Jax. The original code is available at:
https://github.com/LivingMatterLab/xPINNs

Author: Maxwell Bolt
"""

from .dynamics import *