# bpinns-697 project
Implementation of Bayesian PINNs in Jax for ME697 project

## Dependencies
- `jax`
- `numpyro`
- `blackjax`

## What is included here?
The follpwoing folders are included:
- `bpinns`: Contains the implementation of Bayesian PINNs
- `diagnostics`: Contains the code for predicting with the models
- `preprocessing`: Contains the code for preprocessing the data
- `training`: Contains the code for training the models

## Inspired by:
- https://github.com/LivingMatterLab/xPINNs
- https://github.com/PredictiveScienceLab/jifty/blob/wesleyjholt/sgld-1d-diffusion/examples/basics/forward-sgld-1d-heat-equation.ipynb

## TODO:
- Implement SGLD in blackjax
- Add HMCECS in numpyro
- Add blackjax IFT comparison
