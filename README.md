# bpinns-697 project
Implementation of Bayesian PINNs in Jax for ME697 project

## Dependencies
- `jax`
- `numpyro`
- `blackjax`

## What is included here?
The following folders are included:
- `bpinns`: Contains the implementation of Bayesian PINNs
- `diagnostics`: Contains the code for predicting with the models
- `preprocessing`: Contains the code for preprocessing the data
- `training`: Contains the code for training the models


## TODO:
- Implement SGLD in blackjax
- Add Fourier Features to Equinox model
- Add HMCECS in numpyro
- Make numpyro network architecture flexible
- Add IFT comparison

## Paper References
- https://www.sciencedirect.com/science/article/pii/S0045782522004327?fr=RR-2&ref=pdf_download&rr=87dbe30d5c0de241

## Code References:
- https://github.com/LivingMatterLab/xPINNs
- https://github.com/PredictiveScienceLab/jifty/blob/wesleyjholt/sgld-1d-diffusion/examples/basics/forward-sgld-1d-heat-equation.ipynb
