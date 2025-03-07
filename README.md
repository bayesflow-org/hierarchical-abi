# Hierarchical ABI with conditional score matching

This repository contains the code for the paper __to be added__.

It is organized as follows:

The folder `diffusion_model` contains:
- `diffusion_model.py` contains the implementation of the diffusion model based on the SDE interpretation of [Song et al. (2021)](https://arxiv.org/abs/2011.13456). A flat and hierarchical version of the model is implemented.
- `diffusion_sampling.py` contains the implementation of the different sampling algorithms for the diffusion model. Each method allows for mini-batching by specifying the `n_scores_update` parameter.
  - Euler-Maruyama
  - Adaptive Euler-Maruyama with second order correction
  - Annealed Langevin Dynamics
  - ODE Probability Flow
- `train_score_models.py` contains the training routines for the flat and hierarchical score models.

The folder `problems` contains two example problems:
- `gaussian_flat.py` contains the implementation of a flat Gaussian problem.
- `gaussian_grid.py` contains the implementation of a hierarchical Gaussian problem defined on a grid.
Each problem has a corresponding notebook. All models plots are saved in the `models` or `plots` folder.

Different noise schedules and weighting functions are visualized in the notebook `Visualize Schedules`.
