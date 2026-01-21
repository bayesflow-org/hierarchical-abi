# Compositional Amortized Inference for Large-Scale Hierarchical Bayesian Models

This repository accompanies the [paper](https://arxiv.org/abs/2505.14429):
**"Compositional Amortized Inference for Large-Scale Hierarchical Bayesian Models"**.

We propose a method based on **compositional diffusion models** for **efficient, amortized Bayesian inference** in *hierarchical models*. 
In this work, we build on compositional score matching (CSM), a divide-and-conquer strategy for Bayesian updating using diffusion models. 
To address existing stability issues of CSM, we propose adaptive solvers coupled with a novel, error-damping compositional estimator.

## 📁 Repository Structure

### `diffusion_model/`

This folder contains the core components of the score-based diffusion model.

* **`diffusion_model.py`**
  Implements the flat and hierarchical diffusion models using the SDE interpretation from [Song et al. (2021)](https://arxiv.org/abs/2011.13456).

* **`diffusion_sde.py`**
  Defines the forward and reverse stochastic differential equations for the diffusion process.

* **`sampling_algorithms.py`**
  Implements multiple sampling algorithms for the generative diffusion model:

  * Euler-Maruyama
  * Adaptive Euler-Maruyama (with 2nd-order correction)
  * Annealed Langevin Dynamics
  * ODE-based probability flow (Euler discretization)

  Each sampler supports a customizable `sampling_arg`, e.g.:

  ```python
  sampling_arg = {
      'size': 5,
      'damping_factor': lambda t: t0_value * torch.exp(-np.log(t0_value / t1_value) * t),
  }
  ```

* **`train_score_models.py`**
  Training routines for the flat and hierarchical score networks.

---

### `experiments/`

This folder organizes all experimental assets.

* **`plots/`**
  Contains all figures generated for visualization and analysis.

* **`problems/`**
  For each experiment, this subdirectory includes:

  * A simulation model definition.
  * The training objective for the score model.
  * A Jupyter notebook for reproducing results and visualizations.

---

### 📓 Notebooks

* **`Visualize Schedules.ipynb`**
  Interactive notebook to visualize and compare different noise schedules and weighting functions.

---

## 🚀 Reproducing Results

We recommend creating a fresh Python environment (e.g., via `uv`) and installing the dependencies listed in `pyproject.toml`:

```bash
uv venv --python 3.11
uv sync
```

For some experiments, you also need to install `stan` via `cmdstanpy`:
```python
import cmdstanpy
cmdstanpy.install_cmdstan()
```

You can reproduce the experiments from the paper by running the problem-specific scripts (which include training and evaluation) and inspecting the generated plots in `experiments/plots`. 
For each experiment, corresponding Jupyter notebooks are provided to ease evaluation and interpretation.

---
