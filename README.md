# Stiefel Manifold Dynamical Systems for Tracking Representational Drift

This repository contains the code accompanying the paper "Stiefel Manifold Dynamical Systems for Tracking Representational Drift", Lee et al.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Installation](#installation)  

---

## Introduction

A **Stiefel Manifold Dynamical System (SMDS)** is a model designed to capture and track representational drift by modeling the drift on the Stiefel manifold. The model supports both out-of-manifold and within-manifold rotation, parameterized by D(D-1)/2 + D(N-D) degrees of freedom respectively.

This repository includes:

- A Python source code implementing the SMDS framework.  
- A Jupyter notebook demonstrating how to sample data from an SMDS and subsequently re-fit an SMDS to the sampled data.  
- An environment configuration file (`environment.yml`) listing all dependencies.

The implementation leverages Dynamax [1], a JAX-based library for probabilistic state-space models.

[1] Linderman, S. W., Chang, P., Harper-Donnelly, G., Kara, A., Li, X., Duran-Martin, G., & Murphy, K. (2025).
Dynamax: A Python package for probabilistic state space modeling with JAX. *Journal of Open Source Software*, 10(108), 7069.
https://doi.org/10.21105/joss.07069

---

## Quick Start

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from src.smds.models import StiefelManifoldSSM
from src.utils.utils import random_rotation, rotate_subspace

# --- 1. Define model dimensions ---
D = 2           # latent state dim
N = 10          # observation dim
num_trials = 100
num_conditions = 1
num_timesteps = 30

# --- 2. Initialize the model ---
model = StiefelManifoldSSM(
    state_dim=D,
    emission_dim=N,
    num_trials=num_trials,
    num_conditions=num_conditions,
    has_dynamics_bias=False,
    tau_per_dim=True,
)

key = jr.PRNGKey(0)
U_base = jnp.eye(N)                          # base subspace
tau = jnp.ones(model.dof) * 1e-5             # process noise for velocity
dynamics = random_rotation(key, D, theta=jnp.pi / 5)

key, key_init = jr.split(key)
params, props, velocity = model.initialize(
    U_base=U_base,
    tau=tau,
    key=key_init,
    dynamics_weights=dynamics,
    dynamics_covariance=jnp.eye(D) * 1e-2,
    emission_covariance=jnp.eye(N) * 1e-2,
)

# --- 3. Simulate data ---
conditions = jnp.zeros(num_trials, dtype=int)
key, key_sample = jr.split(key)
states, emissions = model.sample(params, key_sample, num_timesteps, conditions=conditions)
print(f"states: {states.shape}, emissions: {emissions.shape}")

# --- 4. Fit with EM ---
block_size = 1
num_blocks = num_trials // block_size
block_ids = jnp.repeat(jnp.eye(num_blocks), block_size, axis=1)
block_masks = jnp.ones(num_blocks, dtype=bool)
trial_masks = jnp.ones(num_trials, dtype=bool)

fitted_params, marginal_lls = model.fit_em(
    params, props,
    emissions,
    conditions=conditions,
    trial_masks=trial_masks,
    block_ids=block_ids,
    block_masks=block_masks,
    num_iters=50,
)
```

See `toy_example.ipynb` for a more complete example with model comparison and visualization.

---

## Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/lindermanlab/smds.git
   cd smds
2. **Create a Conda environment**  
   ```bash
   conda env create -f environment.yml
   conda activate smds