# Stiefel Manifold Dynamical Systems for Tracking Representational Drift

This repository contains the code accompanying the paper "Stiefel Manifold Dynamical Systems for Tracking Representational Drift", Lee et al.

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Installation](#installation)  

---

## Introduction

A **Stiefel Manifold Dynamical System (SMDS)** is a model designed to capture and track representational drift by modeling the drift on the Stiefel manifold. 

This repository includes:

- A Python source code implementing the SMDS framework.  
- A Jupyter notebook demonstrating how to sample data from an SMDS and subsequently re-fit an SMDS to the sampled data.  
- An environment configuration file (`environment.yml`) listing all dependencies.

The implementation leverages Dynamax [1], a JAX-based library for probabilistic state-space models.

[1] Linderman, S. W., Chang, P., Harper-Donnelly, G., Kara, A., Li, X., Duran-Martin, G., & Murphy, K. (2025).  
Dynamax: A Python package for probabilistic state space modeling with JAX. *Journal of Open Source Software*, 10(108), 7069.  
https://doi.org/10.21105/joss.07069

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