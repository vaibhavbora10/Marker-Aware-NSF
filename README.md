# NSF-Marker: Enhancing Nonnegative Spatial Factorization with Marker-Guided Gaussian Processes

This repository contains the full implementation, experiments, and evaluation pipeline for **NSF-Marker**, a marker-aware extension of the Nonnegative Spatial Factorization (NSF) framework for spatial transcriptomics analysis. It builds upon the original NSF model by incorporating prior biological knowledge in the form of marker gene constraints using a custom Gaussian Process (GP) kernel. Our implementation is influenced by techniques from the scDCC model, adapting them for structured spatial priors.

---

## 🧬 Overview

Spatial transcriptomics enables the study of gene expression within the spatial context of tissue architecture. The original NSF model provides a probabilistic nonnegative matrix factorization approach with spatial priors but treats all genes equally. NSF-Marker addresses this limitation by introducing a **marker-aware kernel**, which biases latent spatial factorization toward biologically meaningful genes.

---

## 📁 Repository Structure

```bash
├── run_training_test.ipynb      # Main notebook to train and test models
├── sf.py                        # Core NSF model (base version)
├── marker_kernel.py             # Custom MarkerAwareKernel implementation
├── training.py                  # Training loop and optimization
├── benchmark.py                 # Benchmarking and evaluation utilities
├── misc.py                      # Helper functions and utilities
├── requirements.txt             # Necessary libraries for training, modeling, and visualization pipelines
└── results                      # Saved spatial scores, figures, and comparisons
