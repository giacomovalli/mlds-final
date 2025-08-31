# STAAR: Stylized Time-series Adversarial Autoencoder for Returns

A sophisticated deep learning framework for financial time series analysis and generation using Adversarial Autoencoders (AAE) with attention mechanisms and specialized loss functions for financial data.

## Overview

STAAR (Stylized Time-series Adversarial Autoencoder for Returns) is designed to:

- Learn complex patterns in financial time series data
- Generate synthetic financial sequences that preserve stylized facts
- Perform comprehensive analysis of reconstruction quality
- Support distributed training on multiple GPUs
- Provide extensive visualization and analysis tools

## Features

- **Advanced Architecture**: Combines LSTM, Multi-Head Attention, and Adversarial Training
- **Financial-Specific Losses**: Includes Garman-Klass volatility loss, GARCH loss, and autocorrelation loss
- **Custom Activation Functions**: Scaled tanh activation with configurable ranges
- **Comprehensive Analysis**: t-SNE visualization, MMD testing, Wasserstein distances
- **Stylized Facts Testing**: Autocorrelation analysis, GARCH model fitting, volatility clustering
- **Distributed Training**: Multi-GPU support with TensorFlow's MirroredStrategy

## Requirements

This project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```
## Quick Start

### Training a Model

Train a basic STAAR model on the full dataset:

```bash
uv run python main.py --train --window 120 --latent_dim 32 --epochs 50 --batch_size 100
```

### Loading and Testing a Model

Load a pre-trained model and perform analysis:

```bash
uv run python main.py --test --load-model w120_l32_e50 --window 120 --latent_dim 32
```

### Financial Analysis

Perform various financial analyses on the dataset:

```bash
# Autocorrelation analysis
uv run python main.py --autocorrelation --year 2023

# Volatility clustering analysis
uv run python main.py --volatility_clustering --year 2023

# Gain/loss asymmetry analysis
uv run python main.py --gainloss --year 2023
```

## Command Line Arguments

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train` | flag | - | Run model training |
| `--test` | flag | - | Test loaded model and generate analysis plots |
| `--summary` | flag | - | Display model architecture summary |
| `--load-model` | str | - | Load a saved model using prefix (e.g., `w120_l32_e50`) |
