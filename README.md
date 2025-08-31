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
