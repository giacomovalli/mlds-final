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

### Model Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--window` | int | 120 | Sequence window size (time steps) |
| `--batch_size` | int | 100 | Training batch size |
| `--latent_dim` | int | 32 | Dimension of latent space |
| `--epochs` | int | 10 | Number of training epochs |

### Data Filtering

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--year` | str | - | Filter by year(s): single (`2023`) or multiple (`2022,2023,2024`) |
| `--month` | str | - | Filter by month(s): single (`6`) or multiple (`1,6,12`) |

### Analysis Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--autocorrelation` | flag | - | Analyze autocorrelation of price changes |
| `--gainloss` | flag | - | Analyze gain/loss asymmetry patterns |
| `--volatility_clustering` | flag | - | Analyze volatility clustering using GARCH |
| `--volume_volatility_corr` | flag | - | Analyze volume-volatility correlation |
| `--wavelets` | flag | - | Perform wavelet decomposition analysis |
| `--stationarity` | flag | - | Test stationarity of price series |
| `--fit-kde` | str | - | Fit KDE to specified column |
| `--test-generation` | flag | - | Test generation quality with discriminator |

### Advanced Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--kl-scaling-epochs` | int | 0 | Epochs for KL annealing (0 disables) |
| `--kl-scaling-factor` | float | 1.0 | KL scaling factor |

## Usage Examples

### Basic Training

Train a model on 2023 data with standard parameters:

```bash
uv run python main.py --train --year 2023 --window 120 --latent_dim 32 --epochs 100 --batch_size 128
```

### Multi-Year Training

Train on multiple years with larger latent space:

```bash
uv run python main.py --train --year 2019,2020,2021,2022 --window 120 --latent_dim 64 --epochs 200
```

### Seasonal Analysis

Train and analyze specific months across multiple years:

```bash
# Train on January data from multiple years
uv run python main.py --train --year 2019,2020,2021,2022 --month 1 --epochs 150

# Test the trained model
uv run python main.py --test --load-model w120_l32_e150_y2019_2020_2021_2022_m1 --window 120 --latent_dim 32
```

### Comprehensive Analysis Pipeline

```bash
# 1. Train the model
uv run python main.py --train --year 2023 --epochs 100 --window 120 --latent_dim 32

# 2. Test and analyze
uv run python main.py --test --load-model w120_l32_e100_y2023 --window 120 --latent_dim 32

# 3. Perform stylized facts analysis
uv run python main.py --autocorrelation --year 2023
uv run python main.py --volatility_clustering --year 2023
uv run python main.py --gainloss --year 2023

# 4. Test generation quality
uv run python main.py --test-generation --load-model w120_l32_e100_y2023 --window 120 --latent_dim 32
```

## Model Architecture

The STAAR model consists of:

1. **Encoder**: LSTM + Multi-Head Attention � Latent Distribution
2. **Decoder**: Dense � LSTM + Multi-Head Attention � Reconstruction
3. **Latent Discriminator**: Ensures latent space follows prior distribution
4. **Statistics Discriminator**: Enforces statistical properties of generated data

### Key Components

- **Multi-Head Attention**: Captures long-range dependencies in time series
- **Cholesky Parameterization**: Full covariance matrix in latent space
- **Custom Loss Functions**:
  - Gaussian NLL for reconstruction
  - KL divergence with optional free-bit regularization
  - Garman-Klass volatility loss
  - GARCH-based temporal loss
  - Custom standard deviation preservation loss

## Output Files

### Model Files
- `saved_models/`: Trained model components (.keras files)
- Models are saved with descriptive prefixes (e.g., `w120_l32_e50_y2023`)

### Analysis Plots
- `plots/`: Visualization outputs
  - Training loss curves
  - Reconstruction comparisons (3x2 grid)
  - t-SNE analysis plots
  - Stylized facts analysis
  - Generation quality assessments

### Log Files
Comprehensive logging includes:
- Training progress and loss values
- Model architecture details
- Analysis results and statistics
- Performance metrics and timing

## Advanced Features

### Custom Activation Function

The model includes a `scaled_tanh` activation function that maps inputs to the range [-1, 2], designed specifically for financial data characteristics.

### Multi-GPU Support

Automatic detection and utilization of multiple GPUs using TensorFlow's MirroredStrategy:

```bash
# Training will automatically use available GPUs
uv run python main.py --train --epochs 100 --batch_size 256
```

### Callbacks and Monitoring

Built-in callbacks for robust training:
- NaN detection and stopping
- Early stopping based on validation loss
- Model checkpointing
- KL annealing scheduling
- Negative reconstruction monitoring
