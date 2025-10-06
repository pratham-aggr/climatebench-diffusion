# ClimateBench-Diffusion 🌍

> Fast and accurate climate emulation using state-of-the-art diffusion models

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Extending the Project](#extending-the-project)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

**ClimateBench-Diffusion** is a research framework that uses **diffusion models** to emulate climate data from the ClimateBench dataset. It provides:

- **Fast climate simulations**: Train ML emulators that run 1000x faster than physics-based models
- **Uncertainty quantification**: Generate ensemble predictions with calibrated uncertainty estimates
- **State-of-the-art diffusion models**: EDM, DDPM, and ERDM implementations
- **Modular architecture**: Easy to extend with new models, data, and algorithms

### What This Project Does

The framework takes **climate forcings** (CO2, CH4, SO2, BC, solar radiation) as inputs and predicts **climate outputs** like surface temperature (`tas`) and precipitation (`pr`). It uses diffusion models to generate high-quality predictions with uncertainty estimates, enabling fast exploration of climate scenarios.

### Why This Matters

Traditional climate models are computationally expensive. This project trains **fast ML emulators** that can:
- Run thousands of simulations quickly
- Provide uncertainty quantification
- Help scientists explore climate scenarios efficiently
- Enable real-time climate impact assessments

---

## ✨ Key Features

- 🌊 **Multiple Diffusion Algorithms**: EDM, DDPM, ERDM with configurable sampling
- 🧠 **Advanced Neural Networks**: UNet, ADM (DhariwalUNet), SFNO architectures
- 📊 **ClimateBench Dataset**: Daily climate data from CESM2 model
- 🎛️ **Flexible Configuration**: Hydra-based config system for easy experimentation
- 📈 **Experiment Tracking**: Weights & Biases integration for monitoring
- ⚡ **Production Ready**: PyTorch Lightning with multi-GPU support
- 🔧 **Extensible**: Easy to add new models, data, and algorithms

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CONFIGURATION SYSTEM                      │
│        (Hydra configs: model, data, diffusion, etc.)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   MAIN ENTRY POINT                          │
│  run.py → calls src/train.py → run_model()                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────┐            ┌──────────────────┐
│  DATAMODULE  │            │  LIGHTNING MODULE │
│              │            │                   │
│ ClimateBench │            │  EmulationExpt    │
│ Daily Data   │            │  (BaseExperiment) │
│              │            │        +          │
│ - Loads data │            │  Diffusion Model  │
│ - Normalize  │            │  (EDM/DDPM/etc.)  │
│ - Batching   │            │        +          │
│              │            │  Neural Network   │
│              │            │  (UNet/ADM/SFNO)  │
└──────┬───────┘            └────────┬──────────┘
       │                             │
       └──────────────┬──────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ PyTorch       │
              │ Lightning     │
              │ Trainer       │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ Weights &     │
              │ Biases (wandb)│
              │ Logging       │
              └───────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB+ RAM (recommended)
- 50GB+ disk space (for ClimateBench data)

### Install Dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd climatebench-diffusion

# Create a virtual environment
conda create -n climate-diffusion python=3.9
conda activate climate-diffusion

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install pytorch-lightning hydra-core wandb xarray netcdf4 einops torchmetrics
pip install numpy scipy matplotlib seaborn tqdm
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning as pl; print(f'Lightning: {pl.__version__}')"
```

---

## 🏃‍♂️ Quick Start

### 1. Download ClimateBench Data

```bash
# Edit download_data/download.py to set your data directory
python download_data/download.py
```

This downloads data from the ClimateBench S3 bucket to your local directory.

### 2. Run Your First Experiment

```bash
# Train a simple model
python run.py \
    experiment=climatebench_daily_edm \
    datamodule.data_dir=/path/to/your/data \
    trainer.max_epochs=10 \
    logger=none
```

### 3. Monitor Training (with Wandb)

```bash
# Login to wandb (one-time setup)
wandb login

# Run with experiment tracking
python run.py \
    experiment=climatebench_daily_edm \
    logger=wandb \
    name="my-first-experiment"
```

---

## 📊 Data Preparation

### ClimateBench Dataset Structure

The ClimateBench dataset contains:

**Input Variables (Forcings):**
- `CO2`: Carbon dioxide concentration
- `CH4`: Methane concentration  
- `SO2`: Sulfur dioxide concentration
- `BC`: Black carbon concentration
- `rsdt`: Solar radiation

**Output Variables (Climate):**
- `tas`: Surface air temperature
- `pr`: Precipitation
- `pr90`: 90th percentile precipitation
- `dtr`: Diurnal temperature range

### Data Format

- **Inputs**: `(batch, 1, channels_in, lat, lon)` - Shape: `(B, 1, 5, 96, 144)`
- **Targets**: `(batch, channels_out, lat, lon)` - Shape: `(B, 2, 96, 144)`
- **Format**: NetCDF files with xarray
- **Resolution**: ~2° latitude/longitude
- **Time**: Daily data from 1850-2100

### Download Script

Edit `download_data/download.py`:

```python
# Set your local data directory
local_data_dir = "/path/to/your/data"

# Uncomment files you need
files = [
    "inputs_ssp126.nc",
    "inputs_ssp370.nc", 
    "inputs_ssp585.nc",
    "outputs_ssp126_daily.nc",
    "outputs_ssp370_daily.nc",
    "outputs_ssp585_daily.nc",
    # ... more files
]
```

---

## 💻 Usage

### Basic Training

```bash
# Train temperature prediction
python run.py \
    experiment=climatebench_daily_edm \
    datamodule.output_vars="[tas]" \
    datamodule.data_dir=/data \
    trainer.max_epochs=100

# Train temperature + precipitation
python run.py \
    experiment=climatebench_daily_edm \
    datamodule.output_vars="[tas, pr]" \
    datamodule.data_dir=/data \
    trainer.max_epochs=100
```

### Using Pre-configured Scripts

```bash
# Use the current best configuration
bash scripts/current_best_script/day-tas-pr-edm-adm-pcw-std-wmse-rsdt-44lr-pm-1-all-ens-d64.sh
```

### Evaluation

```bash
# Test on best checkpoint
python run.py \
    logger.wandb.id=<your_run_id> \
    eval_mode=test \
    ckpt_path=best.ckpt

# Generate predictions
python run.py \
    logger.wandb.id=<your_run_id> \
    eval_mode=predict \
    ckpt_path=best.ckpt
```

### Resume Training

```bash
# Resume from wandb run
python run.py logger.wandb.id=<run_id>

# Resume with modifications
python run.py \
    logger.wandb.id=<run_id> \
    trainer.max_epochs=200 \
    module.optimizer.lr=1e-4
```

---

## ⚙️ Configuration

### Hydra Configuration System

The project uses Hydra for hierarchical configuration management. Configs are organized in `src/configs/`:

```
src/configs/
├── main_config.yaml           # Root configuration
├── datamodule/                # Data loading configs
│   └── climatebench_daily.yaml
├── model/                     # Neural network configs
│   ├── adm.yaml
│   ├── unet_resnet.yaml
│   └── sfno.yaml
├── diffusion/                 # Diffusion algorithm configs
│   ├── edm.yaml
│   └── ddpm.yaml
├── module/                    # Experiment type configs
│   └── emulation.yaml
└── experiment/                # Complete experiment configs
    └── climatebench_daily_edm.yaml
```

### Key Configuration Parameters

#### Model Configuration (`model=adm`)
```yaml
model:
  model_channels: 192          # Base number of channels
  channel_mult: [1, 2, 3, 4]  # Channel multipliers per resolution
  dropout: 0.1                 # Dropout probability
  with_time_emb: True          # Include time embeddings
```

#### Diffusion Configuration (`diffusion=edm`)
```yaml
diffusion:
  num_steps: 16                # Number of denoising steps
  P_mean: -1.2                 # Noise level distribution mean
  P_std: 1.2                   # Noise level distribution std
  sigma_min: 0.02              # Minimum noise level
  sigma_max_inf: 400           # Maximum noise level
```

#### Data Configuration
```yaml
datamodule:
  data_dir: "/data"            # Path to NetCDF files
  output_vars: "[tas, pr]"     # Variables to predict
  batch_size: 512              # Training batch size
  window: 1                    # Temporal window size
  mean_over_ensemble: true     # Average over ensemble members
```

### Command Line Overrides

```bash
# Override any config parameter
python run.py \
    model.model_channels=128 \
    diffusion.num_steps=32 \
    datamodule.batch_size=256 \
    trainer.max_epochs=50

# Override nested configs
python run.py \
    optimizer@module.optimizer=adam \
    scheduler@module.scheduler=cosine
```

---

## 📁 Project Structure

```
climatebench-diffusion/
├── README.md                           # This file
├── run.py                             # Main entry point
├── download_data/                      # Data download scripts
│   ├── download.py
│   └── normalize_raw_data.py
├── scripts/                           # Training scripts
│   ├── current_best_script/
│   └── training.sh
└── src/                               # Source code
    ├── train.py                       # Main training loop
    ├── interface.py                   # Model/data instantiation
    ├── configs/                       # Hydra configurations
    │   ├── main_config.yaml
    │   ├── datamodule/
    │   ├── model/
    │   ├── diffusion/
    │   ├── module/
    │   └── experiment/
    ├── datamodules/                   # Data loading
    │   └── climatebench/
    │       ├── climatebench_daily.py
    │       └── climatebench_original.py
    ├── diffusion/                     # Diffusion algorithms
    │   ├── edm.py                     # EDM (Elucidating Diffusion Models)
    │   ├── ddpm.py                    # DDPM
    │   ├── erdm.py                    # Extended RDM
    │   └── schedulers/
    ├── models/                        # Neural networks
    │   ├── networks_edm.py            # DhariwalUNet (ADM)
    │   ├── unet.py                    # Standard UNet
    │   ├── sfno/                      # Spherical Fourier Neural Operator
    │   └── unet3d.py                  # 3D UNet
    ├── experiment_types/              # Training logic
    │   ├── emulation.py               # Input→Output mapping
    │   ├── forecasting_multi_horizon.py
    │   └── _base_experiment.py        # Base Lightning module
    ├── evaluation/                    # Metrics and evaluation
    │   ├── metrics.py                 # RMSE, CRPS, etc.
    │   └── aggregators/
    └── utilities/                     # Helper functions
        ├── config_utils.py            # Hydra utilities
        ├── wandb_callbacks.py         # Logging callbacks
        └── utils.py
```

---

## 🔧 Extending the Project

### What's Safe to Modify

#### ✅ **SAFE** (High-level)
- **Experiment configs** (`src/configs/experiment/*.yaml`)
- **Data augmentation** (add in `datamodules/`)
- **New metrics** (add to `evaluation/metrics.py`)
- **Hyperparameters** (learning rate, batch size, etc.)
- **Logging/callbacks** (`utilities/wandb_callbacks.py`)

#### ⚠️ **CAREFUL** (Mid-level)
- **New model architectures** (`models/*.py`)
- **New diffusion algorithms** (`diffusion/*.py`)
- **New experiment types** (`experiment_types/*.py`)
- **Data loaders** (`datamodules/*.py`)

#### 🔒 **CORE** (Don't modify unless necessary)
- `train.py` (main training loop)
- `interface.py` (instantiation logic)
- `_base_experiment.py` (Lightning module base class)
- `_base_model.py` (model base class)

### Adding New Features

#### 1. Add a New Climate Variable

**File**: `src/datamodules/climatebench/climatebench_daily.py`

```python
# In __init__:
self.ovar_to_var_id = {
    "tas": "tas",
    "pr": "pr", 
    "hus": "specific_humidity",  # NEW!
}

# Then run:
python run.py datamodule.output_vars='[tas, pr, hus]'
```

#### 2. Implement a New Diffusion Algorithm

**File**: `src/diffusion/my_algorithm.py`

```python
from src.diffusion._base_diffusion import BaseDiffusion

class MyDiffusion(BaseDiffusion):
    def get_loss(self, inputs, targets, **kwargs):
        # Your loss computation
        return loss
    
    def sample(self, inputs, **kwargs):
        # Your sampling loop
        return predictions
```

**Config**: `src/configs/diffusion/my_algorithm.yaml`

```yaml
_target_: src.diffusion.my_algorithm.MyDiffusion
sigma_min: 0.002
sigma_max: 80
# ... your hyperparameters
```

**Run**: `python run.py diffusion=my_algorithm`

#### 3. Add Physics-Informed Loss

**File**: `src/experiment_types/emulation.py`

```python
def get_loss(self, batch):
    # Standard diffusion loss
    loss = super().get_loss(batch)
    
    # Physics loss: penalize large spatial gradients
    targets = batch["targets"]
    grad_x = targets[:, :, :, 1:] - targets[:, :, :, :-1]
    grad_y = targets[:, :, 1:, :] - targets[:, :, :-1, :]
    physics_loss = (grad_x.abs().mean() + grad_y.abs().mean())
    
    # Combine losses
    total_loss = loss + 0.01 * physics_loss
    
    self.log("train/physics_loss", physics_loss)
    return total_loss
```

#### 4. Add Attention Visualization

**File**: `src/utilities/wandb_callbacks.py`

```python
class AttentionVisualizationCallback(pl.Callback):
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx == 0:
            # Extract attention maps
            attn_maps = pl_module.model.get_attention_maps()
            
            # Log to wandb
            wandb.log({"attention": wandb.Image(attn_maps)})
```

### Development Roadmap

#### **Phase 1: Understand (1-2 weeks)**
1. Run default experiment
2. Modify hyperparameters
3. Add simple logging
4. Read core files

#### **Phase 2: Minor Modifications (2-3 weeks)**
1. New evaluation metrics
2. Data augmentation
3. New callbacks
4. Hyperparameter tuning

#### **Phase 3: Major Extensions (1-2 months)**
1. New diffusion algorithms
2. New neural architectures
3. Temporal forecasting
4. Conditional generation

#### **Phase 4: Research (3+ months)**
1. Physics-informed losses
2. Uncertainty calibration
3. Multi-scale diffusion
4. Transfer learning

---

## 📚 API Reference

### Core Classes

#### `EmulationExperiment`
Main Lightning module for climate emulation.

```python
class EmulationExperiment(BaseExperiment):
    def __init__(self, pr_clamping=False, **kwargs):
        # Initialize emulation experiment
    
    def get_loss(self, batch):
        # Compute training loss
    
    def predict(self, inputs, **kwargs):
        # Generate predictions
```

#### `EDMPrecond`
EDM diffusion model implementation.

```python
class EDMPrecond(BaseDiffusion):
    def get_loss(self, inputs, targets, **kwargs):
        # EDM loss computation
    
    def sample(self, inputs, num_steps=18):
        # EDM sampling loop
```

#### `ClimateBenchDailyDataModule`
Data loader for ClimateBench daily data.

```python
class ClimateBenchDailyDataModule(BaseDataModule):
    def setup(self, stage=None):
        # Load and preprocess data
    
    def train_dataloader(self):
        # Return training dataloader
```

### Key Functions

#### `get_model_and_data(config)`
Instantiate model and datamodule from config.

```python
model, datamodule = get_model_and_data(config)
```

#### `run_model(config)`
Main training function.

```python
best_score = run_model(config)
```

### Configuration Classes

#### Model Configs
- `adm.yaml`: DhariwalUNet (ADM architecture)
- `unet_resnet.yaml`: UNet with ResNet blocks
- `sfno.yaml`: Spherical Fourier Neural Operator

#### Diffusion Configs
- `edm.yaml`: Elucidating Diffusion Models
- `ddpm.yaml`: Denoising Diffusion Probabilistic Models

#### Experiment Configs
- `climatebench_daily_edm.yaml`: ClimateBench + EDM
- `climatebench_daily.yaml`: ClimateBench baseline

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Shape Mismatches
**Problem**: Inputs and targets have different spatial resolutions.

**Solution**:
```yaml
# In model config:
upsample_condition_by: 2  # Upsample inputs 2x to match targets
```

#### 2. Memory Issues
**Problem**: Batch size too large for GPU.

**Solutions**:
```bash
# Reduce batch size
python run.py datamodule.batch_size=128

# Use gradient accumulation
python run.py trainer.accumulate_grad_batches=4

# Enable torch.compile
python run.py module.torch_compile="module"
```

#### 3. Training Instability
**Problem**: Loss explodes or NaN.

**Solutions**:
```bash
# Lower learning rate
python run.py module.optimizer.lr=1e-4

# Use gradient clipping
python run.py trainer.gradient_clip_val=1.0

# Check data normalization
python run.py datamodule.normalization_type="standard"
```

#### 4. Poor Sample Quality
**Problem**: Generated samples look blurry or incorrect.

**Solutions**:
```bash
# Increase model capacity
python run.py model.model_channels=256

# More denoising steps
python run.py diffusion.num_steps=32

# Adjust noise schedule
python run.py diffusion.P_mean=-1.0 diffusion.P_std=1.0
```

#### 5. Hydra Config Errors
**Problem**: `hydra.errors.InstantiationException`

**Solution**: Scroll up to see the actual error (Hydra error messages are nested).

### Debugging Tips

#### Enable Verbose Logging
```bash
python run.py verbose=True print_config=True
```

#### Run in Debug Mode
```bash
python run.py datamodule.debug_mode=True trainer.fast_dev_run=True
```

#### Inspect Data
```python
from src.interface import get_datamodule
from src.utilities.config_utils import get_config_from_hydra_compose_overrides

config = get_config_from_hydra_compose_overrides(['datamodule=climatebench_daily'])
dm = get_datamodule(config)
dm.setup('fit')

batch = next(iter(dm.train_dataloader()))
print(batch.keys())
print(batch['inputs'].shape)
print(batch['targets'].shape)
```

#### Profile Memory
```python
from src.utilities.utils import print_gpu_memory_usage
print_gpu_memory_usage()
```

### Getting Help

1. **Check the logs**: Look for error messages in the terminal output
2. **Enable debug mode**: Use `debug_mode=True` for detailed logging
3. **Validate config**: Use `print_config=True` to see the full configuration
4. **Check data**: Ensure your data files are properly downloaded and accessible

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{climatebench_diffusion,
  title={ClimateBench-Diffusion: Fast Climate Emulation with Diffusion Models},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/climatebench-diffusion}
}
```

Related papers:
- **EDM**: Karras, T., et al. "Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.
- **ClimateBench**: Watson-Parris, D., et al. "ClimateBench v1.0: A benchmark for data-driven climate projections." JAMES 2022.

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install in development mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Run linting
black src/
flake8 src/
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **ClimateBench Team** for the dataset
- **EDM Authors** for the diffusion model implementation
- **PyTorch Lightning** for the training framework
- **Hydra** for configuration management
- **Weights & Biases** for experiment tracking

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/climatebench-diffusion/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/climatebench-diffusion/discussions)
- **Email**: your-email@example.com

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for the climate science community

</div>