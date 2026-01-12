# Notebook Portability Guide

## Overview

All notebooks in this repository are designed to work on **any machine** - your local computer, HPC clusters, or collaborator's machines. This is achieved through automatic path detection and dependency management.

## Quick Start

In any notebook, use this one-line setup:

```python
from notebook_setup import setup_mft_theory
setup = setup_mft_theory(use_gpu=False)
```

That's it! All modules are now imported and ready to use.

## How It Works

### 1. Automatic Path Detection

The `notebook_setup.py` module automatically finds the `mft-theory` repository root by:
- Checking the current directory and parent directories
- Looking for common cluster paths (`/home/om2382/mft-theory/`, `/scratch/oem214/mft-theory/`)
- Falling back to the current working directory if needed

### 2. Device Selection

Automatically selects CPU or GPU based on availability:
```python
# CPU (local, default)
setup = setup_mft_theory(use_gpu=False)

# GPU (cluster with CUDA)
setup = setup_mft_theory(use_gpu=True)  # Falls back to CPU if GPU unavailable
```

### 3. Dependency Checking

The setup function checks for required packages and provides helpful error messages if anything is missing:

```
⚠️  Missing packages: PyTorch
   Install with: conda env create -f scripts/environment.yml
   Or see README.md for installation instructions
```

### 4. Module Imports

All standard modules are imported into the notebook's namespace:
- `numpy` as `np`
- `matplotlib.pyplot` as `plt`
- `torch`
- All `mft-theory` modules (`functions`, `core`, `theory`, `utils`, `plotting`, etc.)

## Installation Requirements

### Option 1: Conda (Recommended)

```bash
conda env create -f scripts/environment.yml
conda activate mft-theory
```

### Option 2: pip

```bash
pip install -r scripts/requirements.txt
```

### Minimum Requirements

- Python 3.10+
- PyTorch (CPU or GPU version)
- NumPy
- SciPy
- Matplotlib
- Jupyter Notebook

## Using the Setup Function

### Basic Usage

```python
from notebook_setup import setup_mft_theory

# Setup with CPU (local)
setup = setup_mft_theory(use_gpu=False)

# Extract useful variables
device = setup['device']        # torch.device('cpu') or torch.device('cuda')
device_idx = setup['device_idx'] # 'cpu' or 0
to_torch = setup['to_torch']    # Helper function for device-aware tensors
repo_root = setup['repo_root']  # Path to repository root
```

### Advanced Options

```python
# Skip cluster tools (for pure local work)
setup = setup_mft_theory(use_gpu=False, import_cluster=False)

# Skip empirics modules (if not needed)
setup = setup_mft_theory(use_gpu=False, import_empirics=False)

# Skip dependency checking (if you know everything is installed)
setup = setup_mft_theory(use_gpu=False, check_deps=False)
```

## Example: Complete Notebook

```python
# Cell 1: Setup
from notebook_setup import setup_mft_theory
setup = setup_mft_theory(use_gpu=False)
device = setup['device']
to_torch = setup['to_torch']

# Cell 2: Use imported modules (all available now!)
import numpy as np
import matplotlib.pyplot as plt
import torch

# Run simulation
N = 100
g = 1.5
W = np.random.randn(N, N) * g / np.sqrt(N)
W_torch = to_torch(W)  # Automatically on correct device

# Continue with your code...
```

## Troubleshooting

### "ImportError: PyTorch is not installed"

Install PyTorch:
```bash
conda install pytorch -c pytorch  # or pip install torch
```

### "Could not import [module]"

Make sure you're in the repository directory or have cloned the repo. The setup function will try to find the repo automatically.

### "GPU not available" (when use_gpu=True)

This is normal on machines without CUDA. The function automatically falls back to CPU.

### Path Detection Issues

If the repo isn't found automatically, make sure:
1. You've cloned the repository
2. You're running the notebook from within or near the repo directory
3. Or manually add the repo to your Python path:

```python
import sys
sys.path.insert(0, '/path/to/mft-theory')
```

## Best Practices

1. **Always use `notebook_setup`** - Don't manually import paths or modules
2. **Use `to_torch()` helper** - Ensures tensors go to the correct device
3. **Check `device` before running code** - Know whether you're on CPU or GPU
4. **Use relative imports** - Works across machines
5. **Test locally first** - Use CPU mode before running on cluster

## Portability Checklist

Before sharing a notebook, ensure:
- ✅ Uses `from notebook_setup import setup_mft_theory` for setup
- ✅ No hardcoded file paths (use `repo_root` or relative paths)
- ✅ Uses `device` variable for device selection
- ✅ Uses `to_torch()` helper for tensor creation
- ✅ All dependencies are listed in `scripts/environment.yml` or `scripts/requirements.txt`
- ✅ Works with both CPU and GPU (if applicable)
