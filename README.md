# mft-theory

Mean field theory for neural networks.

## Quick Start

Use `core/` methods to simulate networks. Use `LDR_dim/` methods to solve theory.

`Example_Theory_Fits.ipynb` shows functions to call for sampling connectivity matrices, computing empirical autocovariance functions, and computing theoretical predictions.

## Documentation

All documentation is in the [`docs/`](docs/) directory:

- **Setup & Installation**: See [`docs/JUPYTER_SETUP.md`](docs/JUPYTER_SETUP.md)
- **Notebook Portability**: See [`docs/NOTEBOOK_PORTABILITY.md`](docs/NOTEBOOK_PORTABILITY.md)
- **Cluster Sync**: See [`docs/CLUSTER_SYNC_GUIDE.md`](docs/CLUSTER_SYNC_GUIDE.md)
- **Workflow Guides**: See other files in [`docs/`](docs/)

## Scripts

Setup and utility scripts are in the [`scripts/`](scripts/) directory:

- `notebook_setup.py` - Notebook setup helper (import as `from notebook_setup import setup_mft_theory`)
- `cluster_sync.py` - Cluster synchronization tool
- `jupyter_notebook.py` - Jupyter notebook launcher
- `template_notebook.ipynb` - Template notebook

## Project Structure

```
mft-theory/
├── core/           # Core simulation methods
├── theory/         # Theory modules
├── training/       # Training utilities
├── LDR_dim/        # Low-dimensional theory
├── plotting/       # Plotting utilities
├── utils/          # General utilities
├── cluster/         # Cluster management tools
├── docs/            # Documentation
└── scripts/         # Setup and utility scripts
    ├── environment.yml    # Conda environment definition
    └── requirements.txt   # Pip requirements
```
