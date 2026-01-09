Use core/ methods to simulate networks. Use LDR_dim/ methods to solve theory.

Example_Theory_Fits.ipynb shows functions to call for sampling connectivity matrices, computing empirical autocovariance functions, and computing theoretical predictions.

## Getting Started

### Quick Setup (Conda - Recommended)

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd mft-theory
   ```

2. **Create and activate conda environment:**
   ```bash
   conda env create -f environment.yml
   conda activate mft-theory
   ```

3. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

4. **In a new notebook, use the template:**
   ```python
   from notebook_setup import setup_mft_theory
   setup = setup_mft_theory(use_gpu=False)
   # All modules are now imported and ready to use!
   ```

### Alternative Setup (pip)

If you prefer pip or don't have conda:
```bash
pip install -r requirements.txt
jupyter notebook
```

### Notebook Portability

Notebooks using `notebook_setup.py` automatically work on:
- ✅ Any local machine (macOS, Linux, Windows)
- ✅ HPC clusters (auto-detects cluster paths)
- ✅ Different Python environments

The setup function:
- Auto-detects the repository root path
- Handles CPU/GPU device selection
- Imports all necessary modules
- Checks for missing dependencies with helpful error messages

See `template_notebook.ipynb` for a complete example.

## Cluster Sync

This repo includes tools to sync code to HPC clusters where Jupyter notebooks may reference the code.

### Quick Sync

One-time sync to cluster:
```bash
# Sync to Columbia cluster (default)
python cluster_sync.py --cluster columbia

# Sync to NYU/Greene cluster
python cluster_sync.py --cluster nyu
```

### Auto-Sync (Recommended)

Start a file watcher that automatically syncs when you make changes:
```bash
# Watch and auto-sync to Columbia cluster
python cluster_sync.py --watch --cluster columbia

# With custom debounce time (wait 5 seconds after last change)
python cluster_sync.py --watch --cluster columbia --debounce 5
```

The watcher runs in the foreground until you press Ctrl+C. It will:
- Detect file changes, creations, and deletions
- Wait for a quiet period (debounce) before syncing
- Automatically sync to the specified cluster
- Skip syncing temporary files (`.pyc`, `__pycache__`, `.git`, etc.)

### Programmatic Usage

```python
from cluster import sync_columbia_cluster, sync_cluster, watch_and_sync

# One-time sync (auto-detects repo path)
sync_columbia_cluster()  # or sync_cluster() for NYU

# Start auto-sync watcher
observer = watch_and_sync(cluster='columbia', debounce_seconds=2)
# ... do work ...
observer.stop()  # when done
```

### Requirements

For auto-sync watcher, install `watchdog`:
```bash
pip install watchdog
```

The sync functions use `rsync` which should be available on macOS/Linux by default.