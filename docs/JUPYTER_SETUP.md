# Jupyter Setup Guide

## Recommended: Classic Jupyter Notebook

The cluster uses **classic Jupyter Notebook** (via `sjupyter`), and it's the recommended choice because:
- ✅ More stable with Safari (no glitchy scrolling issues)
- ✅ Simpler, cleaner interface
- ✅ Better compatibility with older browsers
- ✅ What the cluster uses, so consistent experience

## Launch Classic Notebook

```bash
conda activate mft-theory
jupyter notebook
```

This will open classic notebook in your browser - the same interface you see on the cluster!

## Alternative: JupyterLab

If you want to try JupyterLab (though it can be buggy with Safari):

```bash
conda activate mft-theory
jupyter lab
```

**Note:** Classic notebook is perfectly fine to use - it's still actively maintained and often more stable than JupyterLab, especially with Safari.

## After Environment Installation

1. **Activate environment:**
   ```bash
   conda activate mft-theory
   ```

2. **Launch JupyterLab (recommended):**
   ```bash
   jupyter lab
   ```

3. **Or use classic notebook (if you prefer):**
   ```bash
   jupyter notebook
   ```

## Template Notebook

The `template_notebook.ipynb` now uses a one-line setup:
```python
from notebook_setup import setup_mft_theory
setup = setup_mft_theory(use_gpu=False)
```

This replaces all the preamble code with a single import!
