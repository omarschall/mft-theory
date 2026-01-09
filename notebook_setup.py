"""
Helper module for notebook setup - reduces preamble code in notebooks.

Usage in notebook:
    from notebook_setup import setup_mft_theory
    setup = setup_mft_theory(use_gpu=False)
    # Now all modules are imported into the notebook's global namespace

This module automatically:
- Finds the mft-theory repository root (works on any machine)
- Sets up device (CPU/GPU) based on availability
- Imports all necessary modules
- Checks for required dependencies
"""

import sys
import os
from pathlib import Path
import importlib

# Try importing required packages - provide helpful errors if missing
try:
    import torch
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Install with:\n"
        "  conda: conda install pytorch -c pytorch\n"
        "  pip: pip install torch\n"
        "  Or see environment.yml for full environment setup."
    )

try:
    import numpy as np
except ImportError:
    raise ImportError(
        "NumPy is not installed. Install with:\n"
        "  conda: conda install numpy\n"
        "  pip: pip install numpy\n"
        "  Or see environment.yml for full environment setup."
    )

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "Matplotlib is not installed. Install with:\n"
        "  conda: conda install matplotlib\n"
        "  pip: pip install matplotlib\n"
        "  Or see environment.yml for full environment setup."
    )

from functools import partial

def find_repo_root():
    """Find the mft-theory repository root."""
    # Try current directory first
    current = Path.cwd()
    while current != current.parent:
        if (current / 'mft-theory').exists() or (current.name == 'mft-theory'):
            if current.name == 'mft-theory':
                return str(current)
            elif (current / 'mft-theory').exists():
                return str(current / 'mft-theory')
        current = current.parent
    
    # Fallback: try common cluster paths
    cluster_paths = [
        '/home/om2382/mft-theory/',
        '/scratch/oem214/mft-theory/',
    ]
    for path in cluster_paths:
        if os.path.exists(path):
            return path
    
    # Last resort: assume we're in the repo
    return str(Path.cwd())

def _import_all_from_module(module_name, caller_globals):
    """Import all public names from a module into caller's global namespace."""
    try:
        module = importlib.import_module(module_name)
        # Get all public names (not starting with _)
        public_names = [name for name in dir(module) if not name.startswith('_')]
        # Update caller's globals
        for name in public_names:
            caller_globals[name] = getattr(module, name)
        return True
    except ImportError:
        return False

def check_dependencies():
    """Check if required dependencies are installed."""
    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib'
    }
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        print("   Install with: conda env create -f environment.yml")
        print("   Or see README.md for installation instructions")
        return False
    return True

def setup_mft_theory(use_gpu=False, import_cluster=True, import_empirics=True, check_deps=True):
    """
    Set up mft-theory environment for notebooks.
    
    This function must be called from a notebook cell, and it will import
    all modules into the notebook's global namespace (like `from X import *`).
    
    Args:
        use_gpu: Whether to use GPU (only works on cluster with CUDA)
        import_cluster: Whether to import cluster tools (optional, for local work)
        import_empirics: Whether to import empirics and LDR_dim modules (optional)
        check_deps: Whether to check for required dependencies (default: True)
    
    Returns:
        dict with 'device', 'device_idx', 'repo_root', and 'to_torch' function
    
    Example:
        >>> from notebook_setup import setup_mft_theory
        >>> setup = setup_mft_theory(use_gpu=False)
        >>> device = setup['device']
        >>> to_torch = setup['to_torch']
    """
    # Check dependencies if requested
    if check_deps:
        if not check_dependencies():
            print("⚠️  Some dependencies are missing, but continuing anyway...")
    
    # Get the caller's global namespace (the notebook's namespace)
    import inspect
    frame = inspect.currentframe().f_back
    notebook_globals = frame.f_globals
    
    # Find and add repo to path
    repo_root = find_repo_root()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    
    print(f"📂 Using repo at: {repo_root}")
    
    # Set up device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        device_idx = 0
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        device_idx = 'cpu'
        if use_gpu:
            print(f"⚠️  GPU requested but not available, using CPU")
        else:
            print(f"💻 Using CPU (set use_gpu=True for GPU on cluster)")
    
    # Import standard modules into notebook's global namespace
    modules_to_import = ['functions', 'core', 'ode_methods', 'theory', 'utils', 'plotting']
    for module_name in modules_to_import:
        if _import_all_from_module(module_name, notebook_globals):
            pass  # Success
        else:
            print(f"⚠️  Warning: Could not import {module_name}")
    print("✅ Core modules imported")
    
    # Optional imports
    if import_cluster:
        if _import_all_from_module('cluster', notebook_globals):
            print("✅ Cluster tools available")
        else:
            print("ℹ️  Cluster tools not available (this is fine for local work)")
    
    if import_empirics:
        if _import_all_from_module('empirics', notebook_globals):
            pass
        if _import_all_from_module('LDR_dim', notebook_globals):
            print("✅ Empirics and LDR_dim modules imported")
    
    # Helper function for device-aware tensors
    def to_torch(x, dtype=torch.float32):
        """Convert numpy array to torch tensor on correct device."""
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).type(dtype).to(device)
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            return torch.tensor(x, dtype=dtype, device=device)
    
    return {
        'device': device,
        'device_idx': device_idx,
        'repo_root': repo_root,
        'to_torch': to_torch
    }
