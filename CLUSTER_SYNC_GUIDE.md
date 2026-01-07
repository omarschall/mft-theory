# Cluster Sync Improvements - Guide

## What Changed

### 1. Enhanced Sync Functions (`cluster/sync_cluster.py`)
- **Auto-detects repo path**: No need to hardcode `/Users/omarschall/mft-theory/`
- **Better error handling**: Shows success/failure messages
- **More exclusions**: Skips `__pycache__`, `.pyc`, `.ipynb_checkpoints`

### 2. Auto-Sync File Watcher (`cluster/watch_and_sync.py`)
- **Watches for file changes**: Automatically syncs when you edit files
- **Debouncing**: Waits 2 seconds after last change before syncing (configurable)
- **Smart filtering**: Only syncs code files, skips temp files

### 3. CLI Tool (`cluster_sync.py`)
- Simple command-line interface for syncing
- Can do one-time sync or start the watcher

### 4. Notebook Access Tool (`cluster/list_cluster_notebooks.py`)
- **List notebooks**: See what notebooks exist on the cluster
- **Download notebooks**: Copy notebooks from cluster to local repo
- **Sync all**: Download all notebooks at once

## How to Use

### One-Time Sync
```bash
python cluster_sync.py --cluster columbia
```

### Auto-Sync (Recommended)
```bash
# Start watcher (runs until Ctrl+C)
python cluster_sync.py --watch --cluster columbia
```

### Access Cluster Notebooks
```bash
# List notebooks on cluster (only .ipynb files, excludes data files)
python cluster/list_cluster_notebooks.py --list --cluster columbia

# List with file sizes
python cluster/list_cluster_notebooks.py --list-sizes --cluster columbia

# Download a specific notebook
python cluster/list_cluster_notebooks.py --download my_notebook.ipynb --cluster columbia

# Download all notebooks (data files in subdirectories are excluded)
python cluster/list_cluster_notebooks.py --sync-all --cluster columbia
```

**Important**: The notebook sync tool **only downloads `.ipynb` files** to avoid syncing large data files that may be in subdirectories of `notebooks/`. This prevents repository bloat.

## Reversibility

All changes are on branch `cluster-sync-improvements`:

**To test:**
- You're already on the branch! Just try the commands above.

**To revert:**
```bash
git checkout main
```

**To keep changes:**
```bash
# Option 1: Merge into main
git checkout main
git merge cluster-sync-improvements

# Option 2: Commit on this branch
git commit -m "Add cluster sync improvements"
```

## Requirements

For auto-sync watcher:
```bash
pip install watchdog
```

## What's Safe

- ✅ All changes are backward compatible
- ✅ Existing functions still work (with better defaults)
- ✅ No changes to core code, only cluster utilities
- ✅ All on a separate git branch

## Next Steps

1. **Test the sync tools** - Try `python cluster_sync.py --watch`
2. **Download cluster notebooks** - Use `list_cluster_notebooks.py` to see what's there
3. **If everything works** - Merge the branch or commit
4. **If something breaks** - Just `git checkout main`
