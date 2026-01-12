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

### 3. CLI Tool (`scripts/cluster_sync.py`)
- Simple command-line interface for syncing
- Can do one-time sync or start the watcher

### 4. Notebook Access Tool (`cluster/list_cluster_notebooks.py`) - Optional
- **List notebooks**: See what notebooks exist on the cluster
- **Download notebooks**: Copy notebooks from cluster to local repo (if you need them locally)
- **Note**: This is optional - you typically only need to sync code TO the cluster, not download notebooks FROM it

## How to Use

### One-Time Sync
```bash
python scripts/cluster_sync.py --cluster columbia
```

### Auto-Sync (Recommended)
```bash
# Start watcher (runs until Ctrl+C)
python scripts/cluster_sync.py --watch --cluster columbia
```

### Access Cluster Notebooks (Optional)

**Note**: You typically don't need to download notebooks from the cluster. The main workflow is:
- **Local code → Cluster**: So your notebooks on the cluster can import/use your code
- **Cluster notebooks → Local**: Only if you want to version control them or edit them locally

If you do want to download notebooks (e.g., to share with me for help, or for version control):
```bash
# List notebooks on cluster (only .ipynb files, excludes data files)
python cluster/list_cluster_notebooks.py --list --cluster columbia

# Download a specific notebook (e.g., to share with Cursor for help)
python cluster/list_cluster_notebooks.py --download my_notebook.ipynb --cluster columbia
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

**Note on Conda Environment**: If you need a specific conda environment (e.g., `restored_env_2`) to run these scripts, activate it first:
```bash
conda activate restored_env_2
python scripts/cluster_sync.py --cluster columbia
```

## What's Safe

- ✅ All changes are backward compatible
- ✅ Existing functions still work (with better defaults)
- ✅ No changes to core code, only cluster utilities
- ✅ All on a separate git branch

## Workflow Summary

**Main Use Case**: Sync your local code to the cluster so Jupyter notebooks running on the cluster can import and use your code.

1. **Edit code locally** (in your editor/Cursor)
2. **Sync to cluster** (automatically with `--watch` or manually)
3. **Use in cluster notebooks** - Your notebooks can now `import` your synced code

**Getting Help with Cluster Notebooks**: If you're editing a notebook on the cluster via browser and need help:
- Copy/paste the relevant notebook cells into the chat
- Or temporarily download the notebook: `python cluster/list_cluster_notebooks.py --download notebook.ipynb --cluster columbia`
- I can help you code, then you can copy the changes back to the browser

## Next Steps

1. **Test the sync tools** - Try `python scripts/cluster_sync.py --watch` (in your conda env if needed)
2. **If everything works** - Merge the branch or commit
3. **If something breaks** - Just `git checkout main`
