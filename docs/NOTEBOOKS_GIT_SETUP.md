# Notebooks Git Repository Setup Guide

## Step 1: Create GitHub Repository (if you haven't already)

1. Go to GitHub and create a new repository (e.g., `cluster-notebooks`)
2. **Don't** initialize it with README, .gitignore, or license (we'll do that locally)

## Step 2: Add Remote to Your Local Repo

```bash
cd ~/notebooks_repo
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
# Or if using SSH:
# git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPO_NAME.git
```

## Step 3: Initial Sync from Cluster

```bash
# Sync all notebooks from cluster (this will download everything)
cd /Users/omarschall/mft-theory
python cluster/sync_notebooks_to_git.py sync ~/notebooks_repo

# This will:
# - Download all notebooks from cluster (including subdirectories)
# - Maintain directory structure
# - Commit changes to git
```

## Step 4: Push to GitHub

```bash
cd ~/notebooks_repo
git push -u origin main
# Or if your default branch is 'master':
# git push -u origin master
```

## Regular Workflow

### Sync Notebooks from Cluster

```bash
# From mft-theory directory
python cluster/sync_notebooks_to_git.py sync ~/notebooks_repo

# Or sync and push in one command:
python cluster/sync_notebooks_to_git.py sync ~/notebooks_repo --push
```

### What Gets Synced

- All `.ipynb` files from `/home/om2382/low-rank-dims/notebooks/` and all subdirectories
- Directory structure is preserved (e.g., `old_notebooks/my_notebook.ipynb`)
- Only notebooks are synced (data files are excluded)

### Git Workflow

The script automatically:
1. Downloads all notebooks from cluster
2. Adds them to git
3. Commits with message "Sync notebooks from cluster (X files)"
4. Optionally pushes to remote (if `--push` flag used)

## Troubleshooting

### If remote already exists
```bash
cd ~/notebooks_repo
git remote set-url origin <your-new-url>
```

### If you want to see what would be synced first
```bash
# List notebooks on cluster
python cluster/list_cluster_notebooks.py --list --cluster columbia
```

### Manual git operations
```bash
cd ~/notebooks_repo
git status          # See what changed
git log            # See commit history
git push           # Push manually
```
