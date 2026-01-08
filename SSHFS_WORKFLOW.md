# SSHFS Notebook Mount - Workflow Guide

## Quick Reference

### ✅ Safe Operations (via SSHFS mount)
- **Editing individual files** in Cursor
- **Viewing files** and browsing structure
- **Small file operations** (create, edit, delete single files)

### ⚠️ Use Direct SSH Instead (via `cluster_file_ops.py`)
- **Bulk file moves** (moving many files at once)
- **Reorganizing directories** (creating folders, moving groups)
- **Any operation on many files** (can cause "Device not configured" errors)

## Common Workflows

### 1. Edit Notebooks in Cursor
```bash
# Mount (do once, leave mounted)
python cluster/mount_notebooks.py mount

# Open ~/cluster_notebooks/ in Cursor
# Edit files - changes sync to cluster in real-time
```

### 2. Bulk File Operations
```bash
# List notebooks
python cluster/cluster_file_ops.py list

# Move notebooks (dry run first!)
python cluster/cluster_file_ops.py move "*.ipynb" old_notebooks --dry-run
python cluster/cluster_file_ops.py move "*.ipynb" old_notebooks

# Move specific pattern
python cluster/cluster_file_ops.py move "LDRG_*.ipynb" ldrg_notebooks
```

### 3. If Mount Looks Wrong
```bash
# Remount to refresh view
python cluster/mount_notebooks.py unmount
python cluster/mount_notebooks.py mount

# Verify files are on cluster (bypass mount)
ssh om2382@axon.rc.zi.columbia.edu "ls /home/om2382/low-rank-dims/notebooks/*.ipynb | wc -l"
```

## Safety Rules

1. **Remote is source of truth**: The cluster filesystem is always authoritative
2. **Failed operations don't delete**: If you see "Device not configured" errors, nothing was deleted
3. **Remount to refresh**: If the mount looks wrong, remount to see current remote state
4. **Bulk ops via SSH**: Use `cluster_file_ops.py` for bulk operations, not the mount

## Troubleshooting

### Mount appears empty but files exist on cluster
```bash
# Remount
python cluster/mount_notebooks.py unmount
python cluster/mount_notebooks.py mount
```

### "Device not configured" errors
- This means the mount had an issue
- **Nothing was deleted** - your files are safe on the cluster
- Use `cluster_file_ops.py` for bulk operations instead

### Need to reorganize files
- Use `cluster/cluster_file_ops.py` or SSH directly
- Don't use the mount for bulk moves

## Best Practices

1. **Mount once, leave mounted** - No need to unmount regularly
2. **Edit in Cursor** - Use the mount for editing individual files
3. **Bulk ops via SSH** - Use `cluster_file_ops.py` for reorganizing
4. **Run in Jupyter** - Use browser Jupyter for execution (as you do now)
