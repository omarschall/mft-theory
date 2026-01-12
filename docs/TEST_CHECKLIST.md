# Test Checklist for Workflow Improvements

Follow this checklist to verify all improvements work correctly before merging to main.

## Prerequisites

- [ ] Conda environment `mft-theory` is created and activated
- [ ] You're in the `mft-theory` repository directory
- [ ] Chrome browser is installed (for Jupyter browser test)

---

## Test 1: Notebook Setup Function

### 1.1 Basic Import
```bash
conda activate mft-theory
python -c "from notebook_setup import setup_mft_theory; print('✅ Import successful')"
```
- [ ] Command runs without errors
- [ ] Prints "✅ Import successful"

### 1.2 Dependency Checking
```bash
python -c "from notebook_setup import check_dependencies; check_dependencies()"
```
- [ ] Command runs without errors
- [ ] Shows status of all dependencies (PyTorch, NumPy, SciPy, Matplotlib)

### 1.3 Repository Root Detection
```bash
python -c "from notebook_setup import find_repo_root; print('Repo:', find_repo_root())"
```
- [ ] Command runs without errors
- [ ] Prints a valid path containing "mft-theory"

---

## Test 2: Jupyter Notebook Setup

### 2.1 Start Jupyter
```bash
conda activate mft-theory
jupyter notebook
```
- [ ] Jupyter starts without errors
- [ ] **Opens in Chrome** (not Safari) ✅
- [ ] File browser shows `mft-theory` directory

### 2.2 Template Notebook - Basic Setup
1. Open `template_notebook.ipynb` in Jupyter
2. Run the first cell (setup cell):
   ```python
   from notebook_setup import setup_mft_theory
   setup = setup_mft_theory(use_gpu=False, import_cluster=True, import_empirics=True)
   device = setup['device']
   device_idx = setup['device_idx']
   to_torch = setup['to_torch']
   ```
- [ ] Cell executes without errors
- [ ] Shows output: "📂 Using repo at: [path]"
- [ ] Shows output: "💻 Using CPU..."
- [ ] Shows output: "✅ Core modules imported"
- [ ] Shows output: "✅ Cluster tools available" (or appropriate message)

### 2.3 Template Notebook - Verify Imports
Run the second cell (check what's available):
```python
print(f"Device: {device}")
print(f"Device index: {device_idx}")
print(f"NumPy version: {np.__version__}")
print(f"PyTorch version: {torch.__version__}")
```
- [ ] Cell executes without errors
- [ ] Prints device information
- [ ] Prints NumPy and PyTorch versions
- [ ] No import errors for `np` or `torch`

### 2.4 Template Notebook - Example Code
Run the example code cells (theory computation and simulation):
- [ ] All cells execute successfully
- [ ] No errors or warnings
- [ ] Output is reasonable (numbers, plots, etc.)

### 2.5 Notebook UI Check
- [ ] Notebook UI looks good (classic notebook, not the "crappy" version)
- [ ] No scrolling issues
- [ ] Cells run smoothly

---

## Test 3: Dependency Installation

### 3.1 Requirements.txt (Optional - pip users)
```bash
# Create a test virtual environment
python -m venv test_env
source test_env/bin/activate  # On macOS/Linux
pip install -r scripts/requirements.txt
```
- [ ] `scripts/requirements.txt` exists and is readable
- [ ] Pip install completes without critical errors
- [ ] Can import: `python -c "import torch, numpy, matplotlib; print('OK')"`

**Note:** This is optional - conda is recommended, but good to verify pip option works.

---

## Test 4: Cluster Sync (If Needed)

### 4.1 One-Time Sync
```bash
python scripts/cluster_sync.py --cluster columbia
```
- [ ] Command runs without errors
- [ ] Shows sync progress/output
- [ ] Files sync to cluster (verify on cluster if possible)

### 4.2 Auto-Sync Watcher (Quick Test)
```bash
# In one terminal, start watcher:
python scripts/cluster_sync.py --watch --cluster columbia --debounce 2

# In another terminal, make a test change:
touch test_sync_file.py
echo "# test" >> test_sync_file.py

# Wait 3-5 seconds, then check watcher terminal
```
- [ ] Watcher starts without errors
- [ ] Detects file change
- [ ] Syncs after debounce period
- [ ] Press Ctrl+C to stop watcher

**Note:** You can skip this if you don't need to test cluster sync right now.

---

## Test 5: Documentation

### 5.1 README.md
- [ ] Open `README.md`
- [ ] Quick start section is clear
- [ ] Installation instructions make sense
- [ ] Links to other docs work

### 5.2 NOTEBOOK_PORTABILITY.md
- [ ] File exists and is readable
- [ ] Examples are clear
- [ ] Troubleshooting section is helpful

### 5.3 JUPYTER_SETUP.md
- [ ] File exists and is readable
- [ ] Instructions for Chrome setup are clear

---

## Test 6: Error Handling

### 6.1 Missing Dependencies (Simulated)
```python
# In a Python shell, temporarily rename a package to test error handling
# (Or just verify the error messages are helpful)
```
- [ ] Error messages are clear and helpful
- [ ] Suggest installation commands
- [ ] Point to `scripts/environment.yml`

**Note:** This is optional - mainly verify error messages in code look good.

---

## Test 7: Cross-Platform Compatibility

### 7.1 Path Detection
- [ ] `find_repo_root()` works from different directories
- [ ] Works when notebook is in a subdirectory
- [ ] Falls back gracefully if repo not found

### 7.2 Device Selection
- [ ] `use_gpu=False` works (CPU mode)
- [ ] `use_gpu=True` falls back to CPU if GPU unavailable (graceful)

---

## Final Verification

### Before Merging
- [ ] All critical tests pass (Tests 1, 2 are most important)
- [ ] No obvious errors or warnings
- [ ] Documentation is accurate
- [ ] Code is clean (no linter errors - already checked ✅)

### Git Status Check
```bash
git status
```
- [ ] Review all changed files
- [ ] No unexpected files
- [ ] Ready to commit/merge

---

## Quick Test (5 minutes)

If you're short on time, just do these essential tests:

1. **Test 2.2** - Run template notebook setup cell ✅
2. **Test 2.3** - Verify imports work ✅
3. **Test 2.1** - Jupyter opens in Chrome ✅

If these pass, you're good to go!

---

## Issues Found?

If you find any issues:
1. Note them here:
   - Issue 1: [description]
   - Issue 2: [description]

2. We can fix before merging, or
3. Document as known limitations

---

## Ready to Merge?

Once all tests pass:
```bash
git checkout main
git merge <current-branch>
git push origin main
```
