#!/usr/bin/env python
"""
Test script for notebook_setup.py
Simulates notebook behavior to verify setup works correctly.
"""

import sys
import os
# Add repo root to path so we can import from scripts
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from scripts.notebook_setup import check_dependencies, find_repo_root, setup_mft_theory

# Test 1: Check dependencies
print("=" * 60)
print("Test 1: Dependency Checking")
print("=" * 60)
deps_ok = check_dependencies()
print(f"✅ Dependencies check: {'PASS' if deps_ok else 'WARN'}\n")

# Test 2: Find repo root
print("=" * 60)
print("Test 2: Repository Root Detection")
print("=" * 60)
repo_root = find_repo_root()
print(f"📂 Found repo at: {repo_root}")
print(f"✅ Repo exists: {os.path.exists(repo_root)}")
if 'mft-theory' in repo_root.lower():
    print("✅ Repo root detection: PASS\n")
else:
    print("⚠️  Repo root might be incorrect\n")

# Test 3: Setup function (in notebook-like context)
print("=" * 60)
print("Test 3: Setup Function (simulated notebook context)")
print("=" * 60)

# Simulate notebook globals
notebook_globals = {}
exec("""
from scripts.notebook_setup import setup_mft_theory
setup = setup_mft_theory(use_gpu=False, import_cluster=False, import_empirics=False, check_deps=False)
print(f"Device: {setup['device']}")
print(f"Device idx: {setup['device_idx']}")
print(f"Repo root: {setup['repo_root']}")
print(f"to_torch function: {callable(setup['to_torch'])}")
print("✅ Setup function: PASS")
""", {'__name__': '__main__', '__file__': __file__}, notebook_globals)

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)
