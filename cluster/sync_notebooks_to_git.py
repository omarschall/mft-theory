#!/usr/bin/env python3
"""
Sync notebooks from cluster to a local git repository for version control.

Handles notebooks in subdirectories of /home/om2382/low-rank-dims/notebooks/
and maintains the directory structure in the local git repo.
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path so we can import cluster modules
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

def sync_notebooks_to_git(local_repo_path, cluster='columbia', project='low-rank-dims',
                          username=None, commit=True, push=False, verbose=True):
    """
    Sync all notebooks from cluster to local git repo, maintaining directory structure.
    
    Args:
        local_repo_path: Path to local git repository
        cluster: 'columbia' or 'nyu'
        project: Project name
        username: Override default username
        commit: Whether to commit changes
        push: Whether to push to remote
        verbose: Print status messages
    """
    local_repo_path = os.path.expanduser(local_repo_path)
    
    # Check if it's a git repo
    if not os.path.exists(os.path.join(local_repo_path, '.git')):
        if verbose:
            print(f"⚠️  {local_repo_path} is not a git repository")
            response = input("Initialize git repository? (y/N): ")
            if response.lower() == 'y':
                subprocess.run(['git', 'init'], cwd=local_repo_path, check=True)
                if verbose:
                    print(f"✅ Initialized git repository")
            else:
                print("❌ Aborted")
                return False
    
    # Determine remote paths
    if cluster == 'columbia':
        if username is None:
            username = 'om2382'
        domain = 'axon.rc.zi.columbia.edu'
        base_path = '/home/{}'.format(username)
    else:
        if username is None:
            username = 'oem214'
        domain = 'greene.hpc.nyu.edu'
        base_path = '/scratch/{}'.format(username)
    
    notebook_dir = os.path.join(base_path, project, 'notebooks')
    remote = '{}@{}'.format(username, domain)
    remote_path = f'{remote}:{notebook_dir}/'
    
    if verbose:
        print(f"📥 Syncing notebooks from cluster using rsync...")
        print(f"   From: {remote_path}")
        print(f"   To: {local_repo_path}")
        print(f"   (Only transfers changed files - much faster!)")
    
    # Use rsync to sync only .ipynb files, maintaining directory structure
    # --include='*/' : include all directories
    # --include='*.ipynb' : include all .ipynb files
    # --exclude='*' : exclude everything else
    # --prune-empty-dirs : remove empty directories
    rsync_cmd = [
        'rsync',
        '-aav',  # archive, archive (verbose), verbose
        '--include=*/',  # Include directories
        '--include=*.ipynb',  # Include .ipynb files
        '--exclude=*',  # Exclude everything else
        '--prune-empty-dirs',  # Remove empty directories
        remote_path,
        local_repo_path
    ]
    
    result = subprocess.run(rsync_cmd, capture_output=not verbose)
    
    if result.returncode != 0:
        if verbose:
            print(f"❌ Sync failed")
            if not verbose:
                print(result.stderr.decode())
        return False
    
    # Count synced files
    find_result = subprocess.run(
        ['find', '.', '-name', '*.ipynb', '-type', 'f'],
        cwd=local_repo_path,
        capture_output=True,
        text=True
    )
    
    synced_count = len(find_result.stdout.strip().split('\n')) if find_result.stdout.strip() else 0
    
    if verbose:
        print(f"✅ Synced {synced_count} notebook(s) (rsync only transfers changed files)")
    
    # Git operations
    if synced_count > 0 and commit:
        if verbose:
            print(f"\n📝 Committing changes to git...")
        
        # Add all notebook files (including in subdirectories)
        # Use find to get all .ipynb files and add them
        find_result = subprocess.run(
            ['find', '.', '-name', '*.ipynb', '-type', 'f'],
            cwd=local_repo_path,
            capture_output=True,
            text=True
        )
        
        if find_result.stdout.strip():
            # Add each file individually to ensure they're tracked
            files_to_add = find_result.stdout.strip().split('\n')
            for file_path in files_to_add:
                subprocess.run(['git', 'add', file_path],
                            cwd=local_repo_path,
                            capture_output=not verbose)
        else:
            # Fallback: try git add with proper globbing
            subprocess.run(['git', 'add', '-A', '*.ipynb'],
                         cwd=local_repo_path,
                         capture_output=not verbose)
            # Also try adding from subdirectories
            subprocess.run(['sh', '-c', 'find . -name "*.ipynb" -exec git add {} \\;'],
                         cwd=local_repo_path,
                         capture_output=not verbose)
        
        # Check if there are changes
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              cwd=local_repo_path,
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            # Commit
            commit_msg = f"Sync notebooks from cluster"
            commit_result = subprocess.run(['git', 'commit', '-m', commit_msg],
                         cwd=local_repo_path,
                         capture_output=not verbose)
            
            if commit_result.returncode == 0:
                if verbose:
                    print(f"✅ Committed changes")
            else:
                if verbose:
                    print(f"⚠️  Commit failed")
                    if not verbose:
                        print(commit_result.stderr.decode())
            
            # Push if requested
            if push:
                if verbose:
                    print(f"📤 Pushing to remote...")
                push_result = subprocess.run(['git', 'push'],
                                          cwd=local_repo_path,
                                          capture_output=True,
                                          text=True)
                if push_result.returncode == 0:
                    if verbose:
                        print(f"✅ Pushed to remote")
                else:
                    if verbose:
                        print(f"⚠️  Push failed")
                        if push_result.stderr:
                            print(f"   Error: {push_result.stderr}")
                        print(f"   Try: cd {local_repo_path} && git push")
        else:
            if verbose:
                print(f"ℹ️  No changes to commit")
    
    # Push even if no new commits (in case there are unpushed commits)
    if push and commit:
        # Check if there are unpushed commits
        push_check = subprocess.run(['git', 'log', '--oneline', '@{u}..HEAD'],
                                  cwd=local_repo_path,
                                  capture_output=True,
                                  text=True)
        if push_check.stdout.strip():
            if verbose:
                print(f"📤 Pushing unpushed commits...")
            push_result = subprocess.run(['git', 'push'],
                                      cwd=local_repo_path,
                                      capture_output=True,
                                      text=True)
            if push_result.returncode == 0:
                if verbose:
                    print(f"✅ Pushed to remote")
            else:
                if verbose:
                    print(f"⚠️  Push failed")
                    if push_result.stderr:
                        print(f"   Error: {push_result.stderr}")
    
    return synced_count > 0

def init_notebooks_repo(repo_path, remote_url=None, verbose=True):
    """
    Initialize a new git repository for notebooks.
    
    Args:
        repo_path: Path where to create the repo
        remote_url: Optional remote URL to add
        verbose: Print status messages
    """
    repo_path = os.path.expanduser(repo_path)
    os.makedirs(repo_path, exist_ok=True)
    
    if os.path.exists(os.path.join(repo_path, '.git')):
        if verbose:
            print(f"ℹ️  Git repository already exists at {repo_path}")
        return True
    
    if verbose:
        print(f"📦 Initializing git repository at {repo_path}...")
    
    subprocess.run(['git', 'init'], cwd=repo_path, check=True)
    
    # Create .gitignore for non-notebook files
    # Pattern: ignore everything, but allow directories and .ipynb files
    gitignore_path = os.path.join(repo_path, '.gitignore')
    with open(gitignore_path, 'w') as f:
        f.write("""# Ignore everything except notebooks
# Ignore all files
*
# But allow directories (so we can traverse into them)
!*/
# Allow .ipynb files anywhere
!*.ipynb
# Allow .gitignore itself
!.gitignore
""")
    
    subprocess.run(['git', 'add', '.gitignore'], cwd=repo_path)
    subprocess.run(['git', 'commit', '-m', 'Initial commit: add .gitignore'], cwd=repo_path)
    
    if remote_url:
        subprocess.run(['git', 'remote', 'add', 'origin', remote_url], cwd=repo_path)
        if verbose:
            print(f"✅ Added remote: {remote_url}")
    
    if verbose:
        print(f"✅ Repository initialized")
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sync notebooks from cluster to local git repository',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize a new notebooks repo
  python cluster/sync_notebooks_to_git.py init ~/notebooks_repo
  
  # Sync notebooks to existing repo
  python cluster/sync_notebooks_to_git.py sync ~/notebooks_repo
  
  # Sync and push to remote
  python cluster/sync_notebooks_to_git.py sync ~/notebooks_repo --push
  
  # Sync without committing
  python cluster/sync_notebooks_to_git.py sync ~/notebooks_repo --no-commit

Note: This maintains the subdirectory structure from the cluster.
        """
    )
    
    parser.add_argument('action', choices=['init', 'sync'],
                       help='Action: init (new repo) or sync (existing repo)')
    parser.add_argument('repo_path', type=str,
                       help='Path to local git repository')
    parser.add_argument('--cluster', choices=['columbia', 'nyu'],
                       default='columbia', help='Which cluster')
    parser.add_argument('--project', default='low-rank-dims',
                       help='Project name')
    parser.add_argument('--username', type=str, default=None,
                       help='Override default username')
    parser.add_argument('--remote-url', type=str, default=None,
                       help='Remote URL (for init)')
    parser.add_argument('--no-commit', action='store_true',
                       help='Sync without committing')
    parser.add_argument('--push', action='store_true',
                       help='Push to remote after committing')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    cluster = 'nyu' if args.cluster == 'nyu' else 'columbia'
    
    if args.action == 'init':
        success = init_notebooks_repo(args.repo_path, args.remote_url, 
                                     verbose=not args.quiet)
        sys.exit(0 if success else 1)
    elif args.action == 'sync':
        success = sync_notebooks_to_git(args.repo_path, cluster, args.project,
                                       args.username, 
                                       commit=not args.no_commit,
                                       push=args.push,
                                       verbose=not args.quiet)
        sys.exit(0 if success else 1)
