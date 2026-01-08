#!/usr/bin/env python3
"""
Helper script for bulk file operations on the cluster.

Use this instead of SSHFS for bulk moves/reorganizations to avoid
"Device not configured" errors. SSHFS is great for editing individual
files, but bulk operations should be done directly on the cluster.
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path

def run_cluster_command(command, cluster='columbia', username=None, verbose=True):
    """
    Run a command on the cluster via SSH.
    
    Args:
        command: Command to run (will be executed in bash -c)
        cluster: 'columbia' or 'nyu'
        username: Override default username
        verbose: Print command and output
    """
    if cluster == 'columbia':
        if username is None:
            username = 'om2382'
        domain = 'axon.rc.zi.columbia.edu'
    else:  # nyu/greene
        if username is None:
            username = 'oem214'
        domain = 'greene.hpc.nyu.edu'
    
    remote = '{}@{}'.format(username, domain)
    
    if verbose:
        print(f"🔧 Running on {cluster} cluster: {command}")
    
    result = subprocess.run(
        ['ssh', remote, f'bash -c "{command}"'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        if verbose and result.stdout:
            print(result.stdout)
        return True, result.stdout
    else:
        if verbose:
            print(f"❌ Command failed")
            if result.stderr:
                print(result.stderr)
        return False, result.stderr

def move_notebooks(pattern, dest_dir, cluster='columbia', project='low-rank-dims', 
                   username=None, dry_run=False):
    """
    Move notebooks matching a pattern to a destination directory.
    
    Args:
        pattern: File pattern (e.g., '*.ipynb' or 'LDRG_*.ipynb')
        dest_dir: Destination directory (relative to notebooks/ or absolute)
        cluster: 'columbia' or 'nyu'
        project: Project name
        username: Override default username
        dry_run: Show what would be moved without actually moving
    """
    if cluster == 'columbia':
        if username is None:
            username = 'om2382'
        base_path = '/home/{}'.format(username)
    else:
        if username is None:
            username = 'oem214'
        base_path = '/scratch/{}'.format(username)
    
    notebook_dir = os.path.join(base_path, project, 'notebooks')
    
    # Make dest_dir absolute if it's relative
    if not dest_dir.startswith('/'):
        dest_dir = os.path.join(notebook_dir, dest_dir)
    
    # Create destination directory
    if not dry_run:
        success, _ = run_cluster_command(f'mkdir -p "{dest_dir}"', cluster, username, verbose=False)
        if not success:
            print(f"❌ Failed to create destination directory")
            return False
    
    # Find and move files
    find_cmd = f'find "{notebook_dir}" -maxdepth 1 -name "{pattern}" -type f'
    move_cmd = f'{find_cmd} -exec mv {{}} "{dest_dir}/" \\;'
    
    if dry_run:
        print(f"🔍 Dry run - would move files matching '{pattern}' to {dest_dir}")
        success, output = run_cluster_command(find_cmd, cluster, username, verbose=True)
        if success and output.strip():
            print(f"\nFiles that would be moved:")
            for line in output.strip().split('\n'):
                print(f"  {line}")
        else:
            print("  (no files found)")
        return True
    else:
        print(f"📦 Moving files matching '{pattern}' to {dest_dir}...")
        success, output = run_cluster_command(move_cmd, cluster, username, verbose=True)
        if success:
            print(f"✅ Move completed")
        return success

def list_notebooks(pattern='*.ipynb', cluster='columbia', project='low-rank-dims',
                   username=None):
    """List notebooks matching a pattern."""
    if cluster == 'columbia':
        if username is None:
            username = 'om2382'
        base_path = '/home/{}'.format(username)
    else:
        if username is None:
            username = 'oem214'
        base_path = '/scratch/{}'.format(username)
    
    notebook_dir = os.path.join(base_path, project, 'notebooks')
    find_cmd = f'find "{notebook_dir}" -maxdepth 1 -name "{pattern}" -type f | sort'
    
    success, output = run_cluster_command(find_cmd, cluster, username, verbose=False)
    if success and output.strip():
        files = output.strip().split('\n')
        print(f"📓 Found {len(files)} file(s) matching '{pattern}':\n")
        for f in files:
            print(f"  {os.path.basename(f)}")
        return files
    else:
        print(f"📭 No files found matching '{pattern}'")
        return []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Bulk file operations on cluster (safer than SSHFS for bulk ops)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all notebooks
  python cluster/cluster_file_ops.py list
  
  # List specific pattern
  python cluster/cluster_file_ops.py list --pattern "LDRG_*.ipynb"
  
  # Move all notebooks to old_notebooks/ (dry run first)
  python cluster/cluster_file_ops.py move "*.ipynb" old_notebooks --dry-run
  python cluster/cluster_file_ops.py move "*.ipynb" old_notebooks
  
  # Move specific pattern
  python cluster/cluster_file_ops.py move "LDRG_*.ipynb" ldrg_notebooks
  
  # Run custom command on cluster
  python cluster/cluster_file_ops.py exec "ls -lh /home/om2382/low-rank-dims/notebooks/ | head -10"

Note: Use this for bulk operations. For editing individual files, use the
SSHFS mount in Cursor.
        """
    )
    
    parser.add_argument('action', choices=['list', 'move', 'exec'],
                       help='Action to perform')
    parser.add_argument('--pattern', type=str, default='*.ipynb',
                       help='File pattern (for list/move)')
    parser.add_argument('--dest', type=str, default=None,
                       help='Destination directory (for move)')
    parser.add_argument('--command', type=str, default=None,
                       help='Command to execute (for exec)')
    parser.add_argument('--cluster', choices=['columbia', 'nyu', 'greene'],
                       default='columbia', help='Which cluster')
    parser.add_argument('--project', type=str, default='low-rank-dims',
                       help='Project name')
    parser.add_argument('--username', type=str, default=None,
                       help='Override default username')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without doing it')
    
    args = parser.parse_args()
    
    cluster = 'nyu' if args.cluster in ['nyu', 'greene'] else 'columbia'
    
    if args.action == 'list':
        list_notebooks(args.pattern, cluster, args.project, args.username)
    elif args.action == 'move':
        if not args.dest:
            print("❌ Error: --dest is required for move action")
            sys.exit(1)
        move_notebooks(args.pattern, args.dest, cluster, args.project, 
                      args.username, args.dry_run)
    elif args.action == 'exec':
        if not args.command:
            print("❌ Error: --command is required for exec action")
            sys.exit(1)
        success, _ = run_cluster_command(args.command, cluster, args.username)
        sys.exit(0 if success else 1)
