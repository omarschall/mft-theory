#!/usr/bin/env python3
"""
Alternative to SSHFS: Sync notebooks for editing when VPN blocks SSHFS.

This downloads a notebook, lets you edit it locally, then syncs it back.
Useful when VPN prevents SSHFS from working.
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path
from cluster.list_cluster_notebooks import download_notebook
from cluster.sync_cluster import sync_columbia_cluster

def edit_notebook_sync(notebook_name, cluster='columbia', project='low-rank-dims',
                       local_edit_dir=None, verbose=True):
    """
    Download notebook, edit locally, then sync back.
    
    Args:
        notebook_name: Name of notebook (e.g., 'my_notebook.ipynb' or path)
        cluster: 'columbia' or 'nyu'
        project: Project name
        local_edit_dir: Where to download for editing (defaults to ~/notebook_edits/)
        verbose: Print status
    """
    if local_edit_dir is None:
        local_edit_dir = os.path.expanduser('~/notebook_edits')
    
    os.makedirs(local_edit_dir, exist_ok=True)
    
    if verbose:
        print(f"📥 Downloading {notebook_name} for editing...")
    
    # Download notebook
    local_path = download_notebook(notebook_name, cluster=cluster, 
                                   local_dir=local_edit_dir, 
                                   project_name=project)
    
    if not local_path:
        return False
    
    if verbose:
        print(f"\n✅ Notebook downloaded to: {local_path}")
        print(f"   📝 Edit it in Cursor, then run:")
        print(f"   python cluster/sync_notebook_edit.py upload {os.path.basename(local_path)}")
    
    return local_path

def upload_notebook(notebook_name, cluster='columbia', project='low-rank-dims',
                    local_edit_dir=None, verbose=True):
    """
    Upload edited notebook back to cluster.
    
    Args:
        notebook_name: Name of notebook file (must be in local_edit_dir)
        cluster: 'columbia' or 'nyu'
        project: Project name
        local_edit_dir: Where edited notebook is (defaults to ~/notebook_edits/)
        verbose: Print status
    """
    if local_edit_dir is None:
        local_edit_dir = os.path.expanduser('~/notebook_edits')
    
    local_path = os.path.join(local_edit_dir, notebook_name)
    
    if not os.path.exists(local_path):
        print(f"❌ Notebook not found: {local_path}")
        return False
    
    if cluster == 'columbia':
        username = 'om2382'
        domain = 'axon.rc.zi.columbia.edu'
        base_path = '/home/{}'.format(username)
    else:
        username = 'oem214'
        domain = 'greene.hpc.nyu.edu'
        base_path = '/scratch/{}'.format(username)
    
    remote_path = os.path.join(base_path, project, 'notebooks', notebook_name)
    remote = '{}@{}'.format(username, domain)
    
    if verbose:
        print(f"📤 Uploading {local_path} to cluster...")
    
    result = subprocess.run(
        ['scp', local_path, f'{remote}:{remote_path}'],
        capture_output=not verbose
    )
    
    if result.returncode == 0:
        if verbose:
            print(f"✅ Successfully uploaded to cluster")
        return True
    else:
        if verbose:
            print(f"❌ Upload failed")
            if not verbose:
                print(result.stderr.decode())
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Sync notebooks for editing when VPN blocks SSHFS',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download notebook for editing
  python cluster/sync_notebook_edit.py download my_notebook.ipynb
  
  # After editing, upload it back
  python cluster/sync_notebook_edit.py upload my_notebook.ipynb
  
  # Or do both in one command (download, edit, then upload manually)
  python cluster/sync_notebook_edit.py edit my_notebook.ipynb
        """
    )
    
    parser.add_argument('action', choices=['download', 'upload', 'edit'],
                       help='Action: download, upload, or edit (download then wait)')
    parser.add_argument('notebook', type=str,
                       help='Notebook name (e.g., my_notebook.ipynb)')
    parser.add_argument('--cluster', choices=['columbia', 'nyu'],
                       default='columbia', help='Which cluster')
    parser.add_argument('--project', default='low-rank-dims',
                       help='Project name')
    parser.add_argument('--edit-dir', type=str, default=None,
                       help='Local directory for editing (default: ~/notebook_edits/)')
    
    args = parser.parse_args()
    
    cluster = 'nyu' if args.cluster == 'nyu' else 'columbia'
    
    if args.action == 'download' or args.action == 'edit':
        success = edit_notebook_sync(args.notebook, cluster, args.project, 
                                    args.edit_dir)
        sys.exit(0 if success else 1)
    elif args.action == 'upload':
        success = upload_notebook(args.notebook, cluster, args.project, args.edit_dir)
        sys.exit(0 if success else 1)
