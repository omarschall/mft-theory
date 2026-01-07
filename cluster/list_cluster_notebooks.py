"""
Helper script to list and download Jupyter notebooks from the cluster.

NOTE: This script ONLY syncs .ipynb files to avoid downloading large data files.
Data files in subdirectories of notebooks/ are intentionally excluded.
"""

import subprocess
import os
import argparse
from pathlib import Path

def list_notebooks(cluster='columbia', project_name='low-rank-dims', username=None, show_sizes=False):
    """
    List Jupyter notebooks on the cluster.
    
    NOTE: Only lists .ipynb files, not data files in subdirectories.
    
    Args:
        cluster: 'columbia' or 'nyu'
        project_name: Project directory name
        username: Override default username
        show_sizes: Show file sizes (slower, requires extra SSH call)
    """
    if cluster == 'columbia':
        if username is None:
            username = 'om2382'
        domain = 'axon.rc.zi.columbia.edu'
        base_path = '/home/{}'.format(username)
        notebook_dir = os.path.join(base_path, project_name, 'notebooks')
    else:  # nyu/greene
        if username is None:
            username = 'oem214'
        domain = 'greene.hpc.nyu.edu'
        base_path = '/scratch/{}'.format(username)
        notebook_dir = os.path.join(base_path, project_name, 'jupyter_notebook')
    
    remote = '{}@{}'.format(username, domain)
    
    print(f"📂 Listing .ipynb files in {notebook_dir} on {cluster} cluster...")
    print("   (Data files in subdirectories are excluded)\n")
    
    # List ONLY .ipynb files - this excludes all data files
    result = subprocess.run(
        ['ssh', remote, f'find {notebook_dir} -name "*.ipynb" -type f 2>/dev/null'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Error: Could not access {notebook_dir}")
        print(f"   Error: {result.stderr}")
        return []
    
    notebooks = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
    
    if notebooks:
        print(f"📓 Found {len(notebooks)} notebook(s):\n")
        
        if show_sizes:
            # Get file sizes (this is slower but informative)
            for i, nb in enumerate(notebooks, 1):
                size_result = subprocess.run(
                    ['ssh', remote, f'ls -lh "{nb}" | awk \'{{print $5}}\''],
                    capture_output=True,
                    text=True
                )
                size = size_result.stdout.strip() if size_result.returncode == 0 else '?'
                print(f"  {i}. {nb} ({size})")
        else:
            for i, nb in enumerate(notebooks, 1):
                print(f"  {i}. {nb}")
    else:
        print(f"📭 No .ipynb files found in {notebook_dir}")
    
    return notebooks

def download_notebook(notebook_path, cluster='columbia', username=None, 
                      local_dir=None, project_name='low-rank-dims'):
    """
    Download a notebook from the cluster.
    
    NOTE: Only downloads .ipynb files. Data files are excluded.
    
    Args:
        notebook_path: Full path to notebook on cluster, or just filename
        cluster: 'columbia' or 'nyu'
        username: Override default username
        local_dir: Where to save locally (defaults to repo root)
        project_name: Project name (used if notebook_path is just a filename)
    """
    if cluster == 'columbia':
        if username is None:
            username = 'om2382'
        domain = 'axon.rc.zi.columbia.edu'
        base_path = '/home/{}'.format(username)
        notebook_dir = os.path.join(base_path, project_name, 'notebooks')
    else:  # nyu/greene
        if username is None:
            username = 'oem214'
        domain = 'greene.hpc.nyu.edu'
        base_path = '/scratch/{}'.format(username)
        notebook_dir = os.path.join(base_path, project_name, 'jupyter_notebook')
    
    remote = '{}@{}'.format(username, domain)
    
    # If just a filename, construct full path
    if not notebook_path.startswith('/'):
        remote_path = os.path.join(notebook_dir, notebook_path)
    else:
        remote_path = notebook_path
    
    # Safety check: ensure it's a .ipynb file
    if not remote_path.endswith('.ipynb'):
        print(f"⚠️  Warning: {remote_path} doesn't end with .ipynb")
        print("   This script only downloads notebook files to avoid large data files.")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Cancelled.")
            return None
    
    # Determine local save location
    if local_dir is None:
        # Auto-detect repo root
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / '.git').exists():
                local_dir = str(current)
                break
            current = current.parent
        if local_dir is None:
            local_dir = str(Path(__file__).resolve().parent.parent)
    
    filename = os.path.basename(remote_path)
    local_path = os.path.join(local_dir, filename)
    
    print(f"⬇️  Downloading {remote_path} to {local_path}...")
    
    result = subprocess.run(
        ['scp', f'{remote}:{remote_path}', local_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"✅ Downloaded to {local_path}")
        return local_path
    else:
        print(f"❌ Download failed: {result.stderr}")
        return None

def sync_all_notebooks(cluster='columbia', project_name='low-rank-dims', 
                      username=None, local_dir=None):
    """
    Download all notebooks from cluster.
    
    NOTE: Only downloads .ipynb files. Data files in subdirectories are excluded.
    """
    notebooks = list_notebooks(cluster, project_name, username, show_sizes=False)
    
    if not notebooks:
        return []
    
    print(f"\n⬇️  Downloading {len(notebooks)} notebook(s) (data files excluded)...\n")
    
    downloaded = []
    for nb_path in notebooks:
        result = download_notebook(nb_path, cluster, username, local_dir, project_name)
        if result:
            downloaded.append(result)
    
    print(f"\n✅ Successfully downloaded {len(downloaded)}/{len(notebooks)} notebook(s)")
    print("   (Only .ipynb files were downloaded; data files were excluded)")
    return downloaded

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='List and download Jupyter notebooks from cluster',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List notebooks on Columbia cluster (only .ipynb files, excludes data)
  python cluster/list_cluster_notebooks.py --list --cluster columbia
  
  # List with file sizes
  python cluster/list_cluster_notebooks.py --list-sizes --cluster columbia
  
  # Download a specific notebook
  python cluster/list_cluster_notebooks.py --download my_notebook.ipynb --cluster columbia
  
  # Download all notebooks (data files excluded)
  python cluster/list_cluster_notebooks.py --sync-all --cluster columbia

Note: This script ONLY syncs .ipynb files to avoid downloading large data files
      that may be in subdirectories of notebooks/.
        """
    )
    
    parser.add_argument('--cluster', choices=['columbia', 'nyu', 'greene'],
                       default='columbia', help='Which cluster')
    parser.add_argument('--project', default='low-rank-dims',
                       help='Project name (default: low-rank-dims)')
    parser.add_argument('--username', default=None,
                       help='Override default username')
    parser.add_argument('--list', action='store_true',
                       help='List notebooks on cluster (only .ipynb files)')
    parser.add_argument('--list-sizes', action='store_true',
                       help='List notebooks with file sizes (slower)')
    parser.add_argument('--download', type=str, metavar='NOTEBOOK',
                       help='Download a specific notebook (filename or full path, must be .ipynb)')
    parser.add_argument('--sync-all', action='store_true',
                       help='Download all notebooks from cluster (only .ipynb files, excludes data)')
    parser.add_argument('--local-dir', type=str, default=None,
                       help='Local directory to save notebooks (default: repo root)')
    
    args = parser.parse_args()
    
    # Normalize cluster name
    cluster = 'nyu' if args.cluster in ['nyu', 'greene'] else 'columbia'
    
    if args.list_sizes:
        list_notebooks(cluster, args.project, args.username, show_sizes=True)
    elif args.list:
        list_notebooks(cluster, args.project, args.username, show_sizes=False)
    elif args.download:
        download_notebook(args.download, cluster, args.username, 
                         args.local_dir, args.project)
    elif args.sync_all:
        sync_all_notebooks(cluster, args.project, args.username, args.local_dir)
    else:
        parser.print_help()
