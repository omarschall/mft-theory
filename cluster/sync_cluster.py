import subprocess, os
from pathlib import Path

def _get_repo_root():
    """Auto-detect the repository root by looking for .git directory."""
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / '.git').exists():
            return str(current) + '/'
        current = current.parent
    # Fallback: use the directory containing this file's parent
    return str(Path(__file__).resolve().parent.parent) + '/'

def sync_cluster(local_module_path=None,
                 module_name='mft-theory',
                 username='oem214', domain='greene.hpc.nyu.edu',
                 verbose=True):
    """Sync local code with module path on cluster.
    
    Args:
        local_module_path: Path to local repo. If None, auto-detects from git root.
        module_name: Name of module directory on cluster
        username: Cluster username
        domain: Cluster domain
        verbose: Print sync status
    """

    if local_module_path is None:
        local_module_path = _get_repo_root()
    
    if not local_module_path.endswith('/'):
        local_module_path += '/'

    scratch_path = '/scratch/{}/'.format(username)
    module_path = os.path.join(scratch_path, module_name)
    remote_path = '{}@{}:{}'.format(username, domain, module_path)
    
    if verbose:
        print(f"Syncing {local_module_path} to {remote_path}")
    
    # Use --checksum to force comparison by content, not just timestamp
    # This ensures file content changes (like deleted lines) are synced
    # --inplace forces in-place updates which ensures files are overwritten
    result = subprocess.run(['rsync', '-av', '--delete', '--checksum', '--inplace',
                    '--exclude', '.git',
                    '--exclude', 'files',
                    '--exclude', '__pycache__',
                    '--exclude', '*.pyc',
                    '--exclude', '.ipynb_checkpoints',
                    local_module_path, remote_path],
                   capture_output=not verbose)
    
    if verbose and result.returncode == 0:
        print("✓ Sync completed successfully")
    elif result.returncode != 0:
        print(f"✗ Sync failed with return code {result.returncode}")
        if not verbose:
            print(result.stderr.decode())
    
    return result.returncode == 0

def sync_columbia_cluster(local_module_path=None,
                          module_name='mft-theory',
                          username='om2382', domain='axon.rc.zi.columbia.edu',
                          remote_base_path=None,
                          verbose=True):
    """Sync local code with module path on cluster.
    
    Args:
        local_module_path: Path to local repo. If None, auto-detects from git root.
        module_name: Name of module directory on cluster
        username: Cluster username
        domain: Cluster domain
        remote_base_path: Base path on cluster (default: tries /share/lkumar/users/{username}/ then /home/{username}/)
        verbose: Print sync status
    """

    if local_module_path is None:
        local_module_path = _get_repo_root()
    
    if not local_module_path.endswith('/'):
        local_module_path += '/'


    # Try /share/lkumar/users/ first (where notebooks are actually looking), then fall back to /home/
    if remote_base_path is None:
        # Check if /share/lkumar/users/ path exists
        test_path = '/share/lkumar/users/{}/{}'.format(username, module_name)
        test_result = subprocess.run(
            ['ssh', '{}@{}'.format(username, domain), f'test -d {test_path}'],
            capture_output=True
        )
        if test_result.returncode == 0:
            remote_base_path = '/share/lkumar/users/{}/'.format(username)
        else:
            remote_base_path = '/home/{}/'.format(username)
    
    module_path = os.path.join(remote_base_path, module_name)
    remote_path = '{}@{}:{}'.format(username, domain, module_path)
    
    print(f"Local module path: {local_module_path}")

    if verbose:
        print(f"Syncing {local_module_path} to {remote_path}")
    
    # Use --checksum to force comparison by content, not just timestamp
    # This ensures file content changes (like deleted lines) are synced
    # --inplace forces in-place updates which ensures files are overwritten
    result = subprocess.run(['rsync', '-av', '--delete', '--checksum', '--inplace',
                    '--exclude', '.git',
                    '--exclude', 'files',
                    '--exclude', '__pycache__',
                    '--exclude', '*.pyc',
                    '--exclude', '.ipynb_checkpoints',
                    local_module_path, remote_path],
                   capture_output=not verbose)
    
    if verbose and result.returncode == 0:
        print("✓ Sync completed successfully")
    elif result.returncode != 0:
        print(f"✗ Sync failed with return code {result.returncode}")
        if not verbose:
            print(result.stderr.decode())
    
    return result.returncode == 0