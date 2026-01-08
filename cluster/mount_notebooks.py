#!/usr/bin/env python3
"""
Mount cluster notebooks directory locally via SSHFS for Cursor editing.

This mounts the remote notebooks directory so Cursor can see and edit them
with full project context. Changes are written to the cluster in real-time.

You can leave this mounted indefinitely - unmount only when you want to
stop using SSHFS (e.g., to free up resources or if you're done with it).
Unmounting is separate from code syncing (which happens via cluster_sync.py).

⚠️  IMPORTANT: For bulk file operations (moving many files, reorganizing),
    use cluster/cluster_file_ops.py instead of the mount. SSHFS can have
    "Device not configured" errors with bulk operations. The mount is best
    for editing individual files.
"""

import subprocess
import os
import sys
from pathlib import Path

def mount_notebooks(mount_point=None, remote_path='/home/om2382/low-rank-dims/notebooks',
                   remote='om2382@axon.rc.zi.columbia.edu', verbose=True):
    """
    Mount cluster notebooks directory locally.
    
    Args:
        mount_point: Local directory to mount to (defaults to ~/cluster_notebooks)
        remote_path: Path on cluster
        remote: Remote host
        verbose: Print status messages
    """
    if mount_point is None:
        mount_point = os.path.expanduser('~/cluster_notebooks')
    else:
        mount_point = os.path.expanduser(mount_point)
    
    # Check if already mounted
    if os.path.ismount(mount_point):
        if verbose:
            print(f"✅ Already mounted at {mount_point}")
        return True
    
    # Create mount point if it doesn't exist
    os.makedirs(mount_point, exist_ok=True)
    
    # Check if directory is empty (should be for mount point)
    if os.listdir(mount_point):
        if verbose:
            print(f"⚠️  Warning: {mount_point} is not empty. It should be empty for mounting.")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != 'y':
                return False
    
    # SSHFS mount command with optimized options
    # VPN-friendly options: shorter timeouts, more aggressive keepalive
    cmd = [
        'sshfs',
        '-o', 'noappledouble,noapplexattr',  # Don't create ._ files on cluster
        '-o', 'reconnect',                    # Auto-reconnect on disconnect
        '-o', 'ServerAliveInterval=5',        # More frequent keepalive (VPN-friendly)
        '-o', 'ServerAliveCountMax=10',       # More retries (VPN-friendly)
        '-o', 'ConnectTimeout=10',           # Connection timeout
        '-o', 'sshfs_sync',                   # Synchronous I/O (more reliable with VPN)
        '-o', 'idmap=user',                   # Map user IDs
        '-o', 'compression=yes',              # Compress data transfer
        '-o', 'cache=no',                     # Disable cache (VPN-friendly, more reliable)
        '-o', 'allow_other',                  # Allow access (if needed)
        f'{remote}:{remote_path}',
        mount_point
    ]
    
    if verbose:
        print(f"🔗 Mounting {remote}:{remote_path}")
        print(f"   to {mount_point}...")
    
    result = subprocess.run(cmd, capture_output=not verbose)
    
    if result.returncode == 0:
        if verbose:
            print(f"✅ Successfully mounted!")
            print(f"   📂 Open {mount_point} in Cursor to edit notebooks with full context")
            print(f"   💡 Changes are written to cluster in real-time")
            print(f"   💡 You can leave this mounted - unmount only when you're done using SSHFS")
        return True
    else:
        if verbose:
            print(f"❌ Mount failed")
            if not verbose:
                print(result.stderr.decode())
            print(f"\n💡 Troubleshooting:")
            print(f"   1. Make sure macFUSE and sshfs are installed")
            print(f"   2. Check SSH connection: ssh {remote}")
            print(f"   3. Verify remote path exists: ssh {remote} 'ls {remote_path}'")
        return False

def unmount_notebooks(mount_point=None, verbose=True):
    """
    Unmount cluster notebooks directory.
    
    Only needed when you want to stop using SSHFS (e.g., to free resources
    or if you're done with it). This is separate from code syncing.
    """
    if mount_point is None:
        mount_point = os.path.expanduser('~/cluster_notebooks')
    else:
        mount_point = os.path.expanduser(mount_point)
    
    if not os.path.ismount(mount_point):
        if verbose:
            print(f"ℹ️  {mount_point} is not mounted")
        return True
    
    if verbose:
        print(f"🔌 Unmounting {mount_point}...")
    
    result = subprocess.run(['umount', mount_point], capture_output=not verbose)
    
    if result.returncode == 0:
        if verbose:
            print(f"✅ Successfully unmounted")
        return True
    else:
        if verbose:
            print(f"❌ Unmount failed")
            if not verbose:
                print(result.stderr.decode())
            print(f"\n💡 Try: umount -f {mount_point}  (force unmount)")
        return False

def check_mount(mount_point=None):
    """Check if notebooks are mounted."""
    if mount_point is None:
        mount_point = os.path.expanduser('~/cluster_notebooks')
    else:
        mount_point = os.path.expanduser(mount_point)
    
    if os.path.ismount(mount_point):
        print(f"✅ Mounted at {mount_point}")
        return True
    else:
        print(f"❌ Not mounted")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Mount/unmount cluster notebooks for Cursor editing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mount notebooks (do this once, leave mounted)
  python cluster/mount_notebooks.py mount
  
  # Unmount notebooks (only if you want to stop using SSHFS)
  python cluster/mount_notebooks.py unmount
  
  # Check mount status
  python cluster/mount_notebooks.py status

Note: Unmounting is separate from code syncing. You can leave this mounted
indefinitely. Unmount only when you want to stop using SSHFS.
        """
    )
    
    parser.add_argument('action', choices=['mount', 'unmount', 'status'],
                       help='Action to perform')
    parser.add_argument('--mount-point', type=str, default=None,
                       help='Local mount point (default: ~/cluster_notebooks)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output')
    
    args = parser.parse_args()
    
    if args.action == 'mount':
        success = mount_notebooks(mount_point=args.mount_point, verbose=not args.quiet)
        sys.exit(0 if success else 1)
    elif args.action == 'unmount':
        success = unmount_notebooks(mount_point=args.mount_point, verbose=not args.quiet)
        sys.exit(0 if success else 1)
    elif args.action == 'status':
        check_mount(mount_point=args.mount_point)
