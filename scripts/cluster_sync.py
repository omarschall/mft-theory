#!/usr/bin/env python3
"""
Simple CLI for syncing code to cluster and managing auto-sync watcher.
"""

import argparse
import sys
import os
from pathlib import Path

# Add repo root to path so we can import cluster module
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from cluster.sync_cluster import sync_cluster, sync_columbia_cluster
from cluster.watch_and_sync import watch_and_sync

def main():
    parser = argparse.ArgumentParser(
        description='Sync code to HPC cluster or start auto-sync watcher',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # One-time sync to Columbia cluster
  python scripts/cluster_sync.py --cluster columbia
  
  # One-time sync to NYU cluster
  python scripts/cluster_sync.py --cluster nyu
  
  # Start auto-sync watcher (runs in background)
  python scripts/cluster_sync.py --watch --cluster columbia
  
  # Watch with custom debounce time
  python scripts/cluster_sync.py --watch --cluster columbia --debounce 5
        """
    )
    
    parser.add_argument('--cluster', choices=['columbia', 'nyu', 'greene', 'axon'],
                       default='columbia',
                       help='Which cluster to sync to (default: columbia)')
    parser.add_argument('--watch', action='store_true',
                       help='Start file watcher for auto-sync (runs until Ctrl+C)')
    parser.add_argument('--debounce', type=float, default=2.0,
                       help='Seconds to wait after last change before syncing (default: 2.0)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    
    args = parser.parse_args()
    
    # Normalize cluster name
    if args.cluster in ['nyu', 'greene']:
        cluster = 'nyu'
    else:
        cluster = 'columbia'
    
    if args.watch:
        # Start file watcher
        import time
        try:
            observer = watch_and_sync(cluster=cluster,
                                     debounce_seconds=args.debounce,
                                     watch_path=None,
                                     verbose=not args.quiet)
            
            # Keep running until interrupted
            while True:
                time.sleep(1)
        except ImportError as e:
            print(f"❌ {e}")
            print("\n💡 Tip: You can still use one-time sync without watchdog:")
            print(f"   python scripts/cluster_sync.py --cluster {cluster}")
            sys.exit(1)
        except KeyboardInterrupt:
            if not args.quiet:
                print("\n🛑 Stopping file watcher...")
            observer.stop()
            observer.join()
            if not args.quiet:
                print("✅ Stopped")
    else:
        # One-time sync
        if cluster == 'columbia':
            success = sync_columbia_cluster(verbose=not args.quiet)
        else:
            success = sync_cluster(verbose=not args.quiet)
        
        sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
