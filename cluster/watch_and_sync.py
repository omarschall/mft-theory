"""
File watcher that automatically syncs code to cluster when files change.
Run this in the background to keep your cluster code up-to-date.
"""

import time
import subprocess
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from cluster.sync_cluster import sync_cluster, sync_columbia_cluster

class ClusterSyncHandler(FileSystemEventHandler):
    """Handler that syncs to cluster when files change."""
    
    def __init__(self, cluster='columbia', debounce_seconds=2, verbose=True):
        """
        Args:
            cluster: 'columbia' or 'nyu' (or 'greene')
            debounce_seconds: Wait this long after last change before syncing
            verbose: Print sync messages
        """
        self.cluster = cluster
        self.debounce_seconds = debounce_seconds
        self.verbose = verbose
        self.last_change_time = 0
        self.sync_timer = None
        self.pending_sync = False
        
    def should_sync_file(self, file_path):
        """Check if a file should trigger a sync."""
        path = Path(file_path)
        
        # Skip hidden files and directories
        if any(part.startswith('.') for part in path.parts):
            return False
        
        # Skip common non-code files
        skip_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.egg', '.swp', '.swo'}
        if path.suffix in skip_extensions:
            return False
        
        # Skip __pycache__ directories
        if '__pycache__' in path.parts:
            return False
        
        # Skip .git directory
        if '.git' in path.parts:
            return False
        
        return True
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if not self.should_sync_file(event.src_path):
            return
        
        self.last_change_time = time.time()
        self.pending_sync = True
        
        if self.verbose:
            print(f"📝 Detected change: {event.src_path}")
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        if not self.should_sync_file(event.src_path):
            return
        
        self.last_change_time = time.time()
        self.pending_sync = True
        
        if self.verbose:
            print(f"➕ New file: {event.src_path}")
    
    def on_deleted(self, event):
        if event.is_directory:
            return
        
        if not self.should_sync_file(event.src_path):
            return
        
        self.last_change_time = time.time()
        self.pending_sync = True
        
        if self.verbose:
            print(f"🗑️  Deleted: {event.src_path}")
    
    def check_and_sync(self):
        """Check if enough time has passed since last change, then sync."""
        if not self.pending_sync:
            return
        
        time_since_change = time.time() - self.last_change_time
        if time_since_change >= self.debounce_seconds:
            self.pending_sync = False
            if self.verbose:
                print(f"🔄 Syncing to {self.cluster} cluster...")
            
            if self.cluster in ['columbia', 'axon']:
                success = sync_columbia_cluster(verbose=self.verbose)
            else:
                success = sync_cluster(verbose=self.verbose)
            
            if success and self.verbose:
                print("✅ Sync complete\n")


def watch_and_sync(cluster='columbia', debounce_seconds=2, watch_path=None, verbose=True):
    """
    Start watching for file changes and auto-syncing to cluster.
    
    Args:
        cluster: 'columbia' or 'nyu' (or 'greene')
        debounce_seconds: Wait this long after last change before syncing
        watch_path: Directory to watch (defaults to repo root)
        verbose: Print sync messages
    
    Returns:
        Observer instance (call observer.stop() to stop watching)
    """
    if watch_path is None:
        # Auto-detect repo root
        current = Path(__file__).resolve()
        while current != current.parent:
            if (current / '.git').exists():
                watch_path = str(current)
                break
            current = current.parent
        if watch_path is None:
            watch_path = str(Path(__file__).resolve().parent.parent)
    
    event_handler = ClusterSyncHandler(cluster=cluster, 
                                      debounce_seconds=debounce_seconds,
                                      verbose=verbose)
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=True)
    
    # Start a timer thread to check for pending syncs
    import threading
    def sync_checker():
        while observer.is_alive():
            event_handler.check_and_sync()
            time.sleep(0.5)
    
    checker_thread = threading.Thread(target=sync_checker, daemon=True)
    checker_thread.start()
    
    observer.start()
    
    if verbose:
        print(f"👀 Watching {watch_path} for changes...")
        print(f"   Auto-syncing to {cluster} cluster")
        print(f"   Debounce: {debounce_seconds}s")
        print("   Press Ctrl+C to stop\n")
    
    return observer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Watch for file changes and auto-sync to cluster')
    parser.add_argument('--cluster', choices=['columbia', 'nyu', 'greene', 'axon'],
                       default='columbia', help='Which cluster to sync to')
    parser.add_argument('--debounce', type=float, default=2.0,
                       help='Seconds to wait after last change before syncing')
    parser.add_argument('--path', type=str, default=None,
                       help='Directory to watch (defaults to repo root)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress output messages')
    
    args = parser.parse_args()
    
    # Normalize cluster name
    if args.cluster in ['nyu', 'greene']:
        cluster = 'nyu'
    else:
        cluster = 'columbia'
    
    try:
        observer = watch_and_sync(cluster=cluster,
                                 debounce_seconds=args.debounce,
                                 watch_path=args.path,
                                 verbose=not args.quiet)
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping file watcher...")
        observer.stop()
        observer.join()
        print("✅ Stopped")
