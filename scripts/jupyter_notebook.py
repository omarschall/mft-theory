import sys
import os
# Add repo root to path so we can import cluster module
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from cluster import start_jupyter_notebook, close_jupyter_notebook, start_axon_jupyter_notebook
import argparse

### Open or close a jupyter notebook depending on arguments passed when
### calling script

parser = argparse.ArgumentParser()
parser.add_argument('-open', dest='open_', action='store_true')
parser.add_argument('-close', dest='close_', action='store_true')
parser.add_argument('-axon', dest='axon_', action='store_true')
parser.add_argument('--time_in_hours', dest='time_in_hours')
parser.add_argument('--mem_in_gb', dest='mem_in_gb')
parser.add_argument('--n_gpus', dest='n_gpus')
parser.add_argument('--env_name', dest='env_name')
parser.set_defaults(open_=True, close_=False, axon_=True,
                    time_in_hours=3, mem_in_gb=16, n_gpus=0, env_name='torch-test-3')

args = parser.parse_args()

if args.open_ and not args.close_:

    if args.axon_:
        start_axon_jupyter_notebook(time_in_hours=args.time_in_hours,
                                    mem_in_gb=args.mem_in_gb,
                                    n_gpus=args.n_gpus,
                                    env_name=args.env_name)
    else:
        start_jupyter_notebook()

if args.close_:

    if args.axon_:
        close_jupyter_notebook(username='om2382', domain='axon.rc.zi.columbia.edu')
    else:
        close_jupyter_notebook()