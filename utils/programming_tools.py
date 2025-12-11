import itertools
from functools import reduce
import numpy as np

def rgetattr(obj, attr):
    """A "recursive" version of getattr that can handle nested objects.

    Args:
        obj (object): Parent object
        attr (string): Address of desired attribute with '.' between child
            objects.
    Returns:
        The attribute of obj referred to."""

    return reduce(getattr, [obj] + attr.split('.'))

def config_generator(**kwargs):
    """Generator object that produces a Cartesian product of configurations.

    Each kwarg should be a list of possible values for the key. Yields a
    dictionary specifying a particular configuration."""

    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

def reverse_index_config(micro_config, configs_array):
    # Generate all micro-configurations using Cartesian product
    values = list(configs_array.values())
    micro_configs = list(itertools.product(*values))

    # Create a reverse index mapping
    reverse_index = {}
    for idx, mc in enumerate(micro_configs):
        reverse_index[mc] = idx

    # Example: get the index of a particular micro-configuration
    index_of_sample = reverse_index[micro_config]

    return index_of_sample

def populations_by_bitmask(all_loadings, on_slice=0):
    """
    Vectorized grouping by bitmask.
    Returns:
      combos: list of tuples length N_tasks with 0/1
      pops:   list of np.ndarray of neuron indices per combo, same order as combos
    """
    N_tasks, N = all_loadings.shape[0], all_loadings.shape[1]

    # Boolean task x neuron: True if neuron is "on" for that task
    on = (all_loadings[:, :, on_slice] != 0)  # shape [T, N]

    # Bitmask code per neuron: sum( on[t,j] << t )
    codes = (on.astype(np.uint32) * (1 << np.arange(N_tasks, dtype=np.uint32)[:, None])).sum(axis=0)

    # Collect indices for every possible code (0..2^T-1)
    combos = []
    pops = []
    for code in range(1 << N_tasks):
        idx = np.flatnonzero(codes == code)
        pops.append(idx)  # np.ndarray of neuron indices
        # decode code -> tuple of bits (0/1) in task order
        combo = tuple((code >> t) & 1 for t in range(N_tasks))
        combos.append(combo)

    return combos, pops