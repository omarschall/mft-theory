import itertools
from functools import reduce

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
