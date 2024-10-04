"""Util functions."""

# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ###########################################################
# file    : utils.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2023-06-19 14:44:17 (Marcel Arpogaus)
# changed : 2023-06-19 17:08:07 (Marcel Arpogaus)
# DESCRIPTION ##################################################################
# ...
# LICENSE ######################################################################
# ...
################################################################################
import argparse
import importlib
from collections.abc import Mapping, MutableMapping, MutableSequence


# FUNCTION DEFINITIONS #########################################################
def flatten_dict(d, parent_key="", sep="."):
    """Recursively flatten a nested dictionary.

    This function takes a nested dictionary and flattens it by concatenating
    keys with the specified separator. It is a utility for working with
    configuration dictionaries and similar structures.

    :param dict d: The input nested dictionary to be flattened.
    :param str parent_key: Used for recursion, indicating the parent key.
    :param str sep: The separator used to concatenate keys.
    :return: A flattened dictionary.
    :rtype: dict
    """
    items = []
    for key, value in d.items():
        new_key = sep.join((parent_key, str(key))) if parent_key else str(key)
        if isinstance(value, Mapping) and len(value):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def str2bool(v):
    """Convert a string to a boolean value.

    This function converts a string representing a boolean value
    ("true", "false", "1", "0", "yes", "no", etc.) to a Python boolean
    value (True or False).

    :param str v: The input string to be converted.
    :return: The boolean value based on the input string.
    :rtype: bool
    :raises argparse.ArgumentTypeError: If the input string cannot be converted
                                        to a boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def filter_recursive(filter_func, collection):
    """Recursively filters out all non-iterable values in a collection.

    Parameters
    ----------
    filter_func : callable
        The filter function to apply to each non-iterable value.
    collection : iterable
        An iterable on which the filter function should be applied.

    Returns
    -------
    iterable
        A new collection only containing values for which `filter_func` evaluates
        to `True`.

    Examples
    --------
    >>> def greater_than_one(x):
    ...     return x > 1
    >>> d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}, 'f': 1}}
    >>> filter_recursive(greater_than_one, d)
    {'b': {'c': 2, 'd': {'e': 3}}}


    """
    if isinstance(collection, Mapping):
        return {
            k: v_filtered
            for k, v in collection.copy().items()
            if (v_filtered := filter_recursive(filter_func, v)) is not None
        }
    elif isinstance(collection, MutableSequence):
        return [
            v_filterd
            for v in collection.copy()
            if (v_filterd := filter_recursive(filter_func, v)) is not None
        ]
    else:
        return collection if filter_func(collection) else None


def getattr_from_module(module_attr):
    module_attr = (
        module_attr.replace("tfb.", "tensorflow_probability.python.bijectors.", 1)
        .replace("tfd.", "tensorflow_probability.python.distributions.", 1)
        .replace("tfp.", "tensorflow_probability.python.", 1)
        .replace("tf.", "tensorflow.", 1)
    )
    module_name, attr = module_attr.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def deepupdate(d: MutableMapping, u: Mapping):
    """Recursively update elements in a Mapping.

    References
    ----------
    https://stackoverflow.com/a/3233356

    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = deepupdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d
