# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : decorators.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-10-09 06:38:42 (Marcel Arpogaus)
# changed : 2024-10-18 17:33:05 (Marcel Arpogaus)

# %% Description ###############################################################
"""Module for useful decorator functions."""

# %% imports ###################################################################
from functools import partial
from typing import Any, Callable, Dict, List

import tensorflow as tf
from tensorflow_probability.python.internal import prefer_static


# %% functions #################################################################
def skip_no_dict(fn: Callable) -> Callable:
    """Decorate a function to skip function execution if input is a bijector.

    Parameters
    ----------
    fn : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function.

    """

    def process(entry: Any) -> Any:
        if isinstance(entry, dict):
            return fn(entry)
        else:
            return entry

    return process


def reduce_dict(result_key: str, keep_keys: List[str] = []) -> Callable:
    """Decorate a function to reduce a dictionary to a single result value.

    The decorator takes a function with a dictionary as input and
    returns a dictionary with the same keys as the input dictionary.

    The values of the keys in `keep_keys` are copied from the input
    dictionary. The value of the key `result_key` is set to the result
    of the decorated function.

    Parameters
    ----------
    result_key : str
        The key of the result value.
    keep_keys : List[str], optional
        The keys to keep from the input dictionary, by default [].

    Returns
    -------
    Callable
        The decorated function.

    """

    def decorator(fn: Callable) -> Callable:
        @skip_no_dict
        def process(entry: Dict[str, Any]) -> Dict[str, Any]:
            result = {k: entry[k] for k in keep_keys if k in entry}
            result[result_key] = fn(**entry)
            return result

        return process

    return decorator


def recurse_on_key(key: str) -> Callable:
    """Decorate a function to apply a function recursively to a dictionary key.

    Parameters
    ----------
    key : str
        The dictionary key to recurse on.

    Returns
    -------
    Callable
        The decorated function.

    """

    def decorator(fn: Callable) -> Callable:
        def process(entry: Dict[str, Any]) -> Dict[str, Any]:
            results = entry.copy()
            if isinstance(entry, dict) and key in entry.keys():
                nested_node = entry.get(key)
                if isinstance(nested_node, dict):
                    nested_node = [nested_node]
                results[key] = list(map(process, nested_node))
            return fn(results)

        return process

    return decorator


def vectorized_map_if_batched(fn: Callable) -> Callable:
    """Decorate a function to use `tf.vectorized_map` if the input tensor is batched.

    Parameters
    ----------
    fn : Callable
        Function to be decorated

    Raises
    ------
    ValueError
        If the input tensors ranks is not in [2, 3]

    Returns
    -------
    Callable
        The decorated function.

    """

    def dec_fn(y: tf.Tensor, *_: Any, **kwargs: Dict[str, Any]):
        y_rank = prefer_static.rank(y)
        if y_rank == 2:
            return fn(y, **kwargs)
        elif y_rank == 3:
            return tf.vectorized_map(partial(fn, **kwargs), y)
        else:
            raise ValueError("Unsupported input rank: %s", y_rank)

    return dec_fn
