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


# FUNCTION DEFINITIONS #########################################################
def flatten_dict(d, parent_key="", sep="."):
    """
    Recursively flattens a nested dictionary.

    This function takes a nested dictionary and flattens it by concatenating
    keys with the specified separator. It is a utility for working with
    configuration dictionaries and similar structures.

    Parameters:
        d (dict): The input nested dictionary to be flattened.
        parent_key (str): Used for recursion, indicating the parent key.
        sep (str): The separator used to concatenate keys.

    Returns:
        dict: A flattened dictionary.

    """

    items = []
    for key, value in d.items():
        new_key = sep.join((parent_key, str(key))) if parent_key else str(key)
        if isinstance(value, dict) and len(value):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def str2bool(v):
    """
    Converts a string to a boolean value.

    This function converts a string representing a boolean value
    ("true", "false", "1", "0", "yes", "no", etc.) to a Python boolean
    value (True or False).

    Parameters:
        v (str): The input string to be converted.

    Returns:
        bool: The boolean value based on the input string.

    Raises:
        argparse.ArgumentTypeError: If the input string cannot be converted
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
