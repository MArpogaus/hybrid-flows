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
    items = []
    for key, value in d.items():
        new_key = sep.join((parent_key, str(key))) if parent_key else str(key)
        if isinstance(value, dict) and len(value):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
