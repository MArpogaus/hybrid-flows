# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : mlflow.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2023-01-16 12:47:35 (Marcel Arpogaus)
# changed : 2021-03-26 11:48:25 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################
# REQUIRED MODULES #############################################################
import logging
import tempfile
import traceback
from contextlib import contextmanager

import mlflow

from mctm.utils import flatten_dict


# PUBLIC FUNCTIONS ############################################################
def log_cfg(cfg: dict):
    """log flattened dictionary as mlflow params"""
    flat_dict = flatten_dict(cfg)
    flat_dict = dict(filter(lambda xy: len(str(xy[1])) < 500, flat_dict.items()))
    mlflow.log_params(flat_dict)


@contextmanager
def start_run_with_exception_logging(run_name):
    run = mlflow.start_run(
        run_name=run_name,
        nested=mlflow.active_run() is not None,
    )
    try:
        yield run
    except Exception as e:
        logging.error("Run falid", exc_info=e)

        with tempfile.NamedTemporaryFile(prefix="traceback", suffix=".txt") as tmpf:
            with open(tmpf.name, "w+") as f:
                f.write(traceback.format_exc())
            mlflow.log_artifact(tmpf.name)

        mlflow.end_run(status="FAILED")
    finally:
        mlflow.end_run(status="FINISHED")
