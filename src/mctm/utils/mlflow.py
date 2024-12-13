"""Mlflow utils."""

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
# REQUIRED MODULES ############################################################
import logging
import os
import tempfile
import traceback
from contextlib import contextmanager

import mlflow
import numpy as np
import tensorflow as tf

from mctm.utils import filter_recursive, flatten_dict


# PUBLIC FUNCTIONS ############################################################
def log_cfg(cfg: dict):
    """Log flattened dict as parameters in the current MLflow run.

    :param dict cfg: Flattened dictionary of parameters and values to be logged.
    :return: None
    """
    filtered_dict = filter_recursive(
        lambda x: not callable(x)
        and not isinstance(
            x,
            (type, np.ndarray, tf.Tensor, tf.data.Dataset, tf.keras.callbacks.Callback),
        ),
        cfg,
    )
    mlflow.log_dict(filtered_dict, "params.yaml")
    flat_dict = flatten_dict(filtered_dict)
    flat_dict = dict(filter(lambda xy: len(str(xy[1])) < 500, flat_dict.items()))
    mlflow.log_params(flat_dict)


@contextmanager
def start_run_with_exception_logging(**kwds):
    """Context manager for running MLflow experiment.

    This function starts an MLflow run within a context and logs any
    unhandled exceptions that occur within the context as run
    tags and artifacts.

    :param str run_name: The name of the run in MLflow.
    :return: A context manager for running MLflow experiments.
    :rtype: contextlib.ExitStack

    Example:
    -------
        with start_run_with_exception_logging(run_name="My_Run"):
            # Your code here

    """
    # if there is already a parent run, start it first
    run_id = os.environ.get("MLFLOW_RUN_ID", False)
    if run_id:
        mlflow.start_run()

    run = mlflow.start_run(nested=mlflow.active_run() is not None, **kwds)
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


def log_and_save_figure(figure, figure_path, file_name, file_format, **kwargs):
    """Save figure to disk and log it with mlflow."""
    figure.savefig(
        os.path.join(figure_path, f"{file_name}.{file_format}"),
        **kwargs,
    )
    mlflow.log_figure(figure, f"{file_name}.svg")
