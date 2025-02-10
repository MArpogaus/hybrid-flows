# %% Description ###############################################################
"""Mlflow utils."""

# %% imports ###################################################################
import logging
import os
import tempfile
import traceback
from contextlib import contextmanager

import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf

from mctm.utils import filter_recursive, flatten_dict


# %% functions #################################################################
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


def extract_data_from_figure(fig, xlabel="x", ylabel="y"):
    """Hacky helper function to get data back from mpl figure.

    Warning:
    -------
    This makes quite some assumptions about the plot structure and is note suitable
    to extract data from arbitrary plots.

    """
    dfs = []
    for ax in fig.get_axes():
        index = ax.title.get_text()
        new_xlabel, new_ylabel = ax.get_xlabel(), ax.get_ylabel()
        columns = [
            xlabel if new_xlabel == "" else new_xlabel,
            ylabel if new_ylabel == "" else new_ylabel,
        ]
        xlabel, ylabel = columns
        for i, line in enumerate(ax.lines):
            data = np.stack(line.get_data(), 1)
            df = pd.DataFrame(data=data, columns=columns).assign(
                var_name="-".join((index, str(i)))
            )
            dfs.append(df)
    return pd.concat(dfs).reset_index(drop=True)


def log_and_save_figure(
    figure, figure_path, file_name, file_format, extract_data=False, **kwargs
):
    """Save figure to disk and log it with mlflow."""
    figure.savefig(
        os.path.join(figure_path, f"{file_name}.{file_format}"),
        **kwargs,
    )
    mlflow.log_figure(figure, f"{file_name}.{file_format}")
    if extract_data:
        dat = extract_data_from_figure(figure)
        dat.to_csv(
            os.path.join(figure_path, f"{file_name}.csv"),
        )
        mlflow.log_artifact(
            os.path.join(figure_path, f"{file_name}.csv"),
        )
