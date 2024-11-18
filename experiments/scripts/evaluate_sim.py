# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# %% Author ####################################################################
# file    : evaluate_sim.py
# author  : Marcel Arpogaus <znepry.necbtnhf@tznvy.pbz>
#
# created : 2024-11-18 14:16:47 (Marcel Arpogaus)
# changed : 2024-11-18 16:07:20 (Marcel Arpogaus)

# %% License ###################################################################

# %% Description ###############################################################
"""Train multivariate density estimation models on different datasets."""

# %% imports ###################################################################
import argparse
import logging
import os
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from mctm.data.sklearn_datasets import get_dataset
from mctm.models import DensityRegressionModel, HybridDensityRegressionModel
from mctm.utils.pipeline import prepare_pipeline
from mctm.utils.visualisation import setup_latex

__LOGGER__ = logging.getLogger(__name__)


# %% functions #################################################################
def pdf_contour_plot(model, figure_path, fig_ext, n=200):
    ls = np.linspace(0, 1, n, dtype=np.float32)
    xx, yy = np.meshgrid(ls[..., None], ls[..., None])
    grid = np.stack([xx.flatten(), yy.flatten()], -1)

    if True:
        joint_dist_0 = model.joint_distribution(tf.convert_to_tensor(0.0))
        joint_dist_1 = model.joint_distribution(tf.convert_to_tensor(1.0))
        p_y_0 = joint_dist_0.prob(grid).numpy().reshape(-1, n)
        p_y_1 = joint_dist_1.prob(grid).numpy().reshape(-1, n)
        p_y = p_y_0 + p_y_1
    else:
        joint_dist = model.joint_distribution(None)
        p_y = joint_dist.prob(grid).numpy().reshape(-1, n)

    fig = plt.figure(figsize=plt.figaspect(1))
    plt.contourf(
        xx,
        yy,
        p_y,
        cmap="plasma",
    )
    plt.axis("off")
    fig.savefig(
        os.path.join(figure_path, f"contour.{fig_ext}"),
        bbox_inches="tight",
        transparent=True,
    )


def evaluate(
    dataset_name: str,
    dataset_type: str,
    results_path: str,
    params: dict,
) -> tuple:
    """Execute experiment.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    dataset_type : str
        Type of the dataset.
    results_path : str
        Destination for model checkpoints and logs.
    params : dict
        Dictionary containing experiment parameters.

    Returns
    -------
    tuple
        Experiment results: history, model, and preprocessed data.

    """
    __LOGGER__.info(f"{tf.__version__=}\n{tfp.__version__=}")
    tf.config.set_visible_devices([], "GPU")

    model_kwargs = params["model_kwargs"]
    dataset_kwargs = params["dataset_kwargs"][dataset_name]
    figure_path = os.path.join(results_path, "eval_figures")
    os.makedirs(figure_path, exist_ok=True)

    fig_ext = "pdf"

    data, dims = get_dataset(dataset_name, **dataset_kwargs)

    # def preprocess_dataset(data, model) -> dict:
    #     return {
    #         "x": tf.convert_to_tensor(data[1][..., None], dtype=model.dtype),
    #         "y": tf.convert_to_tensor(data[0], dtype=model.dtype),
    #     }

    if "marginal_bijectors" in model_kwargs.keys():
        get_model = HybridDensityRegressionModel
    else:
        get_model = DensityRegressionModel

    model = get_model(dims=dims, **model_kwargs)
    model.load_weights(os.path.join(results_path, "model_checkpoint.weights.h5"))

    if which("latex"):
        __LOGGER__.info("Using latex backend for plotting")
        setup_latex(fontsize=10)

    pdf_contour_plot(model, figure_path, fig_ext)


# %% if main ###################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "--log-file",
        type=str,
        help="path for log file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging severity level",
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        help="name of dvc stage",
        required=True,
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        help="type of dataset",
        required=True,
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="name of dataset",
        required=True,
    )
    parser.add_argument(
        "--results-path",
        type=str,
        help="destination for model checkpoints and logs.",
        required=True,
    )

    args = parser.parse_args()
    __LOGGER__.info("CLI arguments: %s", vars(args))

    params = prepare_pipeline(
        results_path=args.results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        stage_name_or_params_file_path=args.stage_name,
    )

    evaluate(
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        results_path=args.results_path,
        params=params,
    )
