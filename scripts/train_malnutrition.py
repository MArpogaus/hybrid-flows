"""Train malnutrinion."""
# IMPORT PACKAGES #############################################################
import argparse
import logging
import os
from functools import partial
from shutil import which

import numpy as np
import pandas as pd
import seaborn as sns
from mctm.data.malnutrion import get_dataset
from mctm.models import DensityRegressionModel, HybridDenistyRegressionModel
from mctm.utils import str2bool
from mctm.utils.pipeline import pipeline, prepare_pipeline
from mctm.utils.tensorflow import set_seed
from mctm.utils.visualisation import get_figsize, setup_latex
from tensorflow_probability import distributions as tfd

__LOGGER__ = logging.getLogger(__name__)


def plot_grid(data, **kwds):
    """Plot sns.PairGrid."""
    sns.set_theme(style="white")
    g = sns.PairGrid(data, diag_sharey=False, **kwds)
    g.map_upper(sns.scatterplot, s=15)

    g.map_lower(sns.kdeplot)
    g.map_diag(sns.kdeplot, lw=2)

    return g


def plot_data(*data, targets, frac=0.1, **kwds):
    """Plot data."""
    train_data, _, _ = data
    data = pd.DataFrame(np.array(train_data[1]), columns=targets).sample(frac=frac)

    g = plot_grid(data, **kwds)
    return g.figure


def get_after_fit_hook(results_path, N, seed, targets, **kwds):
    """Provide after after fit plot."""

    def plot_samples_grid(model, x, y, validation_data, **_):
        x, y = validation_data
        set_seed(seed)
        if targets is None:
            columns = [f"x{i}" for i in range(y.shape[-1])]
        else:
            columns = targets
        df_data = pd.DataFrame(y, columns=columns).sample(N).assign(source="data")
        df_model = (
            pd.DataFrame(model(x).sample(seed=seed).numpy().squeeze(), columns=columns)
            .assign(source="model")
            .sample(N)
        )
        df = pd.concat([df_data, df_model])

        g = plot_grid(df, hue="source", **kwds)
        g = g.add_legend()

        g.figure.savefig(os.path.join(results_path, "samples.pdf"))

    return plot_samples_grid


def main(args):
    """Experiment."""
    # prepare for execution:
    # - read cli arguments
    # - configure logging
    # - load dvc params
    params = prepare_pipeline(args)

    # store to variables
    dataset = args.dataset
    dataset_kwds = params[dataset + "_kwds"]  # [dataset]
    distribution = args.distribution
    stage = args.stage_name.split("@")[0]
    distribution_params = params[stage + "_distributions"][distribution]  # [dataset]
    distribution_kwds = distribution_params["distribution_kwds"]
    fit_kwds = distribution_params["fit_kwds"]
    parameter_kwds = distribution_params["parameter_kwds"]

    model_kwds = dict(
        distribution=distribution,
        distribution_kwds=distribution_kwds,
        parameter_kwds=parameter_kwds,
    )

    # prepare base_distribution if applicable
    if "base_distribution" in distribution_params.keys():
        get_model = HybridDenistyRegressionModel
        model_kwds.update(
            freeze_base_model=distribution_params["freeze_base_model"],
            base_checkpoint_path=(
                f'results/{distribution_params["base_checkpoint_path"]}_{dataset}/mcp/weights'  # noqa: E501
                if distribution_params["base_checkpoint_path"]
                else False
            ),
            base_distribution=distribution_params["base_distribution"],
            base_distribution_kwds=distribution_params["base_distribution_kwds"],
            base_parameter_kwds=distribution_params["base_parameter_kwds"],
        )
        if not distribution_params["base_checkpoint_path"]:
            fit_kwds.update(
                loss=lambda y, dist: -dist.log_prob(y)
                - tfd.Independent(dist.distribution).log_prob(y)
            )
    else:
        get_model = DensityRegressionModel

    # read name from env
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", args.experiment_name)
    run_name = "_".join((stage, distribution))

    if args.test_mode:
        __LOGGER__.info("Running in test-mode")
        experiment_name += "_test"
        fit_kwds.update(epochs=1)

    # configure mpl to use latex
    if which("latex"):
        __LOGGER__.info("Using latex backend for plotting")
        setup_latex(fontsize=10)

    # don't show progress bar if running from CI
    if os.environ.get("CI", False):
        fit_kwds.update(verbose=2)

    fig_width = get_figsize(params["textwidth"], fraction=0.5)[0]

    # actually execute training
    pipeline(
        experiment_name=experiment_name,
        run_name=run_name,
        results_path=args.results_path,
        log_file=args.log_file,
        seed=params["seed"],
        get_dataset_fn=get_dataset,
        dataset_kwds=dataset_kwds,
        get_model_fn=get_model,
        model_kwds=model_kwds,
        preprocess_dataset=lambda data, model: {
            "x": data[0][0],
            "y": data[0][1],
            "validation_data": data[1],
        },
        fit_kwds=fit_kwds,
        plot_data=partial(plot_data, targets=dataset_kwds["targets"], height=fig_width),
        after_fit_hook=get_after_fit_hook(
            N=2000,
            seed=params["seed"],
            results_path=args.results_path,
            targets=dataset_kwds["targets"],
            height=fig_width,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--log-file", type=str, help="path for log file")
    parser.add_argument(
        "--log-level", type=str, default="INFO", help="logging severaty level"
    )
    parser.add_argument(
        "--test-mode", default=False, type=str2bool, help="activate test-mode"
    )
    parser.add_argument(
        "--experiment-name", type=str, help="MLFlow experiment name", required=True
    )
    parser.add_argument(
        "stage_name",
        type=str,
        help="name of dvstage",
    )
    parser.add_argument(
        "distribution",
        type=str,
        help="name of distribution",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="name of dataset",
    )
    parser.add_argument(
        "results_path",
        type=str,
        help="destination for model checkpoints and logs.",
    )
    args = parser.parse_args()

    main(args)
