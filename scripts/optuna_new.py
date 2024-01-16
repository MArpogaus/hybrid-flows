# %% imports
import argparse
import importlib
import logging
import os
import pickle
from copy import deepcopy

import mlflow
import numpy as np
import optuna
from optuna.samplers import CmaEsSampler

from mctm.utils import str2bool
from mctm.utils.pipeline import (
    log_cfg,
    prepare_pipeline,
    set_seed,
    start_run_with_exception_logging,
)

__LOGGER__ = logging.getLogger(__name__)


# %% function definitions
def suggest_new_params(
    trial, inital_params, parameter_space_definition, stage_name, distribution, dataset
):
    params = deepcopy(inital_params)
    stage = stage_name.split("@")[0]
    model_kwds = params[stage + "_distributions"][distribution][dataset]

    for d in parameter_space_definition:
        p = model_kwds
        keys = d["name"].split(".")
        for k in keys[:-1]:
            p = p[k]

        key = keys[-1]
        if key.isdigit():
            key = int(key)
        p[key] = getattr(trial, f'suggest_{d["type"]}')(d["name"], **d["kwargs"])

    return params


def get_objective(
    model_train_script_name,
    experiment_name,
    results_path,
    log_file,
    log_level,
    dataset,
    stage_name,
    distribution,
    initial_params,
    parameter_space_definition,
    test_mode,
):
    run = importlib.import_module(model_train_script_name).run

    def objective(trial):
        params = suggest_new_params(
            trial,
            initial_params,
            parameter_space_definition,
            stage_name,
            distribution,
            dataset,
        )

        # prepare results directory
        trial_results_path = os.path.join(results_path, str(trial.number))
        os.makedirs(trial_results_path, exist_ok=True)

        history, _, _ = run(
            experiment_name=experiment_name,
            results_path=trial_results_path,
            log_file=log_file,
            log_level=log_level,
            dataset=dataset,
            stage_name=stage_name,
            distribution=distribution,
            params=params,
            test_mode=test_mode,
        )

        min_idx = np.argmin(history.history["val_loss"])
        val_loss = history.history["val_loss"][min_idx]

        return val_loss

    return objective


def run_study(
    model_train_script_name,
    experiment_name,
    results_path,
    log_file,
    log_level,
    dataset,
    stage_name,
    distribution,
    initial_params,
    parameter_space_definition,
    test_mode,
    n_trials,
    n_jobs,
    study_name,
    load_study_from_storage,
):
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)
    run_name = "_".join(("optuna", stage_name, distribution, dataset))

    if test_mode:
        __LOGGER__.info("Running in test-mode")
        run_name += "_test"
        n_trials = 4

    mlflow.set_experiment(experiment_name)
    __LOGGER__.info("Logging to MLFlow Experiment: %s", experiment_name)

    with start_run_with_exception_logging(run_name=run_name):
        log_cfg({"parameter_space": parameter_space_definition})
        mlflow.log_dict(
            parameter_space_definition,
            "parameter_space_definition.yaml",
        )

        # Seed?
        set_seed(1)

        study_kwds = dict(study_name=study_name, sampler=CmaEsSampler())
        if load_study_from_storage:
            study = optuna.load_study(storage=load_study_from_storage, **study_kwds)
        else:
            study = optuna.create_study(directions=["minimize"], **study_kwds)

        objective = get_objective(
            model_train_script_name=model_train_script_name,
            experiment_name=experiment_name,
            results_path=results_path,
            log_file=log_file,
            log_level=log_level,
            dataset=dataset,
            stage_name=stage_name,
            distribution=distribution,
            initial_params=deepcopy(inital_params),
            parameter_space_definition=deepcopy(parameter_space_definition),
            test_mode=test_mode,
        )

        study.optimize(
            objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True
        )
        optimum = study.best_params

        __LOGGER__.info(
            f"finished hyperparameter optimization with optimum:\n{optimum}"
        )
        mlflow.log_dict(optimum, "optimal_parameters.json")
        mlflow.log_metric("min_val_loss", study.best_value)
        mlflow.log_metric("best_trial", study.best_trial.number)
        with open("study.pickle", "wb") as handle:
            pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact("study.pickle")

        if optuna.visualization.is_available():
            fig = optuna.visualization.plot_optimization_history(study)
            mlflow.log_figure(fig, "optim_history.html")

        return study.best_params


# %% if main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Hyperparameters of a model")
    parser.add_argument(
        "--log-file",
        type=str,
        help="path for log file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="logging severaty level",
    )
    parser.add_argument(
        "--test-mode",
        default=False,
        type=str2bool,
        help="activate test-mode",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLFlow experiment name",
        required=True,
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="number of trials to run.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="number of threads to run.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="name of the study.",
    )
    parser.add_argument(
        "--load-study-from-storage",
        default=False,
        type=str,
        help="Load and continue existing study fro provided storage",
    )
    parser.add_argument(
        "stage_name",
        type=str,
        help="name of dvc stage",
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

    # load initial params
    inital_params = prepare_pipeline(args)

    def def_param(name, type, **kwargs):
        return {"name": name, "type": type, "kwargs": kwargs}

    parameter_space_definition = [
        def_param(
            "fit_kwds.batch_size",
            "int",
            low=32,
            high=1024,
        ),
        def_param(
            "fit_kwds.learning_rate",
            "float",
            low=0.001,
            high=0.1,
        ),
        def_param(
            "fit_kwds.lr_patience",
            "int",
            low=5,
            high=100,
        ),
        def_param(
            "fit_kwds.early_stopping",
            "int",
            low=0,
            high=4,
        ),
        def_param(
            "distribution_kwds.order",
            "int",
            low=10,
            high=100,
        ),
        def_param(
            "parameter_kwds.activation",
            "categorical",
            choices=["relu", "sigmoid", "tanh"],
        ),
        def_param(
            "parameter_kwds.hidden_units.0",
            "int",
            low=4,
            high=1024,
        ),
    ]

    if "benchmark" in args.stage_name:
        model_train_script_name = "train_benchmark"
    elif "malnutrition" in args.stage_name:
        model_train_script_name = "train_malnutrition"
    else:
        model_train_script_name = "train"

    res = run_study(
        model_train_script_name=model_train_script_name,
        experiment_name=args.experiment_name,
        results_path=args.results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        dataset=args.dataset,
        stage_name=args.stage_name,
        distribution=args.distribution,
        initial_params=inital_params,
        parameter_space_definition=parameter_space_definition,
        test_mode=args.test_mode,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        load_study_from_storage=args.load_study_from_storage,
    )
