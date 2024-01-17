# %% imports
import argparse
import importlib
import logging
import os
import pickle
from copy import deepcopy

import mlflow
import numpy as np
import yaml
from optuna.samplers import CmaEsSampler

import optuna
from mctm.utils import str2bool
from mctm.utils.pipeline import (
    prepare_pipeline,
    set_seed,
    start_run_with_exception_logging,
)

# %% global objects
__LOGGER__ = logging.getLogger(__name__)
__RUN_ID_ATTRIBUTE_KEY__ = "mlflow_run_id"
__BEST_VALUE_ATTRIBUTE_KEY__ = "best_value"


# %% function definitions
def suggest_new_params(
    trial,
    inital_params,
    parameter_space_definition,
    stage_name,
    distribution,
    dataset,
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
    stage,
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
            stage,
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
            stage_name=stage,
            distribution=distribution,
            params=params,
            test_mode=test_mode,
        )

        min_idx = np.argmin(history.history["val_loss"])
        val_loss = history.history["val_loss"][min_idx]

        return val_loss

    return objective


def best_value_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon
    existing best trial values.
    """

    winner = study.user_attrs.get(__BEST_VALUE_ATTRIBUTE_KEY__, np.inf)

    if study.best_value and winner < study.best_value:
        study.set_user_attr(__BEST_VALUE_ATTRIBUTE_KEY__, study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            __LOGGER__.info(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} "
                f"with {improvement_percent: .4f}% improvement"
            )
        else:
            __LOGGER__.info(
                f"Initial trial {frozen_trial.number} "
                f"achieved value: {frozen_trial.value}"
            )


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
    seed,
):
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)

    stage = stage_name.split("@")[0]
    run_name = "_".join(("optuna", study_name))
    print(run_name)
    if test_mode:
        __LOGGER__.info("Running in test-mode")
        run_name += "_test"
        n_trials = 2

    mlflow.set_experiment(experiment_name)
    __LOGGER__.info("Logging to MLFlow Experiment: %s", experiment_name)

    # Seed?
    set_seed(seed)
    study_kwds = dict(study_name=study_name, sampler=CmaEsSampler(seed=seed))
    if load_study_from_storage:
        study = optuna.load_study(storage=load_study_from_storage, **study_kwds)
        # If a study has been started before, a parent may already exists run
        run_id = study.user_attrs.get(__RUN_ID_ATTRIBUTE_KEY__, None)
    else:
        study = optuna.create_study(directions=["minimize"], **study_kwds)
        run_id = None

    with start_run_with_exception_logging(run_name=run_name, run_id=run_id) as run:
        if load_study_from_storage and run_id is None:
            # Store the run id to log subsequent runs to the same parent
            study.set_user_attr(__RUN_ID_ATTRIBUTE_KEY__, run.info.run_id)
        mlflow.log_dict(
            parameter_space_definition,
            "parameter_space_definition.yaml",
        )

        objective = get_objective(
            model_train_script_name=model_train_script_name,
            experiment_name=experiment_name,
            results_path=results_path,
            log_file=log_file,
            log_level=log_level,
            dataset=dataset,
            stage=stage,
            distribution=distribution,
            initial_params=deepcopy(inital_params),
            parameter_space_definition=deepcopy(parameter_space_definition),
            test_mode=test_mode,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            callbacks=[best_value_callback],
        )
        best_params = study.best_params

        winner = study.user_attrs.get(__BEST_VALUE_ATTRIBUTE_KEY__, np.inf)

        __LOGGER__.info(
            f"finished hyperparameter optimization with optimum:\n{best_params}"
        )

        if study.best_value < winner:
            __LOGGER__.info(
                f"This run achieved an new optimum ({study.best_value=}) "
                f"in trial {study.best_trial.number}. "
                "Logging results with MLFlow."
            )

            mlflow.log_dict(best_params, "optimal_parameters.yaml")
            mlflow.log_metric("min_val_loss", study.best_value)
            mlflow.log_metric("best_trial", study.best_trial.number)
            if not load_study_from_storage:
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
        "--experiment-name",
        type=str,
        help="MLFlow experiment name",
        required=True,
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        help="name of dvc stage",
        required=True,
    )
    parser.add_argument(
        "--distribution",
        type=str,
        help="name of distribution",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="name of dataset",
        required=True,
    )
    parser.add_argument(
        "--results_path",
        type=str,
        help="destination for model checkpoints and logs.",
        required=True,
    )
    parser.add_argument(
        "--parameter_space_definition_file",
        type=argparse.FileType("r"),
        help="destination for model checkpoints and logs.",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for sampler's random number generator",
        required=True,
    )
    args = parser.parse_args()

    # load initial params
    inital_params = prepare_pipeline(args)
    with args.parameter_space_definition_file as f:
        parameter_space_definition = yaml.safe_load(f)

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
        seed=args.seed,
    )
