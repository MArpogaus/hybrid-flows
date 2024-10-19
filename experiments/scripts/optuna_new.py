"""Script to run a Optuna study for hyperparameter optimization."""

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
import yaml
from optuna.integration import TFKerasPruningCallback
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from mctm.utils import str2bool
from mctm.utils.mlflow import log_cfg
from mctm.utils.pipeline import (
    prepare_pipeline,
    start_run_with_exception_logging,
)

# %% global objects
__LOGGER__ = logging.getLogger(__name__)
__RUN_ID_ATTRIBUTE_KEY__ = "mlflow_run_id"
__BEST_VALUE_ATTRIBUTE_KEY__ = "best_value"
__EVALUATION_METRIC__ = "val_loss"


# %% function definitions
def suggest_new_params(
    trial: optuna.trial.Trial,
    initial_params: dict,
    parameter_space_definition: list,
    use_pruning: bool,
) -> dict:
    """Suggest new hyperparameters for Optuna trial.

    Parameters
    ----------
    trial : optuna.trial.Trial
        The trial to suggest new hyperparameters.
    initial_params : dict
        The initial set of parameters.
    parameter_space_definition : list
        The definition of the parameter space.
    use_pruning : bool
        Whether to use pruning.

    Returns
    -------
    dict
        The suggested parameters.

    """
    __LOGGER__.debug(f"{trial}: {use_pruning=}")
    params: dict = deepcopy(initial_params)

    for d in parameter_space_definition:
        p: dict = params
        keys: list = d["name"].split(".")
        for k in keys[:-1]:
            if k not in p.keys():
                __LOGGER__.warning(
                    "key '%s' not defined in initial parameters. Typo?", {d["name"]}
                )
                p[k] = {}
            p = p[k]

        key = int(keys[-1]) if keys[-1].isdigit() else keys[-1]
        __LOGGER__.debug(f"trial keyword ({trial.number}): {d['name']}, {d['type']}")

        if d["type"] == "choose_from_list":
            options = d["list"]
            v = options[trial.suggest_int(d["name"], low=0, high=len(options) - 1)]
        else:
            v = getattr(trial, f'suggest_{d["type"]}')(d["name"], **d["kwargs"])
        p[key] = v

    if use_pruning:
        params["fit_kwargs"]["callbacks"] = [
            TFKerasPruningCallback(trial, __EVALUATION_METRIC__)
        ]

    __LOGGER__.info("suggesting new params: %s", params)

    return params


def get_objective(
    model_train_script_name: str,
    dataset_name: str,
    dataset_type: str,
    experiment_name: str,
    run_name: str,
    results_path: str,
    log_file: str,
    log_level: str,
    initial_params: dict,
    parameter_space_definition: list,
    test_mode: bool,
    use_pruning: bool,
) -> callable:
    """Get the objective function for Optuna study.

    Parameters
    ----------
    model_train_script_name : str
        The name of the training script.
    dataset_name : str
        The name of the dataset.
    dataset_type : str
        The type of the dataset.
    experiment_name : str
        The name of the MLflow experiment.
    run_name : str
        The name of the run.
    results_path : str
        The path to store results.
    log_file : str
        The path for the log file.
    log_level : str
        The logging severity level.
    initial_params : dict
        The initial set of parameters.
    parameter_space_definition : list
        The definition of the parameter space.
    test_mode : bool
        Whether to run in test mode.
    use_pruning : bool
        Whether to use pruning.

    Returns
    -------
    callable
        The objective function.

    """
    run = importlib.import_module(model_train_script_name).run

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_new_params(
            trial,
            initial_params,
            parameter_space_definition,
            use_pruning,
        )

        trial_results_path = os.path.join(results_path, str(trial.number))
        os.makedirs(trial_results_path, exist_ok=True)
        if experiment_name:
            mlflow.set_experiment(experiment_name)
            __LOGGER__.info("Logging to MLFlow Experiment: %s", experiment_name)
        with start_run_with_exception_logging(run_name=run_name):
            # Auto log all MLflow entities

            mlflow.autolog(log_models=False)
            mlflow.log_dict(params, "params.yaml")
            log_cfg(params)

            history, _, _ = run(
                dataset_name=dataset_name,
                dataset_type=dataset_type,
                experiment_name=experiment_name,
                run_name=run_name,
                log_file=log_file,
                log_level=log_level,
                results_path=trial_results_path,
                test_mode=test_mode,
                params=params,
            )

        min_idx = np.argmin(history.history[__EVALUATION_METRIC__])
        val_loss = history.history[__EVALUATION_METRIC__][min_idx]

        return val_loss

    return objective


def best_value_callback(
    study: optuna.study.Study, frozen_trial: optuna.trial.FrozenTrial
) -> None:
    """Report when a new trial improves upon existing best trial values.

    Parameters
    ----------
    study : optuna.study.Study
        The study to report the trial to.
    frozen_trial : optuna.trial.FrozenTrial
        The trial that has finished.

    """
    winner = study.user_attrs.get(__BEST_VALUE_ATTRIBUTE_KEY__, np.inf)
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    if len(complete_trials) >= 1 and winner > study.best_value:
        study.set_user_attr(__BEST_VALUE_ATTRIBUTE_KEY__, study.best_value)
        mlflow.log_metric("best_value", min(winner, study.best_value))
        mlflow.log_dict(study.best_params, "optimal_parameters.yaml")

        if winner:
            relative_improvement = abs(winner - study.best_value) / study.best_value
            __LOGGER__.info(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} "
                f"with {relative_improvement * 100: .4f}% improvement"
            )
            mlflow.log_metric("relative_improvement", relative_improvement)
        else:
            __LOGGER__.info(
                f"Initial trial {frozen_trial.number} "
                f"achieved value: {frozen_trial.value}"
            )

    if optuna.visualization.is_available() and len(complete_trials) > 1:
        for plot_fn_name in (
            "plot_contour",
            "plot_optimization_history",
            "plot_parallel_coordinate",
            "plot_param_importances",
            "plot_rank",
            "plot_slice",
            "plot_timeline",
        ):
            fig = getattr(optuna.visualization, plot_fn_name)(study)
            mlflow.log_figure(fig, f"{plot_fn_name}.html")


def reprot_pruned_trials(study: optuna.study.Study, _) -> None:
    """Report the number of pruned and complete trials.

    Parameters
    ----------
    study : optuna.study.Study
        The study to report the trial to.
    _ : Any
        Unused argument.

    """
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    __LOGGER__.info("Study statistics: ")
    __LOGGER__.info(f"Number of finished trials: {len(study.trials)}")
    __LOGGER__.info(f"Number of pruned trials: {len(pruned_trials)}")
    __LOGGER__.info(f"Number of complete trials: {len(complete_trials)}")

    mlflow.log_metric("pruned_trials", len(pruned_trials))
    mlflow.log_metric("complete_trials", len(complete_trials))


def run_study(
    model_train_script_name: str,
    experiment_name: str,
    results_path: str,
    log_file: str,
    log_level: str,
    dataset_name: str,
    dataset_type: str,
    initial_params: dict,
    parameter_space_definition: list,
    test_mode: bool,
    n_trials: int,
    n_jobs: int,
    study_name: str,
    load_study_from_storage: str,
    seed: int,
    use_pruning: bool,
) -> dict:
    """Run the Optuna study for hyperparameter optimization.

    Parameters
    ----------
    model_train_script_name : str
        The name of the training script.
    experiment_name : str
        The name of the MLflow experiment.
    results_path : str
        The path to store results.
    log_file : str
        The path for the log file.
    log_level : str
        The logging severity level.
    dataset_name : str
        The name of the dataset.
    dataset_type : str
        The type of the dataset.
    initial_params : dict
        The initial set of parameters.
    parameter_space_definition : list
        The definition of the parameter space.
    test_mode : bool
        Whether to run in test mode.
    n_trials : int
        The number of trials to run.
    n_jobs : int
        The number of threads to run.
    study_name : str
        The name of the study.
    load_study_from_storage : str
        The storage to load the study from.
    seed : int
        The seed for the sampler's random number generator.
    use_pruning : bool
        Whether to use pruning.

    Returns
    -------
    dict
        The best parameters found by the study.

    """
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", experiment_name)

    run_name = "_".join(("optuna", study_name))
    if test_mode:
        __LOGGER__.info("Running in test-mode")
        run_name += "_test"
        n_trials = 2

    mlflow.set_experiment(experiment_name)
    __LOGGER__.info("Logging to MLFlow Experiment: %s", experiment_name)

    study_kwargs = dict(study_name=study_name, sampler=TPESampler(seed=seed))

    if use_pruning:
        study_kwargs["pruner"] = SuccessiveHalvingPruner(
            min_resource="auto",
            reduction_factor=4,
            min_early_stopping_rate=0,
            bootstrap_count=0,
        )

    if load_study_from_storage:
        __LOGGER__.warning(
            f"loading study from storage {load_study_from_storage}."
            " This can confuse parameter definitions."
        )
        study = optuna.load_study(storage=load_study_from_storage, **study_kwargs)
        run_id = study.user_attrs.get(__RUN_ID_ATTRIBUTE_KEY__, None)
    else:
        study = optuna.create_study(directions=["minimize"], **study_kwargs)
        run_id = None

    with start_run_with_exception_logging(run_name=run_name, run_id=run_id) as run:
        if load_study_from_storage and run_id is None:
            study.set_user_attr(__RUN_ID_ATTRIBUTE_KEY__, run.info.run_id)
        mlflow.log_dict(
            parameter_space_definition,
            "parameter_space_definition.yaml",
        )
        sub_run_name = "_".join(
            (
                run_name,
                initial_params["model_kwargs"]["distribution"],
                dataset_name,
            )
        )
        objective = get_objective(
            model_train_script_name=model_train_script_name,
            experiment_name=experiment_name,
            run_name=sub_run_name,
            results_path=results_path,
            log_file=log_file,
            log_level=log_level,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            initial_params=deepcopy(initial_params),
            parameter_space_definition=deepcopy(parameter_space_definition),
            test_mode=test_mode,
            use_pruning=use_pruning,
        )

        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,
            callbacks=[best_value_callback, reprot_pruned_trials],
        )

        __LOGGER__.info(
            f"Best value achieved ({study.best_value=}) "
            f"in trial {study.best_trial.number}, "
            f"with parameters: {study.best_params}"
        )

        if not load_study_from_storage:
            with open("study.pickle", "wb") as handle:
                pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
            mlflow.log_artifact("study.pickle")

        return study.best_params


# %% if main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Hyperparameters of a model")
    parser.add_argument(
        "--log-file",
        type=str,
        help="path for log file",
        default=None,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        help="logging severity level",
        default="INFO",
    )
    parser.add_argument(
        "--test-mode",
        type=str2bool,
        help="activate test-mode",
        default=False,
    )
    parser.add_argument(
        "--load-study-from-storage",
        type=str,
        help="Load and continue existing study from provided storage",
        default=False,
    )
    parser.add_argument(
        "--use-pruning",
        default=False,
        type=str2bool,
        help="Prune the trials using Asynchronous Successive Halving Algorithm",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        help="number of threads to run.",
        default=1,
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="number of trials to run.",
        required=True,
    )
    parser.add_argument(
        "--study-name",
        type=str,
        help="name of the study.",
        required=True,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="MLFlow experiment name",
        required=True,
    )
    parser.add_argument(
        "--parameter_file_path",
        type=argparse.FileType("r"),
        help="path to parameter file",
        required=True,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of dataset",
        required=True,
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        help="type of dataset",
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
        help="Parameter space definition file.",
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for sampler's random number generator",
        required=True,
    )
    args = parser.parse_args()

    initial_params = prepare_pipeline(
        results_path=args.results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        stage_name_or_params_file_path=args.parameter_file_path,
    )

    with open(os.path.join("params", args.dataset_type, "dataset.yaml")) as f:
        dataset_kwargs = yaml.safe_load(f)["dataset_kwargs"]
    initial_params["dataset_kwargs"] = dataset_kwargs
    initial_params["textwidth"] = "thesis"
    initial_params["seed"] = args.seed

    with args.parameter_space_definition_file as f:
        parameter_space_definition = yaml.safe_load(f)

    model_train_script_name = (
        "train_malnutrition" if args.dataset_type == "malnutrition" else "train"
    )

    res = run_study(
        model_train_script_name=model_train_script_name,
        experiment_name=args.experiment_name,
        results_path=args.results_path,
        log_file=args.log_file,
        log_level=args.log_level,
        dataset_name=args.dataset_name,
        dataset_type=args.dataset_type,
        initial_params=initial_params,
        parameter_space_definition=parameter_space_definition,
        test_mode=args.test_mode,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        load_study_from_storage=args.load_study_from_storage,
        seed=args.seed,
        use_pruning=args.use_pruning,
    )
