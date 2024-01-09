# requires cmaes package to be present,
# not automatically installed with conda install optuna!

import logging
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import mlflow
import numpy as np
import optuna
from optuna.samplers import CmaEsSampler
from train_benchmark import main

from mctm.utils.pipeline import prepare_pipeline


def nested_dictionary():
    return defaultdict(nested_dictionary)


def update_nested_dict(original, update):
    for key, value in update.items():
        if (
            isinstance(value, dict)
            and key in original
            and isinstance(original[key], dict)
        ):
            update_nested_dict(original[key], value)
        else:
            original[key] = value


# def run_cmd(command: str):
#     process = subprocess.Popen(command, shell=True)
#     process.wait()

# run_exp = f"dvc exp run {model}@{distribution}-{dataset} -S testmode=true"#
# read_results = f"cat results/{model}_{distribution}_{dataset}/metrics.yaml"
# print(run_exp)
# run_cmd(run_exp)
# run_cmd(read_results)


def build_args(
    dataset="miniboone",
    distribution="masked_autoregressive_flow",
    model="unconditional_benchmark",
):
    log_level = "info"
    log_file = f"results/{model}_{distribution}_{dataset}/train.log"
    experiment_name = f"{model}_{dataset}"
    stage_name = f"{model}@{distribution}-{dataset}"
    results_path = f"results/{model}_{distribution}_{dataset}"

    args = SimpleNamespace(
        log_file=log_file,
        log_level=log_level,
        test_mode=True,
        experiment_name=experiment_name,
        stage_name=stage_name,
        distribution=distribution,
        dataset=dataset,
        results_path=results_path,
    )
    return args


def args_inverse(args):
    dataset = args.dataset
    distribution = args.distribution
    model = args.stage_name.split("@")[0]
    return dataset, distribution, model


args = build_args()
params = prepare_pipeline(args)


def run_exp_with_overrides(overrides: dict):
    try:
        exp_params = deepcopy(params)
        update_nested_dict(exp_params, overrides)
        logging.info(f"{exp_params=}")
        history, _, _ = main(
            args.experiment_name,
            args.results_path,
            args.log_file,
            args.log_level,
            args.dataset,
            args.stage_name,
            args.distribution,
            exp_params,
            args.test_mode,
        )

        min_idx = np.argmin(history.history["val_loss"])
        val_loss = history.history["val_loss"][min_idx]
    except:
        val_loss = 1234567890
    return val_loss

    # cmd = f"dvc exp run {model}@{distribution}-{dataset} -S {overrides}"
    # print(f"executing {cmd}")
    # run_cmd(cmd)
    # with open(f"results/{model}_{distribution}_{dataset}/metrics.yaml", "r") as file:
    #    metrics = yaml.safe_load(file)
    #    return float(metrics["val_loss"])


study = optuna.create_study(sampler=CmaEsSampler(), directions=["minimize"])


def objective(trial):
    # SEED

    dataset, distribution, model = args_inverse(args)

    batch_size = trial.suggest_int("batch_size", 32, 512)
    learning_rate = trial.suggest_float("key_learning_rate", 0.001, 0.01)
    patience = trial.suggest_int("key_patience", 5, 10)
    layers = [trial.suggest_int(f"hidden_units_{i}", 64, 1028) for i in [1, 2, 3]]

    overrides = nested_dictionary()
    # TODO: this is equal to exp name so replace model_distributions
    overrides[f"{model}_distributions"][distribution][dataset]["fit_kwds"][
        "batch_size"
    ] = batch_size
    overrides[f"{model}_distributions"][distribution][dataset]["fit_kwds"][
        "learning_rate"
    ] = learning_rate
    overrides[f"{model}_distributions"][distribution][dataset]["fit_kwds"][
        "lr_patience"
    ] = patience
    overrides[f"{model}_distributions"][distribution][dataset]["parameter_kwds"][
        "hidden_units"
    ] = layers
    val_loss = run_exp_with_overrides(overrides)
    return val_loss


mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment(
    f"optuna-optim-{args_inverse(args)[2]}_{args_inverse(args)[1]}_{args_inverse(args)[0]}"
)
with mlflow.start_run(nested=True) as run:
    import time

    start_time = time.time()
    study.optimize(objective, n_trials=1000, n_jobs=3)  # to many jobs cause freezing
    end_time = time.time()

    optimum = study.best_params

    print(
        (
            f"finished hyperparameter optimization with optimum:\n{optimum}"
            "\n writing result to _data.json_."
        )
    )
    mlflow.log_dict(optimum, "optimal_parameters.json")

    with open("study.pickle", "wb") as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
    mlflow.log_artifact("study.pickle")

    if optuna.visualization.is_available():
        fig = optuna.visualization.plot_optimization_history(study)
        mlflow.log_figure(fig, "optim_history.html")
    # mlflowlog -> bernsteinflow/cml/hp_

    # with open("data.json", "w") as f:
    #     json.dump(optimum, f)

    print("\n\n")
    print(f"best trial: {study.best_trial}")
    print(f"result value: {study.best_value}")
    print(f"optimizer took {end_time - start_time} seconds")
