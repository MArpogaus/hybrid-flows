# requires cmaes package to be present,
# not automatically installed with conda install optuna!

import json
import pickle
import subprocess

import optuna
import yaml
from optuna.samplers import CmaEsSampler
from train_benchmark import main
from types import SimpleNamespace
from collections import defaultdict

def nested_dictionary():
    return defaultdict(nested_dictionary)

def run_cmd(command: str):
    process = subprocess.Popen(command, shell=True)
    process.wait()


dataset = "miniboone"
distribution = "elementwise_flow"
model = "unconditional_benchmark"

run_exp = f"dvc exp run {model}@{distribution}-{dataset} -S testmode=true"
read_results = f"cat results/{model}_{distribution}_{dataset}/metrics.yaml"
# print(run_exp)
# run_cmd(run_exp)
# run_cmd(read_results)


def run_exp_with_overrides(overrides: dict):
    log_level = "info"
    log_file = f"results/{model}_{distribution}_{dataset}/train.log"
    experiment_name = f"{model}_{dataset}"
    stage_name = f"{model}@{distribution}-{dataset}"
    results_path = f"results/{model}_{distribution}_{dataset}"

    args = SimpleNamespace(log_file=log_file, log_level=log_level, test_mode=True, experiment_name=experiment_name, stage_name=stage_name, distribution=distribution, dataset=dataset, results_path=results_path)

    history, _, _ = main(args, overrides)

    val_loss = history.history["val_loss"]

    return val_loss

    #cmd = f"dvc exp run {model}@{distribution}-{dataset} -S {overrides}"
    #print(f"executing {cmd}")
    #run_cmd(cmd)
    #with open(f"results/{model}_{distribution}_{dataset}/metrics.yaml", "r") as file:
    #    metrics = yaml.safe_load(file)
    #    return float(metrics["val_loss"])


study = optuna.create_study(sampler=CmaEsSampler(), directions=["minimize"])


def objective(trial):
    #SEED
    
    batch_size = trial.suggest_int("batch_size", 32, 512)
    #learning_rate = trial.suggest_float(key_learning_rate, 0.001, 0.01)
    #patience = trial.suggest_int(key_patience, 5, 10)
    # n_hidden_layers = 3

    overrides = nested_dictionary()
    overrides[f"{model}_distributions"][distribution][dataset]["fit_kwds"]["batch_size"] = batch_size
    val_loss = run_exp_with_overrides(overrides)
    return val_loss

import time
start_time = time.time()
study.optimize(objective, n_trials=3, n_jobs=1)
end_time = time.time()

optimum = study.best_params

print(
    (
        f"finished hyperparameter optimization with optimum:\n{optimum}"
        "\n writing result to _data.json_."
    )
)


with open("data.json", "w") as f:
    json.dump(optimum, f)


with open("study.pickle", "wb") as handle:
    pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(study.best_trial)
print(study.best_value)
print(end_time - start_time)
