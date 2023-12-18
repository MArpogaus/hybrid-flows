# requires cmaes package to be present,
# not automatically installed with conda install optuna!

import json
import pickle
import subprocess

import optuna
import yaml
from optuna.samplers import CmaEsSampler


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
    overrides = [(o[0]) + "=" + str(o[1]) for o in overrides.items()]
    overrides = [f'"{o}"' for o in overrides]
    overrides = " -S ".join(overrides)

    cmd = f"dvc exp run {model}@{distribution}-{dataset} -S {overrides}"
    print(f"executing {cmd}")
    run_cmd(cmd)
    with open(f"results/{model}_{distribution}_{dataset}/metrics.yaml", "r") as file:
        metrics = yaml.safe_load(file)
        return float(metrics["val_loss"])


study = optuna.create_study(sampler=CmaEsSampler(), directions=["minimize"])


def objective(trial):
    key_batch_size = (
        f"{model}_distributions.{distribution}.{dataset}.fit_kwds.batch_size"
    )
    key_learning_rate = (
        f"{model}_distributions.{distribution}.{dataset}.fit_kwds.learning_rate"
    )
    key_patience = (
        f"{model}_distributions.{distribution}.{dataset}.fit_kwds.lr_patience"
    )

    batch_size = trial.suggest_int(key_batch_size, 32, 512)
    learning_rate = trial.suggest_float(key_learning_rate, 0.001, 0.01)
    patience = trial.suggest_int(key_patience, 5, 10)
    # n_hidden_layers = 3

    overrides = {
        key_batch_size: batch_size,
        key_learning_rate: learning_rate,
        key_patience: patience,
    }
    val_loss = run_exp_with_overrides(overrides)
    return val_loss


study.optimize(objective, n_trials=100)

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
