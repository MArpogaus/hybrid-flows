import mlflow
import pandas as pd
from mlflow import tracking
from mlflow.entities import ViewType
import yaml
import json
import numpy as np
import sys  # Import the sys module for stderr
import os
import time
import argparse
import re

# Define a custom function to convert NumPy objects to standard Python objects
def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return np.asscalar(obj)
    return obj

# Set the MLflow tracking server URI
#tracking_uri = "http://127.0.0.1:5000"  # Replace with your MLflow server URI
tracking_uri = os.environ["MLFLOW_TRACKING_URI"]  # Replace with your MLflow server URI
print(f"connecting with: {tracking_uri}", file=sys.stderr)
tracking.set_tracking_uri(tracking_uri)

def download_data():
    # Retrieve all experiments and create a dictionary to map experiment IDs to names
    print(f"searching for experiments", file=sys.stderr)
    experiment_id_to_name = {}
    for exp in mlflow.search_experiments():
        if "_test" not in exp.name:
            print(f" - found {exp.name}", file=sys.stderr)
            experiment_id_to_name[exp.experiment_id] = exp.name

    # Retrieve all experiments
    all_experiments = list(experiment_id_to_name.keys())

    ts = time.time()
    print("downloading runs", file=sys.stderr)
    all_runs = []
    progress_counter = 0
    for experiment_id in all_experiments:
        try:
            runs = mlflow.search_runs(
                experiment_ids=[experiment_id],
                run_view_type=ViewType.ALL,
                max_results=1000000
            )
            all_runs.append(runs)
            progress_counter += 1
            print(f"Downloaded {progress_counter}/{len(all_experiments)} experiments", file=sys.stderr)
            time.sleep(1) # to avoid to many requests error (and speed up download time??)
        except Exception as e:
            all_runs.append(None)
            progress_counter += 1
            print(f"skipping {progress_counter} {list(experiment_id_to_name.items())[progress_counter-1]}", file=sys.stderr)
            
    te = time.time()
    print(f"download took {te-ts}")

    df = pd.concat(all_runs)
    df["experiment_name"] = df.apply(lambda row: experiment_id_to_name[row["experiment_id"]], axis=1)

    df = df.dropna(axis=1, how='all')
    df = df.sort_values(["metrics.val_loss"])

    df.to_feather("mlflow_data.feather")

    return df


def generate_combinations(dictionary, keys=None):
    if keys is None:
        keys = list(dictionary.keys())
    
    if not keys:
        return []
    
    current_key = keys[0]
    remaining_keys = keys[1:]
    
    if remaining_keys:
        combinations = []
        for value in dictionary[current_key]:
            sub_combinations = generate_combinations(dictionary, remaining_keys)
            for sub_combo in sub_combinations:
                combinations.append([(current_key, value)] + sub_combo)
        return combinations
    else:
        return [[(current_key, value)] for value in dictionary[current_key]]


def query_data(exp_name: str, run_name: str,query: dict):
    
    # Generate all combinations
    combinations = generate_combinations(query)
    df = pd.read_feather("mlflow_data.feather")
    combo_exist = []
    for combo in combinations:
        try:
            print(f"evaluating {(exp_name, run_name)} {combo}")
            q = [df["params."+c[0]] == c[1] for c in combo]
            # q1 = df["params."+"fit_kwds.learning_rate"] == "0.01"
            # q2 = df["experiment_name"] == "unconditional-circles"
            # q3 = df["tags.mlflow.runName"] == "bernstein_flow"
            q.append(df["experiment_name"] == exp_name+"doesnotexist")
            q.append(df["tags.mlflow.runName"] == run_name)
            res = df[np.logical_and.reduce(q)]
            num_solutions = res.shape[0]
            print(num_solutions)
            if num_solutions == 0:
                combo_exist.append(False)
            else:
                combo_exist.append(True)
        except:
            combo_exist.append(False)

    print(list(zip(combo_exist, combinations)))
    #TODO: zip not working
    for c in list(zip([combo_exist, combinations[0]])):
        print("res",c)
    return []
    uncached_combinations = [print(c[1]) for c in list(zip([combo_exist, combinations])) if c[0] == False]
    return uncached_combinations



def split_string_with_lists(input_string):
    # Define a regular expression pattern to find Python lists
    pattern = r'\[[^\[\]]*\]'

    # Find all Python lists in the input string
    lists = re.findall(pattern, input_string)

    # Replace the Python lists with a placeholder
    placeholder = '###LIST###'
    string_without_lists = re.sub(pattern, placeholder, input_string)

    # Split the string by commas
    result = string_without_lists.split(',')

    # Replace the placeholders back with the original lists
    for i, part in enumerate(result):
        if placeholder in part:
            result[i] = lists.pop(0)

    # Remove leading and trailing whitespace from each part
    result = [part.strip() for part in result]

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("TASK", choices=["download","find"])
    parser.add_argument('--exp_name', '-e',nargs=1)
    parser.add_argument('--run_name', '-r',nargs=1)
    parser.add_argument('-S', action='append', nargs='+')
    args = parser.parse_args()

    task = args.TASK
    if task == "find":
        S = args.S if args.S else []
        S = [s[0] for s in S] # we only consider first positional element
        S = [ ".".join(s.split(".")[3:]) for s in S]
        S = dict(s.split("=") for s in S)
        for key, value in S.items():
            S[key] = split_string_with_lists(value)
        exp_name, run_name = args.exp_name[0], args.run_name[0]
        exp_name = exp_name.replace("_distributions", "")
        run_name = exp_name.split("-")[0]+"_"+run_name
        exp_name = exp_name.replace("_","-")
        print(exp_name, run_name, S)
        uncached = query_data(exp_name, run_name, S)
        print(uncached, file=sys.stderr)
    elif task == "download":
        download_data()

