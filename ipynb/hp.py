import mlflow
import pandas as pd
from mlflow import tracking
from mlflow.entities import ViewType
import yaml
import json
import numpy as np
import sys  # Import the sys module for stderr
import os

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

# Retrieve all experiments and create a dictionary to map experiment IDs to names
print(f"searching for experiments", file=sys.stderr)
experiment_id_to_name = {}
for exp in mlflow.search_experiments():
    print(f" - found {exp.name}", file=sys.stderr)
    experiment_id_to_name[exp.experiment_id] = exp.name

# Retrieve all experiments
all_experiments = list(experiment_id_to_name.keys())

# Create an empty list to store dictionaries for each result
results_list = []

# Initialize progress counter
progress_counter = 0

print(f"summarizing best run for each experiment", file=sys.stderr)
# Iterate through experiments
for experiment_id in all_experiments:
    runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        run_view_type=ViewType.ALL,
    )

    # Check if there are any valid runs with non-empty DataFrame
    if not runs.empty:
        # Find the index of the run with the best validation loss
        best_run_index = runs["metrics.val_loss"].idxmin()

        # Check if best_run_index is NaN
        if not pd.isna(best_run_index):
            # Convert best_run_index to an integer
            best_run_index = int(best_run_index)

            # Use iloc to access the row with the minimum validation loss
            best_run = runs.iloc[best_run_index]

            # Get the experiment name from the dictionary
            experiment_name = experiment_id_to_name[experiment_id]

            # Get the run name from the "runName" tag
            run_name = best_run["tags.mlflow.runName"]

            # Extract other relevant information
            best_validation_loss = best_run["metrics.val_loss"]

            # Extract parameters from columns starting with "params"
            params = {
                key: value
                for key, value in best_run.items()
                if key.startswith("params.")
            }

            # Create a dictionary for the current result
            result_dict = {
                "Experiment Name": experiment_name,
                "Run Name": run_name,
                "Best Validation Loss": best_validation_loss,
                "Parameters": params,
            }

            # Append the result to the list
            results_list.append(result_dict)

    # Update progress
    progress_counter += 1
    print(f"Processed {progress_counter}/{len(all_experiments)} experiments", file=sys.stderr)

# Create a DataFrame from the list of results
results_df = pd.DataFrame(results_list)

# Print the final results DataFrame
#print(results_df["Run Name"])

# Serialize the list of dictionaries to JSON with the custom function
results_json = json.dumps(results_list, indent=4, default=numpy_to_python)
print(results_json)