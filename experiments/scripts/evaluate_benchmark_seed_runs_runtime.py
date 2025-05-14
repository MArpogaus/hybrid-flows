# %% imports
import mlflow
import pandas as pd

# %% globals
exp_name = "benchmark-seeds-2025-05-13"
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(
    experiment_names=[exp_name]
    # filter_string="attributes.run_name like '%evaluation'",
)
eval_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("evaluation")]
train_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("train")]

# %% check if we have 20 runs per model
train_runs.groupby("tags.mlflow.runName")["params.seed"].count()
eval_runs.groupby("tags.mlflow.runName")["params.seed"].count()

assert (train_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()
assert (eval_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()


# %% functions
def get_model_name(x):
    if "hybrid" in x:
        if "joint" in x:
            return "HMAF J"
        if "marginals" in x:
            return "HMAF M"
        else:
            return "HMAF"
    else:
        return "MAF"


# %% analyze eval runtime
relevant_columns = [
    "params.seed",
    "params.dataset_name",
    "start_time",
    "end_time",
]
df = eval_runs[relevant_columns]
df.columns = df.columns.map(lambda x: x.split(".")[-1].replace("_", " "))


df = df.assign(model=eval_runs["tags.mlflow.runName"].apply(get_model_name))

eval_run_time_table = (
    df.assign(
        run_time=df[["start time", "end time"]].agg(
            lambda x: (x.iloc[1] - x.iloc[0]).total_seconds() / 60, 1
        )
    )
    .groupby(["model", "dataset name"])
    .run_time.agg(["mean", "std"])
).aggregate(lambda x: f"{x.iloc[0]:.3f} $\pm$ {2 * x.iloc[1]:.3f}", 1)
print(eval_run_time_table)

# %% analyze train runtime
relevant_columns = [
    "params.seed",
    "params.dataset_kwargs.dataset_name",
    "start_time",
    "end_time",
]
df = train_runs[relevant_columns]
df.columns = df.columns.map(lambda x: x.split(".")[-1].replace("_", " "))


df = df.assign(model=train_runs["tags.mlflow.runName"].apply(get_model_name))

train_run_time_table = (
    df.assign(
        run_time=df[["start time", "end time"]].agg(
            lambda x: (x.iloc[1] - x.iloc[0]).total_seconds() / 60, 1
        )
    )
    .groupby(["model", "dataset name"])
    .run_time.agg(["mean", "std"])
).aggregate(lambda x: f"{x.iloc[0]:.3f} $\pm$ {2 * x.iloc[1]:.3f}", 1)
print(train_run_time_table)

# %% Print LaTeX table
print(
    pd.concat(
        [
            train_run_time_table.rename("train"),
            eval_run_time_table.rename("evaluation"),
        ],
        axis=1,
    ).to_latex(
        caption="Runtime in Minutes for training and evaluation of models on benchmark data.\nVariance resulting deviations from 20 runs reported as standard deviation."
    )
)
