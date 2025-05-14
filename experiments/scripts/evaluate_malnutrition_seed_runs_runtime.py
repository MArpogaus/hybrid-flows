# %% imports
import mlflow
import pandas as pd


# %% functions
def get_model_abrev(run_name):
    model_abrev = "ERR"
    if "multivariate_normal" in run_name:
        model_abrev = "MVN"
    elif "transformation_model" in run_name:
        model_abrev = "MCTM"
    elif "coupling_flow" in run_name:
        model_abrev = "CF"
    elif "masked_autoregressive_flow" in run_name:
        model_abrev = "MAF"

    if "hybrid" in run_name:
        model_abrev = "H" + model_abrev

    if "spline" in run_name:
        model_abrev = model_abrev + " (S)"
    elif "bernstein" in run_name:
        model_abrev = model_abrev + " (B)"

    return model_abrev


# %% globals
exp_name = "malnutrition-seeds-2025-05-13"
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(
    experiment_names=[exp_name],
)
eval_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("evaluation")]
train_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("train")]
train_runs.groupby("tags.mlflow.runName")["params.seed"].count()

# %% check if we have 20 runs per model
assert (train_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()
assert (eval_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()

# %% analyze eval runtime
relevant_columns = [
    "params.seed",
    "start_time",
    "end_time",
]
df = eval_runs[relevant_columns]
df = df.assign(
    conditional=eval_runs["tags.mlflow.runName"].apply(
        lambda x: x.startswith("conditional")
    ),
    model=eval_runs["tags.mlflow.runName"].apply(get_model_abrev),
)
df.columns = df.columns.map(lambda x: x.split(".")[-1].replace("_", " "))
df = df.assign(
    run_time=df[["start time", "end time"]].agg(
        lambda x: (x.iloc[1] - x.iloc[0]).total_seconds(), 1
    )
)

eval_run_time_table = (df.groupby("model").run_time.agg(["mean", "std"])).aggregate(
    lambda x: f"{x.iloc[0]:.3f} $\pm$ {2 * x.iloc[1]:.3f}", 1
)
eval_run_time_table

# %% analyze train runtime
relevant_columns = [
    "params.seed",
    "start_time",
    "end_time",
]
df = train_runs[relevant_columns]
df = df.assign(
    conditional=train_runs["tags.mlflow.runName"].apply(
        lambda x: x.startswith("conditional")
    ),
    model=train_runs["tags.mlflow.runName"].apply(get_model_abrev),
)
df.columns = df.columns.map(lambda x: x.split(".")[-1].replace("_", " "))
df = df.assign(
    run_time=df[["start time", "end time"]].agg(
        lambda x: (x.iloc[1] - x.iloc[0]).total_seconds(), 1
    )
)

train_run_time_table = (df.groupby("model").run_time.agg(["mean", "std"])).aggregate(
    lambda x: f"{x.iloc[0]:.3f} $\pm$ {2 * x.iloc[1]:.3f}", 1
)
train_run_time_table

# %% Print LaTeX table
print(
    pd.concat(
        [
            train_run_time_table.rename("train"),
            eval_run_time_table.rename("evaluation"),
        ],
        axis=1,
    ).to_latex(
        caption="Mean runtime in seconds for training and evaluation of models on malnutrition data.\nVariance resulting deviations from 20 runs reported as standard deviation."
    )
)
