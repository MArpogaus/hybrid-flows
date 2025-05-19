# %% imports
import mlflow
import pandas as pd
import yaml


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


# %% globals
exp_name = "benchmark-seeds-2025-05-13"
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(
    experiment_names=[exp_name],
    # filter_string="attributes.run_name like '%evaluation'",
    # filter_string="attributes.run_name like 'unconditional_masked_autoregressive_flow_quadratic_spline_power_evaluation'"
)
eval_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("evaluation")]
train_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("train")]

# %% check if we have 20 runs per model
train_runs.groupby("tags.mlflow.runName")["params.seed"].count()
eval_runs.groupby("tags.mlflow.runName")["params.seed"].count()
assert (train_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()
assert (eval_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()

# %% unfinished runs
pt = eval_runs.pivot_table(
    index=["tags.mlflow.runName"],
    columns=["params.seed"],
    values="experiment_id",
    aggfunc="count",
)
print(pt.sort_index().to_markdown())

# %% load dataset names
with open("experiments/params/benchmark/dataset.yaml", "r") as f:
    dataset_kwargs = yaml.safe_load(f)
dataset_names = list(dataset_kwargs["dataset_kwargs"].keys())
dataset_names

# %% get run commands
run_cmds = pt.stack(dropna=False)
run_cmds = run_cmds[run_cmds.isna()]
print(len(run_cmds))

python = "srun --partition=gpu1 --gres=gpu:1 --mem=256GB --time=48:00:00 --export=ALL,MLFLOW_TRACKING_URI=http://login1:5000 python"

cmd_str = "dvc exp run --temp --pull -S 'python=\"{python}\"' -S 'seed={seed}' -S 'train-benchmark-experiment-name={exp_name}-DUPLICATED' -S 'eval-benchmark-experiment-name={exp_name}' eval-benchmark@dataset{dataset}-{model} &"


for _, row in run_cmds.reset_index().iterrows():
    model, dataset_name, _ = row["tags.mlflow.runName"].rsplit("_", 2)
    dataset_id = dataset_names.index(dataset_name)
    print(
        cmd_str.format(
            python=python,
            dataset=dataset_id,
            exp_name=exp_name,
            model=model,
            seed=row["params.seed"],
        )
    )
    print("sleep 10")


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
