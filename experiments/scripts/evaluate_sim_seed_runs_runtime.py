# %% imports
import mlflow
import pandas as pd
import yaml
from tqdm import tqdm

# %% globals
date = "2025-04-10"
exp_names = [
    f"sim-seeds-unconditional_multivariate_transformation_model-{date}",
    f"sim-seeds-unconditional_multivariate_normal-{date}",
    f"sim-seeds-unconditional_masked_autoregressive_flow_quadratic_spline-{date}",
    f"sim-seeds-unconditional_masked_autoregressive_flow_bernstein_poly-{date}",
    f"sim-seeds-unconditional_hybrid_coupling_flow_quadratic_spline-{date}",
    f"sim-seeds-unconditional_hybrid_coupling_flow_bernstein_poly-{date}",
    f"sim-seeds-unconditional_coupling_flow_quadratic_spline-{date}",
    f"sim-seeds-unconditional_coupling_flow_bernstein_poly-{date}",
    f"sim-seeds-conditional_multivariate_transformation_model-{date}",
    f"sim-seeds-conditional_multivariate_normal-{date}",
    f"sim-seeds-conditional_masked_autoregressive_flow_quadratic_spline-{date}",
    f"sim-seeds-conditional_masked_autoregressive_flow_bernstein_poly-{date}",
    f"sim-seeds-conditional_hybrid_coupling_flow_quadratic_spline-{date}",
    f"sim-seeds-conditional_hybrid_coupling_flow_bernstein_poly-{date}",
    f"sim-seeds-conditional_coupling_flow_quadratic_spline-{date}",
    f"sim-seeds-conditional_coupling_flow_bernstein_poly-{date}",
]
mlflow.set_tracking_uri("http://localhost:5000")
runs = pd.concat(
    [
        mlflow.search_runs(
            experiment_names=[e],
            # filter_string="attributes.run_name like '%evaluation'",
        )
        for e in tqdm(exp_names)
    ]
)
eval_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("evaluation")]
train_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("train")]

# %%
eval_runs["tags.mlflow.runName"].unique()

# %% finished runs
pt = eval_runs.pivot_table(
    index=["tags.mlflow.runName"],
    columns=["params.seed"],
    values="experiment_id",
    aggfunc="count",
)
print(pt.sort_index().to_markdown())

# %% load dataset names
with open("experiments/params/sim/dataset.yaml", "r") as f:
    dataset_kwargs = yaml.safe_load(f)
dataset_names = list(dataset_kwargs["dataset_kwargs"].keys())
dataset_names

# %% get run commands
run_cmds = pt.stack(dropna=False)
run_cmds = run_cmds[run_cmds.isna()]

exp_name = "sim-seeds-{model}-{date}"
cmd_str = "dvc exp run --force --queue -S 'seed={seed}' -S 'train-sim-experiment-name={exp_name}' -S 'train-sim-experiment-name={exp_name}' eval-sim@dataset{dataset}-{model}"  # noqa: E501


for _, row in run_cmds.reset_index().iterrows():
    model, dataset_name, _ = row["tags.mlflow.runName"].rsplit("_", 2)
    dataset_id = dataset_names.index(dataset_name)
    print(
        cmd_str.format(
            dataset=dataset_id,
            exp_name=exp_name.format(model=model, date=date),
            model=model,
            seed=row["params.seed"],
        )
    )
    print("sleep 10")


# %% check if we have 20 runs per model
train_runs.groupby("tags.mlflow.runName")["params.seed"].count()
eval_runs.groupby("tags.mlflow.runName")["params.seed"].count()
assert (train_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()
assert (eval_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()


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


# %% analyze eval runtime
relevant_columns = [
    "params.seed",
    "params.dataset_name",
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

eval_run_time_table = (
    df.groupby(["dataset name", "conditional", "model"]).run_time.agg(["mean", "std"])
).aggregate(lambda x: f"{x.iloc[0]:.3f} $\pm$ {2 * x.iloc[1]:.3f}", 1)
eval_run_time_table

# %% analyze train runtime
relevant_columns = [
    "params.seed",
    "params.dataset_kwargs.dataset_name",
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

train_run_time_table = (
    df.groupby(["dataset name", "conditional", "model"]).run_time.agg(["mean", "std"])
).aggregate(lambda x: f"{x.iloc[0]:.3f} $\pm$ {2 * x.iloc[1]:.3f}", 1)
train_run_time_table

# %% Print LaTeX table
caption = (
    "Mean runtime in seconds for training and evaluation of models on simulated data.\n"
    "Variance resulting deviations from 20 runs reported as standard deviation."
)
print(
    pd.concat(
        [
            train_run_time_table.rename("train"),
            eval_run_time_table.rename("evaluation"),
        ],
        axis=1,
    ).to_latex(caption=caption)
)
