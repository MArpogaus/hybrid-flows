# %% imports ###################################################################
import mlflow
import pandas as pd
import yaml
from tqdm import tqdm

from hybrid_flows.utils.visualisation import get_figsize


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


# %%
with open("experiments/params/sim/dataset.yaml", "r") as f:
    dataset_kwargs = yaml.safe_load(f)
dataset_names = list(dataset_kwargs["dataset_kwargs"].keys())
dataset_names

# %% globals ###################################################################
fig_height = get_figsize(487.8225)[1]

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

# %% check if we have 20 runs per model
assert (train_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()
assert (eval_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()

# %% eval runs
relevant_columns = [
    "tags.mlflow.runName",
    "params.dataset_name",
    "params.seed",
    "metrics.test_loss",
]
df = eval_runs[relevant_columns]  # .dropna()
df = df.assign(
    conditional=eval_runs["tags.mlflow.runName"].apply(
        lambda x: x.startswith("conditional")
    ),
    model=eval_runs["tags.mlflow.runName"].apply(get_model_abrev),
)
df.columns = df.columns.map(lambda x: x.split(".")[-1].replace("_", " "))

# %% finished runs
pt = df.pivot_table(
    index=["runName"],
    columns=["seed"],
    values="test loss",
    aggfunc="count",
)
print(pt.sort_index().to_markdown())

# %% completed runs
pt = df.pivot_table(
    index=["model"],
    columns=["dataset name", "conditional"],
    values="test loss",
    aggfunc=lambda x: f"{x.mean().round(3):.3f} $\pm$ {2*x.std().round(3):.3f}",
)
print(
    pt.sort_index(ascending=False)
    .round(4)
    .to_latex(multicolumn_format="c", float_format="%.4f")
)

# %%
pt = df.pivot_table(
    index=["model"],
    columns=["dataset name", "conditional"],
    values="test loss",
    aggfunc="min",
)
print(pt.sort_index(ascending=False).round(4).to_markdown())

# %%
pt = df.pivot_table(
    index=["model"],
    columns=["dataset name", "conditional"],
    values="test loss",
    aggfunc="first",
)
print(pt.sort_index(ascending=False).round(4).to_markdown())
