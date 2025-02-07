# %% imports ###################################################################
import mlflow
import pandas as pd
import seaborn as sns
import yaml
from tqdm import tqdm

from mctm.utils.visualisation import get_figsize, setup_latex

# %%
with open("params/sim/dataset.yaml", "r") as f:
    dataset_kwargs = yaml.safe_load(f)
dataset_names = list(dataset_kwargs["dataset_kwargs"].keys())
dataset_names

# %% globals ###################################################################
fig_height = get_figsize(487.8225)[1]

exp_name = "sim_evaluation"
mlflow.set_tracking_uri("https://marcel-mlflow.ai4grids.ei.htwg-konstanz.de")

exp_name = "sim_seeds_2025-02-01"
exp_names = [
    "sim_seeds_2025-02-02_unconditional_multivariate_transformation_model",
    "sim_seeds_2025-02-02_unconditional_multivariate_normal",
    "sim_seeds_2025-02-02_unconditional_masked_autoregressive_flow_quadratic_spline",
    "sim_seeds_2025-02-02_unconditional_masked_autoregressive_flow_bernstein_poly",
    "sim_seeds_2025-02-02_unconditional_hybrid_coupling_flow_quadratic_spline",
    "sim_seeds_2025-02-02_unconditional_hybrid_coupling_flow_bernstein_poly",
    "sim_seeds_2025-02-02_unconditional_coupling_flow_quadratic_spline",
    "sim_seeds_2025-02-02_unconditional_coupling_flow_bernstein_poly",
    "sim_seeds_2025-02-02_conditional_multivariate_transformation_model",
    "sim_seeds_2025-02-02_conditional_multivariate_normal",
    "sim_seeds_2025-02-02_conditional_masked_autoregressive_flow_quadratic_spline",
    "sim_seeds_2025-02-02_conditional_masked_autoregressive_flow_bernstein_poly",
    "sim_seeds_2025-02-02_conditional_hybrid_coupling_flow_quadratic_spline",
    "sim_seeds_2025-02-02_conditional_hybrid_coupling_flow_bernstein_poly",
    "sim_seeds_2025-02-02_conditional_coupling_flow_quadratic_spline",
    "sim_seeds_2025-02-02_conditional_coupling_flow_bernstein_poly",
]
mlflow.set_tracking_uri("http://localhost:5000")
eval_runs = pd.concat(
    [
        mlflow.search_runs(
            experiment_names=[e],
            filter_string="attributes.run_name like '%evaluation'",
        )
        for e in tqdm(exp_names)
    ]
)


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
# %% get run commands
run_cmds = pt.stack(dropna=False)
run_cmds = run_cmds[run_cmds.isna()]

cmd_str = "sbatch --export=ALL,DATASET_NUM={dataset_id},MODEL_NAME={model},SEED={seed} slurm_sim_seeds.sh"

for _, row in run_cmds.reset_index().iterrows():
    model, dataset_name, _ = row.runName.rsplit("_", 2)
    dataset_id = dataset_names.index(dataset_name)
    print(cmd_str.format(dataset_id=dataset_id, model=model, seed=row.seed))
    print("sleep 10")

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


# %%
latex_table = (
    loss_table.stack(0)
    .reset_index()
    .set_index(["model", "dataset name", "level_2"])[["mean", "sem"]]
    .aggregate(lambda x: f"{x[0]:.3f} $\pm$ {2*x[1]:.3f}", 1)
    .unstack(1)
    .to_latex()
)
print(latex_table)

# %% plot
setup_latex(10)
g = sns.catplot(
    df,
    y="test loss",
    col="dataset name",
    hue="model",
    col_wrap=2,
    sharey=False,
    sharex=False,
    height=fig_height / 2,
    kind="box",
    palette="pastel",
    gap=0.2,
    fliersize=False,
    legend=True,
    # col_order=sorted(df["dataset name"].unique()),
)
g.map_dataframe(sns.swarmplot, y="test loss", hue="model", palette="muted", dodge=True)
g.tick_params(bottom=False)
sns.despine(offset=10, bottom=True)
sns.move_legend(
    g,
    loc="center left",
    bbox_to_anchor=(0.6, 0.18),
)
g.figure.savefig("seed_test_nll.pdf", bbox_inches="tight")

# %% analyze eval runtime
relevant_columns = [
    "params.seed",
    "params.dataset_name",
    "start_time",
    "end_time",
]
df = eval_runs[relevant_columns]
df.columns = df.columns.map(lambda x: x.split(".")[-1].replace("_", " "))


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


df = df.assign(model=eval_runs["tags.mlflow.runName"].apply(get_model_name))
# %%
eval_run_time_table = (
    df.assign(run_time=df[["start time", "end time"]].agg(lambda x: x[1] - x[0], 1))
    .groupby(["model", "dataset name"])
    .run_time.agg(["mean", "std"])
).aggregate(lambda x: f"{x[0].seconds / 60:.3f} $\pm$ {2*x[1].seconds / 60:.3f}", 1)
print(eval_run_time_table.to_markdown())

# %% analyze train runtime
train_runs = mlflow.search_runs(
    experiment_ids=exp.experiment_id,
    filter_string="attributes.run_name like '%train'",
)
relevant_columns = [
    "params.seed",
    "params.dataset_kwargs.dataset_name",
    "start_time",
    "end_time",
]
df = train_runs[relevant_columns]
df.columns = df.columns.map(lambda x: x.split(".")[-1].replace("_", " "))


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


df = df.assign(model=train_runs["tags.mlflow.runName"].apply(get_model_name))
# %%
train_run_time_table = (
    df.assign(run_time=df[["start time", "end time"]].agg(lambda x: x[1] - x[0], 1))
    .groupby(["model", "dataset name"])
    .run_time.agg(["mean", "std"])
).aggregate(lambda x: f"{x[0].seconds / 60:.3f} $\pm$ {2*x[1].seconds / 60:.3f}", 1)
print(train_run_time_table.to_markdown())
