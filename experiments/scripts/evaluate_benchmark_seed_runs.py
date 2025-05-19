# %% imports
import mlflow
import seaborn as sns

from hybrid_flows.utils.visualisation import get_figsize, setup_latex


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
fig_height = get_figsize(487.8225)[1]

exp_name = "seeds_2025-01-22"
mlflow.set_tracking_uri("http://localhost:5000")
exp = mlflow.get_experiment_by_name(exp_name)
eval_runs = mlflow.search_runs(
    experiment_names=[exp_name],
    filter_string="attributes.run_name like '%evaluation'",
)

# %% check if we have 20 runs per model
eval_runs.groupby("tags.mlflow.runName")["params.seed"].count()
assert (eval_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()

# %% eval runs
relevant_columns = [
    "params.seed",
    "params.dataset_name",
    "metrics.train_loss",
    "metrics.validation_loss",
    "metrics.test_loss",
]
df = eval_runs[relevant_columns]
df = df.assign(
    model=eval_runs["tags.mlflow.runName"].apply(
        lambda x: "HMAF" if "hybrid" in x else "MAF"
    )
)
df.columns = df.columns.map(lambda x: x.split(".")[-1].replace("_", " "))
loss_table = (
    df.dropna()
    .groupby(["model", "dataset name"])[
        [
            "train loss",
            "validation loss",
            "test loss",
        ]
    ]
    .aggregate(["mean", "min", "max", "sem", "std"])
)

# %% completed runs
pt = df.pivot_table(index=["model", "dataset name"], columns="seed", values="test loss")
pt.sort_index().isna()

# %%
latex_table = (
    loss_table.stack(0)
    .reset_index()
    .set_index(["model", "dataset name", "level_2"])[["mean", "sem"]]
    .aggregate(lambda x: f"{x[0]:.3f} $\pm$ {2 * x[1]:.3f}", 1)
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
    col_order=sorted(df["dataset name"].unique()),
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
