# %% imports ###################################################################
import mlflow
import pandas as pd
import seaborn as sns

from hybrid_flows.utils.visualisation import get_figsize, setup_latex


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


# %% globals ###################################################################
fig_width, fig_height = get_figsize(487.8225, fraction=0.9)

exp_name = "malnutrition-seeds-2025-05-13"
mlflow.set_tracking_uri("http://localhost:5000")
runs = mlflow.search_runs(
    experiment_names=[exp_name],
)
eval_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("evaluation")]
train_runs = runs.loc[runs["tags.mlflow.runName"].str.endswith("train")]

# %% check if we have 20 runs per model
assert (train_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()
assert (eval_runs.groupby("tags.mlflow.runName")["params.seed"].count() == 20).all()

# %% eval runs
relevant_columns = [
    "tags.mlflow.runName",
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
loss_tab = df.groupby("model")["test loss"].agg(
    lambda x: f"{x.mean().round(3):.3f} $\pm$ {2 * x.std().round(3):.3f}",
)
print(
    loss_tab.sort_index(ascending=False).to_latex(
        multicolumn_format="c", float_format="%.4f"
    )
)

# %% plot
setup_latex(10)
g = sns.catplot(
    df,
    y="test loss",
    col="model",
    sharey=False,
    height=fig_height / 2,
    kind="box",
    gap=0.2,
    fliersize=False,
    legend=True,
    # col_order=sorted(df["dataset name"].unique()),
)
g.map_dataframe(sns.swarmplot, y="test loss", palette="muted", dodge=True)
g.tick_params(bottom=False)
sns.despine(offset=10, bottom=True)
sns.move_legend(
    g,
    loc="center left",
    bbox_to_anchor=(0.6, 0.18),
)
g.figure.savefig("seed_test_nll.pdf", bbox_inches="tight")

# %% parms plot ################################################################
# exp_name = "malnutrition_seeds_2025-02-06_5"
eval_runs = mlflow.search_runs(
    experiment_names=[exp_name],
    filter_string="attributes.run_name like '%evaluation'",
)
# %%
relevant_columns = [
    "run_id",
    "tags.mlflow.runName",
    "params.seed",
    "metrics.test_loss",
]
eval_run_ids = eval_runs[relevant_columns]  # .dropna()
eval_run_ids = eval_run_ids.assign(
    model=eval_runs["tags.mlflow.runName"].apply(get_model_abrev),
)

eval_run_ids
# %%
run_ids = eval_run_ids.set_index("model").loc["HMAF (B)"].run_id
# %%
dfs = []
for i, run_id in enumerate(run_ids):
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="malnutrition_conditional_hybrid_masked_autoregressive_flow_bernstein_poly_params.csv",
    )
    dfs.append(pd.read_csv(path).assign(seed=i))

params_df = pd.concat(dfs)
params_df.var_name = params_df.var_name.apply(lambda x: x.split("-")[0])
params_df = params_df.rename(
    columns={
        r"$\beta_x$": r"$-\beta_{j,x}$",
    }
)

# %%
setup_latex(10)
g = sns.relplot(
    params_df,
    x="cage",
    y=r"$-\beta_{j,x}$",
    row="var_name",
    kind="line",
    height=fig_height / 2,
    aspect=fig_width / fig_height,
    errorbar=("pi", 95),
    # linewidth=1,
    facet_kws=dict(margin_titles=True),
)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.tight_layout(w_pad=0)
g.figure.savefig(
    "malnutrition_marginal_shift_inv_seeds.pdf",
    bbox_inches="tight",
    transparent=True,
)

# %% Q-Q marginals #############################################################
dfs = []
for i, run_id in enumerate(run_ids):
    path = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="malnutrition_conditional_hybrid_masked_autoregressive_flow_bernstein_poly_qq_w_base.csv",
    )
    dfs.append(pd.read_csv(path).assign(seed=i))

qq_marginals_df = pd.concat(dfs)
qq_marginals_df = qq_marginals_df.loc[qq_marginals_df.var_name.str.endswith("-1")]
qq_marginals_df.var_name = qq_marginals_df.var_name.apply(lambda x: x.split("-")[0])

# %%
low = -5
high = 5
setup_latex(10)
g = sns.relplot(
    qq_marginals_df,
    x="Normal Quantile",
    y="$W$ Quantile",
    col="var_name",
    kind="line",
    height=fig_height / 2,
    errorbar=("ci", 95),
    # estimator=None,
    lw=0.5,
    # units="seed",
    facet_kws=dict(margin_titles=True, legend_out=False),
)
for ax in g.axes.flat:
    ax.plot([low, high], [low, high], "k:", linewidth=1)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set(aspect="equal", xticks=[], yticks=[], xlim=(low, high), ylim=(low, high))
g.tight_layout(w_pad=0)
g.figure.savefig(
    "malnutrition_qq_w_base_seeds.pdf",
    bbox_inches="tight",
    transparent=True,
)


# %% Q-Q plot ##################################################################
dfs = []
for _, row in eval_run_ids.iterrows():
    base_run_name = row["tags.mlflow.runName"].rsplit("_", 2)[0]
    path = mlflow.artifacts.download_artifacts(
        run_id=row.run_id,
        artifact_path=f"malnutrition_{base_run_name}_qq_data_samples.csv",
    )
    dfs.append(pd.read_csv(path).assign(seed=row["params.seed"], model=row.model))
qq_df = pd.concat(dfs)
qq_df = qq_df.loc[qq_df.var_name.str.endswith("-1")]
qq_df.var_name = qq_df.var_name.apply(lambda x: x.split("-")[0])

# %%
low = -4
high = 6.5
setup_latex(10)
g = sns.relplot(
    qq_df,
    x="Observed Quantile",
    y="Estimated Quantile",
    col="var_name",
    hue="model",
    kind="line",
    height=fig_height / 2,
    errorbar=("ci", 95),
    # estimator=None,
    # aspect=0.6,
    lw=0.5,
    # units="seed",
    facet_kws=dict(margin_titles=True, legend_out=False),
)
for ax in g.axes.flat:
    ax.plot([low, high], [low, high], "k:", linewidth=1)
g.set_titles(col_template="{col_name}", row_template="{row_name}")
g.set(aspect="equal", xticks=[], yticks=[], xlim=(low, high), ylim=(low, high))
sns.move_legend(
    g,
    "lower center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=3,
    title=None,
    frameon=False,
)
g.tight_layout(w_pad=0)
g.figure.savefig(
    "malnutrition_qq_data_samples_seeds.pdf",
    bbox_inches="tight",
    transparent=True,
)
