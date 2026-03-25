import marimo

__generated_with = "0.16.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    mo.md(
        """
        # Comparative simulation output (model performance): Reference vs "Expt" ParCa Outputs
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    import duckdb
    from ecoli.library.parquet_emitter import dataset_sql, read_stacked_columns

    return Path, dataset_sql, duckdb, pd, plt, read_stacked_columns, sns


@app.cell
def _(Path):
    id_ref = "test_rnaseq_ingestion_defaults_sim"
    id_expt = "test_rnaseq_ingestion_gbw0002_from_simdata"

    outdir_ref = Path(f"out/{id_ref}")
    outdir_expt = Path(f"out/{id_expt}")

    # Paths to multiseed analysis outputs for each experiment
    multiseed_ref = outdir_ref / "analyses" / "variant=0" / "plots"
    multiseed_expt = outdir_expt / "analyses" / "variant=0" / "plots"

    higher_ref_path = multiseed_ref / "higher_order_properties.tsv"
    higher_expt_path = multiseed_expt / "higher_order_properties.tsv"

    print(
        higher_ref_path,
        "\n",
        higher_expt_path,
    )
    return higher_expt_path, higher_ref_path, id_expt, id_ref


@app.cell
def _(higher_expt_path, higher_ref_path, pd):
    # Load higher-order properties
    higher_ref = pd.read_csv(higher_ref_path, sep="\t")
    higher_expt = pd.read_csv(higher_expt_path, sep="\t")

    higher_ref["run"] = "ref"
    higher_expt["run"] = "expt"
    higher_all = pd.concat([higher_ref, higher_expt], ignore_index=True)

    higher_all.head()
    return (higher_all,)


@app.cell
def _(higher_all, pd):
    # Extract per-cell distributions for cell mass and cell volume

    # Columns that are actual cells (everything except Properties/mean/std/run)
    cell_cols = [
        c for c in higher_all.columns if c not in ("Properties", "mean", "std", "run")
    ]

    # Helper to extract one property into long form: [run, metric, value]
    def property_long(prop_label: str, metric_name: str) -> pd.DataFrame:
        df_prop = higher_all[higher_all["Properties"] == prop_label].copy()
        if df_prop.empty:
            return pd.DataFrame(columns=["run", "metric", "value"])
        long = df_prop.melt(
            id_vars=["run"],
            value_vars=cell_cols,
            var_name="cell_id",
            value_name="value",
        )
        long["metric"] = metric_name
        return long[["run", "metric", "value"]]

    mass_long = property_long("Cell mass (mg/10^9 cells)", "Cell mass (mg/10^9 cells)")
    volume_long = property_long("Cell volume (um^3)", "Cell volume (um^3)")
    return mass_long, volume_long


@app.cell
def _(volume_long):
    volume_long
    return


@app.cell
def _(dataset_sql, id_expt, id_ref):
    # Build DuckDB SQL for the stacked history for each experiment
    history_sql_ref, config_sql_ref, success_sql_ref = dataset_sql("out", [id_ref])
    history_sql_expt, config_sql_expt, success_sql_expt = dataset_sql("out", [id_expt])
    return history_sql_expt, history_sql_ref, success_sql_expt, success_sql_ref


@app.cell
def _(duckdb, read_stacked_columns):
    def load_doubling_times(history_sql, success_sql, experiment_id: str):
        """
        Return a DataFrame with columns:
        ['Doubling Time (hr)', 'experiment_id', 'variant',
         'lineage_seed', 'generation', 'agent_id'].
        """
        # Subquery with just 'time' (and id columns) for all sims
        time_subquery = read_stacked_columns(
            history_sql,
            ["time"],
            remove_first=False,
            success_sql=success_sql,
        )

        conn = duckdb.connect()
        dt_df = conn.sql(
            f"""
            SELECT
                (MAX(time) - MIN(time)) / 3600.0 AS "Doubling Time (hr)",
                experiment_id,
                variant,
                lineage_seed,
                generation,
                agent_id
            FROM ({time_subquery})
            WHERE experiment_id = '{experiment_id}'
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
            ORDER BY experiment_id, variant, lineage_seed, generation, agent_id
            """
        ).to_df()
        conn.close()
        return dt_df

    return (load_doubling_times,)


@app.cell
def _(duckdb, read_stacked_columns):
    def load_protein_mass_per_cell(history_sql, success_sql, experiment_id: str):
        """
        Returns a DataFrame with columns:
        ['protein_mass_fg', 'experiment_id', 'variant',
         'lineage_seed', 'generation', 'agent_id'].
        Values are time-averaged protein mass per cell.
        """
        subquery = read_stacked_columns(
            history_sql,
            ["listeners__mass__protein_mass"],
            remove_first=False,
            success_sql=success_sql,
        )
        conn = duckdb.connect()
        df = conn.sql(
            f"""
            SELECT
                AVG(listeners__mass__protein_mass) AS protein_mass_fg,
                experiment_id,
                variant,
                lineage_seed,
                generation,
                agent_id
            FROM ({subquery})
            WHERE experiment_id = '{experiment_id}'
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
            ORDER BY experiment_id, variant, lineage_seed, generation, agent_id
            """
        ).to_df()
        conn.close()
        return df

    return (load_protein_mass_per_cell,)


@app.cell
def _():
    return


@app.cell
def _(
    history_sql_expt,
    history_sql_ref,
    id_expt,
    id_ref,
    load_doubling_times,
    load_protein_mass_per_cell,
    mass_long,
    pd,
    success_sql_expt,
    success_sql_ref,
    volume_long,
):
    dt_ref = load_doubling_times(history_sql_ref, success_sql_ref, id_ref)
    dt_expt = load_doubling_times(history_sql_expt, success_sql_expt, id_expt)

    prot_ref = load_protein_mass_per_cell(history_sql_ref, success_sql_ref, id_ref)
    prot_expt = load_protein_mass_per_cell(history_sql_expt, success_sql_expt, id_expt)

    dt_ref["run"] = "ref"
    dt_expt["run"] = "expt"
    # dt_all = dt_ref
    dt_all = pd.concat([dt_ref, dt_expt], ignore_index=True)

    # Doubling time long form: [run, metric, value]
    dt_long = dt_all.rename(columns={"Doubling Time (hr)": "value"})[
        ["run", "value"]
    ].copy()
    dt_long["metric"] = "Doubling Time (hr)"
    dt_long = dt_long[["run", "metric", "value"]]

    prot_ref["run"] = "ref"
    prot_expt["run"] = "expt"
    prot_all = pd.concat([prot_ref, prot_expt], ignore_index=True)

    prot_long = prot_all.rename(columns={"protein_mass_fg": "value"})[
        ["run", "value"]
    ].copy()
    prot_long["metric"] = "Protein mass (fg)"
    prot_long = prot_long[["run", "metric", "value"]]

    # Combine all three metrics
    metrics_all = pd.concat(
        [dt_long, mass_long, volume_long, prot_long], ignore_index=True
    )
    metrics_all.head()

    # # Example: distribution of per-cell generation times
    # alt.Chart(
    #     # dt_all
    #     dt_ref
    #          ).mark_boxplot().encode(
    #     x="run:N",
    #     y="Doubling Time (hr):Q",
    #     color="run:N",
    # ).properties(title="Per-cell generation times (ref vs expt)")
    return (metrics_all,)


@app.cell
def _(metrics_all, plt, sns):
    # 1. Create the base violin plots
    g = sns.catplot(
        data=metrics_all,
        x="run",
        y="value",
        col="metric",
        hue="run",
        kind="violin",
        inner=None,  # Removes the default mini-boxplot for a cleaner look
        alpha=0.4,  # Makes violins slightly transparent so points pop
        height=5,
        aspect=0.8,
        sharey=False,
    )

    # 2. Map the strip plot onto each facet
    g.map_dataframe(
        sns.stripplot,
        x="run",
        y="value",
        hue="run",
        palette="dark:black",  # Makes points dark for contrast
        alpha=0.6,  # Semi-transparent points
        dodge=True,  # Aligns points with the color-coded violins
    )

    # 3. Clean up the aesthetics
    g.fig.suptitle("Ref vs Expt Distribution with Raw Data Points", y=1.05)
    g.set_titles("{col_name}")
    g.set_axis_labels("Run", "Value")

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
