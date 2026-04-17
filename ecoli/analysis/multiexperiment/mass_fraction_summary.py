"""
Plot mean and spread of mass fraction (subcomponents of dry mass) over time
across multiple experiments. Uses the same mass subcomponents as the single-cell
mass_fraction_summary: Protein, tRNA, rRNA, mRNA, DNA, Small Mol, and Dry.

For each experiment (generation + lineage_seed), plots the mean mass fraction
across single cells with spread (confidence interval) over time.
"""

from typing import Any, TYPE_CHECKING, cast
import numpy as np
import os

from ecoli.library.parquet_emitter import read_stacked_columns
import altair as alt
import polars as pl

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")


def plot(
    params: dict[str, Any],
    conn: "DuckDBPyConnection",
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    # parse plot parameters from config
    group_by = params.get("group_by", "gen_seed")

    mass_columns = {
        "Protein": "listeners__mass__protein_mass",
        "tRNA": "listeners__mass__tRna_mass",
        "rRNA": "listeners__mass__rRna_mass",
        "mRNA": "listeners__mass__mRna_mass",
        "DNA": "listeners__mass__dna_mass",
        "Small Mol": "listeners__mass__smallMolecule_mass",
        "Dry": "listeners__mass__dry_mass",
    }

    mass_data = pl.DataFrame(
        read_stacked_columns(history_sql, list(mass_columns.values()), conn=conn)
    )

    # Add generation and lineage_seed columns for grouping
    mass_data = mass_data.with_columns(
        gen_seed=(
            pl.lit("gen=")
            + pl.col("generation").cast(pl.Utf8)
            + pl.lit(", seed=")
            + pl.col("lineage_seed").cast(pl.Utf8)
        ),
        sim_meta=(
            pl.concat_str(
                pl.lit("experiment_id="),
                pl.col("experiment_id"),
                pl.lit(", variant="),
                pl.col("variant").cast(pl.Utf8),
                pl.lit(", seed="),
                pl.col("lineage_seed").cast(pl.Utf8),
                pl.lit(", generation="),
                pl.col("generation").cast(pl.Utf8),
            )
        ),
    )

    # Relative time per (generation, lineage_seed) so each generation starts at t=0
    # and x-axis can be tight to data per facet
    if group_by == "gen_seed" or group_by == "generation":
        min_t = mass_data.group_by(["generation", "lineage_seed"]).agg(
            pl.col("time").min().alias("t_min")
        )
        mass_data = mass_data.join(min_t, on=["generation", "lineage_seed"])
        mass_data = mass_data.with_columns(
            ((pl.col("time") - pl.col("t_min")) / 60).alias("Time (min)")
        )
    else:
        mass_data = mass_data.with_columns(
            ((pl.col("time") - pl.col("time").min()) / 60).alias("Time (min)")
        )

    fractions = {
        k: cast(float, (mass_data[v] / mass_data["listeners__mass__dry_mass"]).mean())
        for k, v in mass_columns.items()
    }

    # create new dataframe for normalized growth to t=0
    assert group_by in mass_data.columns, (
        f"plot_by column '{group_by}' not found in data"
    )

    new_columns = {
        "Time (min)": mass_data["Time (min)"],
        "group_by": mass_data[group_by],
        "experiment_id": mass_data["experiment_id"],
        "generation": mass_data["generation"],
        "sim_meta": mass_data["sim_meta"],
        **{
            f"{k} ({fractions[k]:.3f})": mass_data[v] / mass_data[v][0]
            for k, v in mass_columns.items()
        },
    }

    mass_fold_change_df = pl.DataFrame(new_columns)

    # Long form to follow altair format (include generation so we can break lines at division)
    melted = mass_fold_change_df.melt(
        id_vars=["Time (min)", "group_by", "generation", "experiment_id", "sim_meta"],
        variable_name="Submass",
        value_name="Mass (normalized by t = 0 min)",
    )

    # Plot mean and confidence interval of mass fraction for each submass component, individual plots by group_by
    n_experiments = len(np.unique(mass_data["experiment_id"]))

    line = (
        alt.Chart()
        .mark_line(strokeWidth=0.5)
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "mean(Mass (normalized by t = 0 min)):Q", title="Mean mass fraction"
            ),
            color=alt.Color("Submass:N", legend=alt.Legend(title="Mass subcomponent")),
            detail=alt.Detail(
                "generation:N"
            ),  # separate path per generation so no line across division
        )
    )

    spread = (
        alt.Chart()
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("ci0(Mass (normalized by t = 0 min)):Q"),
            y2=alt.Y2("ci1(Mass (normalized by t = 0 min)):Q"),
            color=alt.Color("Submass:N", legend=alt.Legend(title="Mass subcomponent")),
            detail=alt.Detail(
                "generation:N"
            ),  # separate path per generation so no line across division
        )
    )

    # --- Save Plot 1: Combined plot with mean and spread ---
    figure_combined = (
        alt.layer(spread, line, data=melted)
        .facet(column=alt.Facet("group_by:N"))
        .resolve_scale(
            x="independent"
        )  # tight x-axis per facet, 0 to max for that group
        .properties(
            title=f"Mass fraction across {n_experiments} experiments grouped by {group_by}"
        )
    )
    out_path = os.path.join(outdir, "multiexperiment_mass_fraction_summary.html")
    figure_combined.save(out_path)

    # --- Save Plot 2: Individual Cell Component Mass per Experiment (Mean across gen and seed) ---
    figure_individual = (
        alt.layer(spread, line, data=melted)
        .facet(facet=alt.Facet("sim_meta:N"), columns=4)
        .resolve_scale(
            x="independent"
        )  # tight x-axis per facet, 0 to max for that group
        .properties(
            title="Mean Mass fraction of each experiment across the entire sim (gen and seed)"
        )
    )
    out_path = os.path.join(
        outdir, "multiexperiment_mass_fraction_summary_individual.html"
    )
    figure_individual.save(out_path)

    print(f"Saved multi-experiment mass fraction summary to: {out_path}")
    return
