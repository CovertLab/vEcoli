"""
Plot mean and spread of mass fraction (subcomponents of dry mass) over time
across multiple experiments. Uses the same mass subcomponents as the single-cell
mass_fraction_summary: Protein, tRNA, rRNA, mRNA, DNA, Small Mol, and Dry.

For each experiment (generation + lineage_seed), plots the mean mass fraction
across single cells with spread (confidence interval) over time.
"""

from typing import Any, TYPE_CHECKING
import os

from ecoli.library.parquet_emitter import read_stacked_columns
import altair as alt
import polars as pl

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

MASS_QUERY_ALIASES = [
    ("protein_mass", "listeners__mass__protein_mass"),
    ("tRNA_mass", "listeners__mass__tRna_mass"),
    ("rRNA_mass", "listeners__mass__rRna_mass"),
    ("mRNA_mass", "listeners__mass__mRna_mass"),
    ("dna_mass", "listeners__mass__dna_mass"),
    ("smallMolecule_mass", "listeners__mass__smallMolecule_mass"),
    ("dry_mass", "listeners__mass__dry_mass"),
]
SUBMASS_DISPLAY_NAMES = [
    "Protein",
    "tRNA",
    "rRNA",
    "mRNA",
    "DNA",
    "Small Mol",
    "Dry",
]


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
    """Plot mean and spread of mass fractions over time per experiment."""
    query = [f"{col} AS {alias}" for alias, col in MASS_QUERY_ALIASES]
    query.append("time/60 AS time_min")

    raw = pl.DataFrame(
        read_stacked_columns(history_sql, query, order_results=True, conn=conn)
    )

    # Mass fraction = component / dry_mass for each row (per cell, per time)
    dry = raw["dry_mass"]
    fraction_cols = {}
    for alias, _ in MASS_QUERY_ALIASES:
        if alias == "dry_mass":
            fraction_cols["dry_frac"] = pl.lit(1.0)
        else:
            fraction_cols[f"{alias.replace('_mass', '')}_frac"] = pl.col(alias) / dry

    raw = raw.with_columns(**fraction_cols)

    # gen_seed for grouping/coloring like cell_mass.py
    raw = raw.with_columns(
        gen_seed=(
            pl.lit("gen=")
            + pl.col("generation").cast(pl.Utf8)
            + pl.lit(", seed=")
            + pl.col("lineage_seed").cast(pl.Utf8)
        )
    )

    # Long form: one row per (time_min, gen_seed, agent_id, Submass, fraction)
    frac_col_names = list(fraction_cols.keys())
    melted = raw.select(["time_min", "gen_seed", "agent_id"] + frac_col_names).melt(
        id_vars=["time_min", "gen_seed", "agent_id"],
        value_vars=frac_col_names,
        variable_name="frac_key",
        value_name="fraction",
    )

    # Map frac_key to display name (same order as MASS_QUERY_ALIASES)
    key_to_display = dict(zip(frac_col_names, SUBMASS_DISPLAY_NAMES))
    melted = melted.with_columns(Submass=pl.col("frac_key").replace(key_to_display))

    max_time_min = float(raw["time_min"].max())
    x_scale = alt.Scale(domain=[0, max_time_min], nice=False)

    # Mean line per (time_min, gen_seed, Submass)
    line_chart = (
        alt.Chart(melted)
        .mark_line(strokeWidth=0.5)
        .encode(
            x=alt.X("time_min:Q", title="Time (min)", scale=x_scale),
            y=alt.Y("mean(fraction):Q", title="Mean mass fraction"),
            color=alt.Color(
                "gen_seed:N", legend=alt.Legend(title="Generation and Seed")
            ),
        )
        .properties(width=450, height=220)
    ).interactive()

    # Spread (ci0–ci1) per (time_min, gen_seed, Submass)
    spread_chart = (
        alt.Chart(melted)
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X("time_min:Q", title="Time (min)", scale=x_scale),
            y=alt.Y("ci0(fraction):Q"),
            y2=alt.Y2("ci1(fraction):Q"),
            color=alt.Color(
                "gen_seed:N", legend=alt.Legend(title="Generation and Seed")
            ),
        )
        .properties(width=450, height=220)
    ).interactive()

    layered = (spread_chart + line_chart).properties(
        title="Mean mass fraction with spread across single cells"
    )

    # One facet per mass subcomponent
    figure = layered.facet(
        row=alt.Facet("Submass:N", title="Mass component", sort=SUBMASS_DISPLAY_NAMES),
    ).resolve_scale(y="independent")

    out_path = os.path.join(outdir, "multiexperiment_mass_fraction_summary.html")
    figure.save(out_path)
    print(f"Saved multi-experiment mass fraction summary to: {out_path}")
    return
