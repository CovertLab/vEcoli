"""
Plot mean and spread of mass fraction (subcomponents of dry mass) over time
for multivariant simulation.

One subplot per variant, stacked vertically. Within each variant the mean
(± CI spread) of each mass subcomponent (normalized by t=0) is shown over
cell-cycle-relative time, with lines broken at cell division.
"""

from typing import Any, TYPE_CHECKING, cast
import os

from ecoli.library.parquet_emitter import read_stacked_columns
import altair as alt
import polars as pl

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

MASS_COLUMNS = {
    "Protein": "listeners__mass__protein_mass",
    "tRNA": "listeners__mass__tRna_mass",
    "rRNA": "listeners__mass__rRna_mass",
    "mRNA": "listeners__mass__mRna_mass",
    "DNA": "listeners__mass__dna_mass",
    "Small Mol": "listeners__mass__smallMolecule_mass",
    "Dry": "listeners__mass__dry_mass",
}


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
    """Plot mean mass fraction over time, one subplot per variant."""
    mass_data = pl.DataFrame(
        read_stacked_columns(history_sql, list(MASS_COLUMNS.values()), conn=conn)
    )

    # Relative time per (generation, lineage_seed) so each generation starts at t=0
    min_t = mass_data.group_by(["generation", "lineage_seed"]).agg(
        pl.col("time").min().alias("t_min")
    )
    mass_data = mass_data.join(min_t, on=["generation", "lineage_seed"])
    mass_data = mass_data.with_columns(
        ((pl.col("time") - pl.col("t_min")) / 60).alias("Time (min)")
    )

    fractions = {
        k: cast(float, (mass_data[v] / mass_data["listeners__mass__dry_mass"]).mean())
        for k, v in MASS_COLUMNS.items()
    }

    new_columns = {
        "Time (min)": mass_data["Time (min)"],
        "variant": mass_data["variant"],
        "generation": mass_data["generation"],
        "lineage_seed": mass_data["lineage_seed"],
        **{
            f"{k} ({fractions[k]:.3f})": mass_data[v] / mass_data[v][0]
            for k, v in MASS_COLUMNS.items()
        },
    }
    mass_fold_change_df = pl.DataFrame(new_columns)

    melted = mass_fold_change_df.melt(
        id_vars=["Time (min)", "variant", "generation", "lineage_seed"],
        variable_name="Submass",
        value_name="Mass (normalized by t = 0 min)",
    )

    line = (
        alt.Chart()
        .mark_line(strokeWidth=0.5)
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "mean(Mass (normalized by t = 0 min)):Q", title="Mean mass fraction"
            ),
            color=alt.Color("Submass:N", legend=alt.Legend(title="Mass subcomponent")),
            # Break line at cell division
            detail=alt.Detail("generation:N"),
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
            detail=alt.Detail("generation:N"),
        )
    )

    variants = melted["variant"].unique().sort()
    plots = []
    for variant_val in variants:
        variant_name = variant_names.get(variant_val, f"Variant {variant_val}")
        variant_melted = melted.filter(pl.col("variant") == variant_val).to_pandas()

        subplot = (
            alt.layer(spread, line, data=variant_melted)
            .resolve_scale(x="independent")
            .properties(width=600, height=250, title=variant_name)
        )
        plots.append(subplot)

    final = (
        alt.vconcat(*plots)
        .resolve_scale(x="independent", y="independent")
        .properties(title="Mass Fraction by Variant")
    )

    out_path = os.path.join(outdir, "mass_fraction_summary.html")
    final.save(out_path)
    print(f"Saved multivariant mass fraction summary to: {out_path}")
