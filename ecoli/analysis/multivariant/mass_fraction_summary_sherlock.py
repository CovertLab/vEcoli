"""
Plot mean mass fraction (subcomponents of dry mass) over time for
multivariant simulation, faceted by variant.

Size-optimized variant of ``mass_fraction_summary.py`` for sweeps with
100+ variants: the per-(variant, generation, Submass, time bin) averaging
is pushed into DuckDB instead of shipping every raw timestep to Vega's
client-side ``mean()``/``ci0()``/``ci1()`` aggregation, and variants are
faceted into one chart instead of vconcat-ing one independently-compiled
chart per variant. The confidence-interval band is dropped: it only
reflects spread across lineage_seeds, and most large sweeps run a single
seed per variant, making the band collapse to zero width anyway. Use
``mass_fraction_summary.py`` instead for small variant counts where the
per-seed spread is meaningful.
"""

import os
from typing import TYPE_CHECKING, Any

import altair as alt
import polars as pl

from ecoli.analysis.multivariant.utils import compute_variant_grid, create_variant_label
from ecoli.library.parquet_emitter import read_stacked_columns

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

DEFAULT_TIME_BIN_MIN = 1.0

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
    """Plot mean mass fraction over time, one facet per variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )
    time_bin_min = params.get("time_bin_min", DEFAULT_TIME_BIN_MIN)

    raw_sql = read_stacked_columns(
        history_sql, list(MASS_COLUMNS.values()), order_results=False
    )

    # Average fraction of dry mass per submass, for legend labels
    # (e.g. "Protein (0.431)"), matching mass_fraction_summary.py.
    fraction_selects = ", ".join(
        f"AVG({col} / listeners__mass__dry_mass) AS {name.replace(' ', '_')}_frac"
        for name, col in MASS_COLUMNS.items()
    )
    fractions_row = conn.sql(f"SELECT {fraction_selects} FROM ({raw_sql})").fetchone()
    fractions = dict(zip(MASS_COLUMNS.keys(), fractions_row))
    labels = {name: f"{name} ({fractions[name]:.3f})" for name in MASS_COLUMNS}

    # Reference row (t=0 of the *first* generation) for each lineage
    # (variant, lineage_seed), used to normalize that lineage's whole
    # multi-generation trajectory continuously -- mass_fraction_summary.py
    # instead resets both the time axis and the normalization reference at
    # every generation boundary, which stacks all generations on top of
    # each other (they all start at 1.0 and cover roughly the same
    # 0-45 min span) and makes every variant's plot look like the same
    # fuzzy band. Normalizing and timing off a single per-lineage t0
    # instead makes time flow continuously across generations (division
    # shows up as a real drop in mass fraction, not a reset to 1.0).
    safe_names = {name: name.replace(" ", "_") for name in MASS_COLUMNS}
    t0_selects = ", ".join(
        f"{col} AS {safe_names[name]}_t0" for name, col in MASS_COLUMNS.items()
    )
    t0_sql = f"""
        SELECT DISTINCT ON (variant, lineage_seed)
            variant, lineage_seed, time AS time_ref, {t0_selects}
        FROM ({raw_sql})
        ORDER BY variant, lineage_seed, generation, time
    """

    norm_selects = ", ".join(
        f'r.{col} / t0.{safe_names[name]}_t0 AS "{labels[name]}"'
        for name, col in MASS_COLUMNS.items()
    )
    unpivot_cols = ", ".join(f'"{labels[name]}"' for name in MASS_COLUMNS)
    normalized_sql = f"""
        SELECT
            r.variant, r.generation,
            FLOOR(((r.time - t0.time_ref) / 60.0) / {time_bin_min}) * {time_bin_min}
                AS "Time (min)",
            {norm_selects}
        FROM ({raw_sql}) r
        JOIN ({t0_sql}) t0
            USING (variant, lineage_seed)
    """
    agg_sql = f"""
        SELECT variant, generation, "Time (min)", Submass,
            AVG(mass_norm) AS mass_norm
        FROM ({normalized_sql})
        UNPIVOT (mass_norm FOR Submass IN ({unpivot_cols}))
        GROUP BY variant, generation, "Time (min)", Submass
    """
    agg = conn.sql(agg_sql).pl().with_columns(pl.col("mass_norm").round(4))

    def _make_label(v: int) -> str:
        raw_label = create_variant_label(v, per_variant_params)
        return "\n".join(raw_label) if isinstance(raw_label, list) else raw_label

    variant_label_map = {v: _make_label(v) for v in agg["variant"].unique()}
    agg = agg.with_columns(
        pl.col("variant").replace_strict(variant_label_map).alias("variant_label")
    )

    # Row-major grid order (grouped by first/second sweep param, baseline
    # first), wrapped into a grid instead of one tall column -- a single
    # column of 100+ variants exceeds the browser's max canvas height and
    # renders as a broken image.
    _, grid_columns, ordered_variant_ids = compute_variant_grid(per_variant_params)
    variant_sort_order = [
        variant_label_map[v] for v in ordered_variant_ids if v in variant_label_map
    ]

    final = (
        alt.Chart(agg.to_pandas())
        .mark_line(strokeWidth=0.5)
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("mass_norm:Q", title="Mean mass fraction"),
            color=alt.Color("Submass:N", legend=alt.Legend(title="Mass subcomponent")),
            detail=alt.Detail("generation:N"),
        )
        .properties(width=280, height=200)
        .facet(
            facet=alt.Facet(
                "variant_label:N",
                title="Variant",
                sort=variant_sort_order,
                header=alt.Header(labelExpr="split(datum.value, '\\n')"),
            ),
            columns=grid_columns,
        )
        .resolve_scale(x="independent", y="independent")
        .properties(title="Mass Fraction by Variant")
    )

    out_path = os.path.join(outdir, "mass_fraction_summary_sherlock.html")
    final.save(out_path)
    print(f"Saved multivariant mass fraction summary (Sherlock-scale) to: {out_path}")
