"""
Distribution of time-mean estimated fluxes across reaction pairs for multivariant
simulation.

For each variant, each FBA reaction's estimated flux is averaged over time and
cells, then forward and reverse directions are summed into one value per
reaction pair. All variants are overlaid in one log-scale histogram with
semi-transparent bars, so their distributions can be compared directly. Pairs
with zero flux can't sit on a log axis, so they are counted in a separate
"zero flux" panel on the left, and each variant's zero count is in its legend
label.

Params:
    in_molar (bool, default True): plot mmol/L per step (counts x counts_to_molar)
        instead of molecules per step.
    is_reduxclassic (bool, default True): drop the first emitted step (empty
        placeholder for metabolism_redux_classic).
    bin_edges (list[float], optional): histogram edges; values <= 0 are ignored
        (zero has its own panel). Default: n_bins log-spaced edges spanning the
        nonzero data.
    n_bins (int, default 40): number of log bins when bin_edges isn't given.
    zero_threshold (float, default 0): pairs with mean flux <= this count as zero.
    y_scale ("log" or "linear", default "log"): y-axis scale.
    subplot_width, subplot_height (int): size of the histogram panel.
"""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING, cast

import altair as alt
import numpy as np
import polars as pl

from ecoli.analysis.multivariant.utils import compute_variant_grid, create_variant_label
from ecoli.library.parquet_emitter import (
    METADATA_PREFIX,
    field_metadata,
    ndlist_to_ndarray,
    read_stacked_columns,
)
from reconstruction.ecoli.dataclasses.process.metabolism import REVERSE_TAG

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

BAR_OPACITY = 0.4


def _pair_index(reaction_names: list[str]) -> tuple[np.ndarray, int]:
    """Index of each FBA reaction's forward/reverse pair (id minus REVERSE_TAG)."""
    keys = [
        r[: -len(REVERSE_TAG)] if r.endswith(REVERSE_TAG) else r for r in reaction_names
    ]
    index = {k: i for i, k in enumerate(dict.fromkeys(keys))}
    return np.array([index[k] for k in keys]), len(index)


def _diversity_weights(conn: "DuckDBPyConnection", config_sql: str) -> dict[int, float]:
    """Diversity weight per variant, from the diversity_term listener metadata."""
    column = f"{METADATA_PREFIX}listeners__fba_results__diversity_term"
    try:
        rows = conn.sql(
            f'SELECT variant, first("{column}") FROM ({config_sql}) GROUP BY variant'
        ).fetchall()
    except Exception:
        return {}
    weights = {}
    for variant, value in rows:
        value = value[0] if isinstance(value, list) and value else value
        if value is not None:
            weights[int(variant)] = float(value)
    return weights


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
) -> None:
    """Overlaid histogram of per-pair mean estimated flux, one color per variant."""
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )
    _, _, ordered_variant_ids = compute_variant_grid(per_variant_params)

    in_molar = params.get("in_molar", True)
    is_reduxclassic = params.get("is_reduxclassic", True)
    bin_edges_param = params.get("bin_edges")
    n_bins = int(params.get("n_bins", 40))
    zero_threshold = float(params.get("zero_threshold", 0.0))
    y_scale = params.get("y_scale", "log")
    subplot_w = int(params.get("subplot_width", 600))
    subplot_h = int(params.get("subplot_height", 350))

    hist_subquery = cast(
        str,
        read_stacked_columns(
            history_sql,
            [
                "listeners__fba_results__estimated_fluxes AS estimated_fluxes",
                "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar",
            ],
            order_results=True,
            success_sql=success_sql,
            remove_first=is_reduxclassic,
        ),
    )
    flux_expr = (
        "unnest(estimated_fluxes) * counts_to_molar"
        if in_molar
        else "unnest(estimated_fluxes)"
    )

    # Time-mean flux per FBA reaction, per cell
    raw = conn.sql(
        f"""
        WITH unnested AS (
            SELECT {flux_expr} AS flux,
                generate_subscripts(estimated_fluxes, 1) AS idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({hist_subquery})
        ),
        avg_flux AS (
            SELECT avg(flux) AS avg_flux,
                experiment_id, variant, lineage_seed, generation, agent_id, idx
            FROM unnested
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id, idx
        )
        SELECT list(avg_flux ORDER BY idx) AS estimated_fluxes,
            experiment_id, variant, lineage_seed, generation, agent_id
        FROM avg_flux
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).pl()

    if raw.is_empty():
        print("mean_flux_distribution: no rows returned; skipping.")
        return

    flux_matrix = ndlist_to_ndarray(raw["estimated_fluxes"])
    unit_label = "mmol/L per step" if in_molar else "molecules per step"

    # estimated_fluxes share solution_fluxes' column order (the FBA reaction names)
    reaction_names = field_metadata(
        conn, config_sql, "listeners__fba_results__solution_fluxes"
    )
    if len(reaction_names) != flux_matrix.shape[1]:
        raise ValueError(
            f"{len(reaction_names)} reaction names but {flux_matrix.shape[1]} flux columns"
        )
    pair_of_rxn, n_pairs = _pair_index(list(reaction_names))

    # Mean flux per reaction pair for each variant (forward + reverse summed)
    variant_col = raw["variant"].to_numpy()
    unique_variants = sorted(raw["variant"].unique().to_list())
    pair_flux: dict[int, np.ndarray] = {}
    for variant in unique_variants:
        rxn_mean = flux_matrix[variant_col == variant, :].mean(axis=0)
        pair_flux[variant] = np.bincount(
            pair_of_rxn, weights=rxn_mean, minlength=n_pairs
        )

    # Shared log bins over the nonzero values of all variants
    nonzero = np.concatenate([f[f > zero_threshold] for f in pair_flux.values()])
    if bin_edges_param is not None:
        bin_edges = np.asarray(bin_edges_param, dtype=float)
        bin_edges = bin_edges[bin_edges > 0]
    elif nonzero.size:
        lo = np.floor(np.log10(nonzero.min()))
        hi = np.ceil(np.log10(nonzero.max()))
        bin_edges = np.logspace(lo, hi, n_bins + 1)
    else:
        bin_edges = np.array([])

    # Legend labels: variant, its diversity weight (if logged), its zero count
    div_weights = _diversity_weights(conn, config_sql)
    labels: dict[int, str] = {}
    hist_rows: list[dict] = []
    zero_rows: list[dict] = []
    for variant in unique_variants:
        flux = pair_flux[variant]
        is_zero = flux <= zero_threshold
        base = create_variant_label(variant, per_variant_params)
        base = " ".join(base) if isinstance(base, list) else base
        weight = div_weights.get(variant)
        weight_txt = f", λ_div = {weight:.1e}" if weight is not None else ""
        labels[variant] = f"{base}{weight_txt} (zero: {int(is_zero.sum())})"
        zero_rows.append(
            {
                "Variant": labels[variant],
                "Count": int(is_zero.sum()),
                "Bin": "zero flux",
            }
        )
        if bin_edges.size > 1:
            counts, _ = np.histogram(flux[~is_zero], bins=bin_edges)
            dropped = int((~is_zero).sum() - counts.sum())
            if dropped:
                print(
                    f"mean_flux_distribution: {labels[variant]}: {dropped} nonzero "
                    "pairs fall outside bin_edges and are not shown"
                )
            for i, count in enumerate(counts):
                hist_rows.append(
                    {
                        "Variant": labels[variant],
                        "bin_lo": float(bin_edges[i]),
                        "bin_hi": float(bin_edges[i + 1]),
                        "Count": int(count),
                    }
                )

    variant_sort = [labels[v] for v in ordered_variant_ids if v in labels] + [
        labels[v] for v in unique_variants if v not in ordered_variant_ids
    ]
    color = alt.Color("Variant:N", sort=variant_sort, title="Variant")
    y_title = "Reaction pairs"
    # Log y can't show 0: drop empty bins, and start bars at 0.5 (below the
    # smallest nonzero count) instead of 0 so they draw as solid bars.
    count_filter = "datum.Count > 0" if y_scale == "log" else "true"
    y_baseline = 0.5 if y_scale == "log" else 0.0
    max_count = max(row["Count"] for row in zero_rows + hist_rows)
    y_top = max_count * 2 if y_scale == "log" else max_count * 1.05
    y_enc = alt.Y(
        "Count:Q",
        stack=None,
        scale=alt.Scale(type=y_scale, domain=[y_baseline, y_top], nice=False),
        title=y_title,
    )
    y2_enc = alt.datum(y_baseline)

    zero_chart = (
        alt.Chart(pl.DataFrame(zero_rows).to_pandas())
        .transform_filter(count_filter)
        .mark_bar(opacity=BAR_OPACITY, strokeWidth=1)
        .encode(
            x=alt.X("Bin:N", title=None, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Variant:N", sort=variant_sort),
            y=y_enc,
            y2=y2_enc,
            color=color,
            stroke=alt.Stroke("Variant:N", sort=variant_sort, legend=None),
            tooltip=[
                alt.Tooltip("Variant:N"),
                alt.Tooltip("Count:Q", title="Zero-flux pairs"),
            ],
        )
        .properties(width=max(40, 18 * len(unique_variants)), height=subplot_h)
    )

    hist_df = pl.DataFrame(hist_rows).to_pandas() if hist_rows else None
    if hist_df is not None:
        hist_chart = (
            alt.Chart(hist_df)
            .transform_filter(count_filter)
            .mark_bar(opacity=BAR_OPACITY, strokeWidth=1)
            .encode(
                x=alt.X(
                    "bin_lo:Q",
                    scale=alt.Scale(type="log"),
                    title=f"Mean estimated flux per reaction pair ({unit_label})",
                    axis=alt.Axis(format=".0e"),
                ),
                x2="bin_hi:Q",
                y=y_enc,
                y2=y2_enc,
                color=color,
                stroke=alt.Stroke("Variant:N", sort=variant_sort, legend=None),
                tooltip=[
                    alt.Tooltip("Variant:N"),
                    alt.Tooltip("bin_lo:Q", title="From", format=".2e"),
                    alt.Tooltip("bin_hi:Q", title="To", format=".2e"),
                    alt.Tooltip("Count:Q", title="Pairs"),
                ],
            )
            .properties(width=subplot_w, height=subplot_h)
        )
        # Reference line at 1 molecule per step
        one_molecule = 1.0
        if in_molar:
            row = conn.sql(
                f"SELECT avg(counts_to_molar) FROM ({hist_subquery})"
            ).fetchone()
            one_molecule = float(cast(tuple, row)[0])
        rule = (
            alt.Chart(pl.DataFrame({"x": [one_molecule]}).to_pandas())
            .mark_rule(color="gray", strokeDash=[4, 3])
            .encode(x="x:Q")
        )
        chart = alt.hconcat(zero_chart, alt.layer(hist_chart, rule)).resolve_scale(
            y="shared"
        )
    else:
        chart = zero_chart

    chart = chart.properties(
        title=f"Distribution of mean fluxes across {n_pairs} reaction pairs, by variant"
    )
    out_path = os.path.join(outdir, "mean_flux_distribution.html")
    chart.save(out_path)
    print(f"Saved multivariant mean flux distribution to {out_path}")
