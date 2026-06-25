"""
Validate simulated growth rate and grow/no-growth predictions against
experimentally measured values per condition, for multivariant "condition"
sweeps (``ecoli/variants/condition.py``).

Three figures are saved:

- A scatter of simulated vs. experimental growth rate, one dot per variant,
  with a y=x reference line and R^2. Experimental growth rate comes from
  ``validation/ecoli/flat/carbon_source_growth_rates.tsv``, joined on
  condition name via its "carbon source" column — variants whose condition
  has no matching row are excluded from this figure only.
- A grow/no-growth confusion matrix, one decision per condition.
- A categorical dot plot of simulated doubling time, grouped by experimental
  grow/no-growth category.

Experimental doubling times (used for the no-growth threshold in the latter
two figures) come from ``sim_data.condition_to_doubling_time`` (built from
``reconstruction/ecoli/flat/condition/condition_defs.tsv``). Today, every
condition in that file is experimentally growth-permitting, so the "no
growth" experimental category and the confusion matrix's negative-class
cells will be empty until conditions with documented non-growth are added —
the script handles that generically rather than assuming exactly two
categories.
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Any, TYPE_CHECKING

import altair as alt
import numpy as np
import plotly.express as px
import polars as pl

from ecoli.analysis.multivariant.utils import create_variant_label
from ecoli.library.parquet_emitter import open_arbitrary_sim_data, read_stacked_columns
from wholecell.utils import units

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

alt.data_transformers.enable("vegafusion")

DEFAULT_NO_GROWTH_MULTIPLIER = 3.0
CARBON_SOURCE_GROWTH_TSV = "validation/ecoli/flat/carbon_source_growth_rates.tsv"
PASTEL = px.colors.qualitative.Pastel


def _fmt_pct(x: float) -> str:
    return "N/A" if math.isnan(x) else f"{x:.0%}"


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
    """
    All options have default values (do not need to be explicitly provided).

    Args:
        params: Dictionary of parameters given under analysis
            name in configuration JSON. Config options look like this:

            .. code-block:: json

                {
                    // A cell is classified "predicted no growth" if its
                    // doubling time exceeds this multiplier times the
                    // experimentally expected doubling time for its
                    // condition.
                    "no_growth_doubling_time_multiplier": 3.0
                }
    """
    experiment_id = next(iter(variant_metadata.keys()), None)
    per_variant_params: dict[int, Any] = (
        variant_metadata[experiment_id] if experiment_id else {}
    )

    no_growth_multiplier = float(
        params.get("no_growth_doubling_time_multiplier", DEFAULT_NO_GROWTH_MULTIPLIER)
    )

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Resolve each variant's condition name. The baseline variant has no
    # "condition" param (per_variant_params[variant_id] == "baseline"), and
    # runs whatever sim_data.condition defaults to.
    variant_condition: dict[int, str] = {}
    for variant_id, var_params in per_variant_params.items():
        if var_params == "baseline":
            variant_condition[variant_id] = "basal"
        else:
            variant_condition[variant_id] = var_params["condition"]

    exp_doubling_time_min = {
        condition: dt.asNumber(units.min)
        for condition, dt in sim_data.condition_to_doubling_time.items()
    }

    # Literature-compiled experimental growth rates, keyed by carbon source
    # name. Only rows whose "carbon source" matches a condition name (e.g.
    # "sole_arabinose") are usable here — see module docstring.
    carbon_source_growth_df = pl.read_csv(
        CARBON_SOURCE_GROWTH_TSV, separator="\t", comment_prefix="#", quote_char='"'
    )
    carbon_source_growth_rate_per_min = {
        row["carbon source"]: row["growth rate (1/units.h)"] / 60
        for row in carbon_source_growth_df.iter_rows(named=True)
    }

    # ── Per-cell doubling time (also used for the no-growth threshold) ─────
    doubling_time_sql = read_stacked_columns(
        history_sql,
        ["time"],
        order_results=False,
        success_sql=success_sql,
    )
    doubling_times = conn.sql(
        f"""
        SELECT (max(time) - min(time)) / 60 AS doubling_time_min,
            variant, lineage_seed, generation, agent_id
        FROM ({doubling_time_sql})
        GROUP BY variant, lineage_seed, generation, agent_id
        """
    ).pl()

    # ── Mean simulated instantaneous growth rate, per variant ──────────────
    growth_rate_sql = read_stacked_columns(
        history_sql,
        ["listeners__mass__instantaneous_growth_rate AS growth_rate"],
        order_results=False,
        success_sql=success_sql,
    )
    growth_rates = conn.sql(
        f"""
        SELECT avg(growth_rate) * 60 AS sim_growth_rate_per_min, variant
        FROM ({growth_rate_sql})
        GROUP BY variant
        """
    ).pl()

    if doubling_times.is_empty() or growth_rates.is_empty():
        print("growth_rate_validation: no rows returned; skipping.")
        return

    doubling_times = doubling_times.with_columns(
        pl.col("variant")
        .replace_strict(variant_condition, return_dtype=pl.Utf8)
        .alias("condition")
    )
    doubling_times = doubling_times.with_columns(
        pl.col("condition")
        .replace_strict(exp_doubling_time_min, return_dtype=pl.Float64)
        .alias("exp_doubling_time_min")
    )

    # Detect incomplete sims: fewer generations than the baseline (basal)
    # variant for the same lineage_seed.  Falls back gracefully if no baseline
    # variant is present (e.g., non-condition sweeps).
    baseline_variant_ids = [v for v, p in per_variant_params.items() if p == "baseline"]
    if baseline_variant_ids:
        baseline_max_gen = (
            doubling_times.filter(pl.col("variant").is_in(baseline_variant_ids))
            .group_by("lineage_seed")
            .agg(pl.col("generation").max().alias("baseline_max_gen"))
        )
        variant_max_gen = doubling_times.group_by(["variant", "lineage_seed"]).agg(
            pl.col("generation").max().alias("variant_max_gen")
        )
        doubling_times = (
            doubling_times.join(variant_max_gen, on=["variant", "lineage_seed"])
            .join(baseline_max_gen, on="lineage_seed", how="left")
            .with_columns(
                (pl.col("variant_max_gen") < pl.col("baseline_max_gen"))
                .fill_null(False)
                .alias("incomplete_sim")
            )
            .drop(["variant_max_gen", "baseline_max_gen"])
        )
    else:
        print(
            "growth_rate_validation: no baseline variant found in "
            "per_variant_params; skipping incomplete-sim detection."
        )
        doubling_times = doubling_times.with_columns(
            pl.lit(False).alias("incomplete_sim")
        )

    doubling_times = doubling_times.with_columns(
        (
            (
                pl.col("doubling_time_min")
                > no_growth_multiplier * pl.col("exp_doubling_time_min")
            )
            | pl.col("incomplete_sim")
        ).alias("predicted_no_growth")
    )
    doubling_times = doubling_times.with_columns(
        pl.when(pl.col("predicted_no_growth"))
        .then(pl.lit("No growth"))
        .otherwise(pl.lit("Grows"))
        .alias("predicted_label"),
        # Every condition_defs.tsv condition is experimentally
        # growth-permitting today; this is computed generically (not
        # hardcoded to "Grows" alone) so it scales once non-growth
        # conditions exist.
        pl.lit("Grows").alias("experimental_label"),
    )

    growth_rates = growth_rates.with_columns(
        pl.col("variant")
        .replace_strict(variant_condition, return_dtype=pl.Utf8)
        .alias("condition")
    )
    growth_rates = growth_rates.with_columns(
        pl.col("condition")
        .replace_strict(
            carbon_source_growth_rate_per_min, default=None, return_dtype=pl.Float64
        )
        .alias("exp_growth_rate_per_min")
    )
    variant_labels: dict[int, str] = {}
    for variant_val in sorted(growth_rates["variant"].unique().to_list()):
        label_l = create_variant_label(variant_val, per_variant_params)
        variant_labels[variant_val] = (
            " ".join(label_l) if isinstance(label_l, list) else label_l
        )
    growth_rates = growth_rates.with_columns(
        pl.col("variant")
        .replace_strict(variant_labels, return_dtype=pl.Utf8)
        .alias("variant_label")
    )

    # Shared color scale (by variant) across Figures 1 and 3, so a given
    # variant keeps the same color in both.
    variant_label_list = [variant_labels[v] for v in sorted(variant_labels.keys())]
    variant_color_scale = alt.Scale(
        domain=variant_label_list, range=PASTEL[: len(variant_label_list)]
    )

    # ── Figure 1: growth rate scatter ───────────────────────────────────────
    # Only variants whose condition has a matching carbon_source_growth_rates.tsv
    # row can be plotted against an experimental value; this is routine
    # (literature coverage is partial by design), not an error.
    unmatched = growth_rates.filter(pl.col("exp_growth_rate_per_min").is_null())
    if not unmatched.is_empty():
        print(
            "growth_rate_validation: no carbon_source_growth_rates.tsv entry for "
            f"conditions {sorted(unmatched['condition'].unique().to_list())}; "
            "excluding from growth rate scatter."
        )
    growth_rates_matched = growth_rates.filter(
        pl.col("exp_growth_rate_per_min").is_not_null()
    )

    if growth_rates_matched.is_empty():
        print(
            "growth_rate_validation: no variants matched "
            "carbon_source_growth_rates.tsv; skipping growth rate scatter."
        )
    else:
        scatter_pdf = growth_rates_matched.to_pandas()
        sim_vals = scatter_pdf["sim_growth_rate_per_min"].to_numpy()
        exp_vals = scatter_pdf["exp_growth_rate_per_min"].to_numpy()
        ss_res = np.sum((sim_vals - exp_vals) ** 2)
        ss_tot = np.sum((exp_vals - np.mean(exp_vals)) ** 2)
        r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        axis_lo = float(min(sim_vals.min(), exp_vals.min()) * 0.9)
        axis_hi = float(max(sim_vals.max(), exp_vals.max()) * 1.1)
        diag_pdf = pl.DataFrame({"x": [axis_lo, axis_hi]}).to_pandas()

        points = (
            alt.Chart(scatter_pdf)
            .mark_point(size=80, filled=True)
            .encode(
                x=alt.X(
                    "exp_growth_rate_per_min:Q",
                    title="Experimental growth rate (1/min)",
                ),
                y=alt.Y(
                    "sim_growth_rate_per_min:Q", title="Simulated growth rate (1/min)"
                ),
                color=alt.Color(
                    "variant_label:N", title="Variant", scale=variant_color_scale
                ),
                tooltip=[
                    alt.Tooltip("variant_label:N", title="Variant"),
                    alt.Tooltip("condition:N", title="Condition"),
                    alt.Tooltip("exp_growth_rate_per_min:Q", title="Experimental"),
                    alt.Tooltip("sim_growth_rate_per_min:Q", title="Simulated"),
                ],
            )
        )
        diagonal = (
            alt.Chart(diag_pdf)
            .mark_line(color="gray", strokeDash=[4, 4])
            .encode(x="x:Q", y="x:Q")
        )
        annotation = (
            alt.Chart(pl.DataFrame({"r_squared": [r_squared]}).to_pandas())
            .mark_text(align="left", dx=5, dy=5, fontSize=12)
            .encode(x=alt.value(0), y=alt.value(0))
            .transform_calculate(label="'R² = ' + format(datum.r_squared, '.2f')")
            .encode(text="label:N")
        )
        scatter_chart = alt.layer(diagonal, points, annotation).properties(
            width=400, height=400, title="Simulated vs Experimental Growth Rate"
        )
        scatter_out = os.path.join(outdir, "growth_rate_scatter.html")
        scatter_chart.save(scatter_out)
        print(f"Saved growth rate scatter to {scatter_out}")

    # ── Figure 2: grow/no-growth confusion matrix ───────────────────────────
    condition_decision = doubling_times.group_by("condition").agg(
        pl.col("predicted_no_growth").mean().alias("frac_no_growth"),
        pl.col("experimental_label").first().alias("experimental_label"),
    )
    condition_decision = condition_decision.with_columns(
        pl.when(pl.col("frac_no_growth") >= 0.5)
        .then(pl.lit("No growth"))
        .otherwise(pl.lit("Grows"))
        .alias("predicted_label")
    )

    n_conditions = condition_decision.height
    tp = condition_decision.filter(
        (pl.col("experimental_label") == "Grows")
        & (pl.col("predicted_label") == "Grows")
    ).height
    fn = condition_decision.filter(
        (pl.col("experimental_label") == "Grows")
        & (pl.col("predicted_label") == "No growth")
    ).height
    fp = condition_decision.filter(
        (pl.col("experimental_label") == "No growth")
        & (pl.col("predicted_label") == "Grows")
    ).height
    tn = condition_decision.filter(
        (pl.col("experimental_label") == "No growth")
        & (pl.col("predicted_label") == "No growth")
    ).height

    accuracy = (tp + tn) / n_conditions if n_conditions > 0 else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    matrix_pdf = pl.DataFrame(
        [
            {
                "predicted": "Predicted: grows",
                "experimental": "Experimental: grows",
                "count": tp,
                "label": "true positive",
            },
            {
                "predicted": "Predicted: grows",
                "experimental": "Experimental: no growth",
                "count": fp,
                "label": "false positive",
            },
            {
                "predicted": "Predicted: no growth",
                "experimental": "Experimental: grows",
                "count": fn,
                "label": "false negative",
            },
            {
                "predicted": "Predicted: no growth",
                "experimental": "Experimental: no growth",
                "count": tn,
                "label": "true negative",
            },
        ]
    ).to_pandas()

    cells = (
        alt.Chart(matrix_pdf)
        .mark_rect()
        .encode(
            x=alt.X(
                "experimental:N",
                sort=["Experimental: grows", "Experimental: no growth"],
                title=None,
            ),
            y=alt.Y(
                "predicted:N",
                sort=["Predicted: grows", "Predicted: no growth"],
                title=None,
            ),
            color=alt.Color(
                "label:N",
                legend=None,
                scale=alt.Scale(
                    domain=[
                        "true positive",
                        "true negative",
                        "false positive",
                        "false negative",
                    ],
                    range=["#d9f0e3", "#d9f0e3", "#fde0dc", "#fde0dc"],
                ),
            ),
        )
    )
    count_text = cells.mark_text(fontSize=28, fontWeight="bold", dy=-8).encode(
        text="count:Q", color=alt.value("black")
    )
    label_text = cells.mark_text(fontSize=12, dy=16).encode(
        text="label:N", color=alt.value("black")
    )

    stats_pdf = pl.DataFrame(
        {
            "metric": ["Accuracy", "Sensitivity", "Specificity", "n conditions"],
            "value": [
                _fmt_pct(accuracy),
                _fmt_pct(sensitivity),
                _fmt_pct(specificity),
                str(n_conditions),
            ],
        }
    ).to_pandas()
    stats_text = (
        alt.Chart(stats_pdf)
        .mark_text(fontSize=14, fontWeight="bold", dy=20)
        .encode(
            x=alt.X("metric:N", title=None, axis=alt.Axis(labelFontSize=11)),
            text="value:N",
        )
        .properties(width=320, height=50)
    )

    confusion_chart = alt.vconcat(
        alt.layer(cells, count_text, label_text).properties(
            width=320, height=320, title="Predicted vs Experimental Growth"
        ),
        stats_text,
    )
    confusion_out = os.path.join(outdir, "growth_confusion_matrix.html")
    confusion_chart.save(confusion_out)
    print(f"Saved growth confusion matrix to {confusion_out}")

    # ── Figure 3: categorical dot plot ──────────────────────────────────────
    # One dot per variant: average doubling time across generations/seeds,
    # then re-derive the no-growth call from that average (rather than
    # averaging the old per-cell boolean).
    variant_doubling = doubling_times.group_by(
        ["variant", "condition", "exp_doubling_time_min", "experimental_label"]
    ).agg(
        pl.col("doubling_time_min").mean().alias("doubling_time_min"),
        pl.col("incomplete_sim").any().alias("incomplete_sim"),
    )
    variant_doubling = variant_doubling.with_columns(
        (
            (
                pl.col("doubling_time_min")
                > no_growth_multiplier * pl.col("exp_doubling_time_min")
            )
            | pl.col("incomplete_sim")
        ).alias("predicted_no_growth")
    )
    variant_doubling = variant_doubling.with_columns(
        pl.when(pl.col("predicted_no_growth"))
        .then(pl.lit("No growth"))
        .otherwise(pl.lit("Grows"))
        .alias("predicted_label"),
        pl.col("variant")
        .replace_strict(variant_labels, return_dtype=pl.Utf8)
        .alias("variant_label"),
    )

    dot_pdf = variant_doubling.to_pandas()
    dot_chart = (
        alt.Chart(dot_pdf)
        .mark_circle(size=120, opacity=0.85)
        .encode(
            x=alt.X("predicted_label:N", title=None),
            y=alt.Y("doubling_time_min:Q", title="Simulated doubling time (min)"),
            color=alt.Color(
                "variant_label:N", title="Variant", scale=variant_color_scale
            ),
            tooltip=[
                alt.Tooltip("variant_label:N", title="Variant"),
                alt.Tooltip("condition:N", title="Condition"),
                alt.Tooltip("doubling_time_min:Q", title="Doubling time (min)"),
                alt.Tooltip("predicted_label:N", title="Predicted"),
            ],
        )
        .properties(
            width=400,
            height=400,
            title="Simulated Doubling Time by Experimental Category (mean per variant)",
        )
    )
    dot_out = os.path.join(outdir, "growth_no_growth_dotplot.html")
    dot_chart.save(dot_out)
    print(f"Saved growth no-growth dot plot to {dot_out}")
