"""
Scatter plots of mean homeostatic target dmdt_conc vs. mean estimated
dmdt_conc per metabolite, averaged over late generations (>= 6) for each
variant. Produces one chart per variant plus a combined chart with all
variants colored together.
"""

import os
from typing import Any

import altair as alt

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl

from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns


LATE_GEN_THRESHOLD = 6
TARGET_COL = "listeners__fba_results__target_homeostatic_dmdt_conc"
ESTIMATED_COL = "listeners__fba_results__estimated_homeostatic_dmdt_conc"


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    os.makedirs(outdir, exist_ok=True)

    metabolite_names = field_metadata(conn, config_sql, TARGET_COL)
    n_mets = len(metabolite_names)

    sub = read_stacked_columns(
        history_sql,
        [TARGET_COL, ESTIMATED_COL],
        order_results=False,
    )
    df = conn.sql(f"""
        SELECT variant,
               {TARGET_COL} AS target,
               {ESTIMATED_COL} AS estimated
        FROM ({sub})
        WHERE generation >= {LATE_GEN_THRESHOLD}
          AND len({TARGET_COL}) = {n_mets}
          AND len({ESTIMATED_COL}) = {n_mets}
    """).pl()

    if df.height == 0:
        print(
            f"[homeostatic_target_vs_estimated] No rows with generation >= "
            f"{LATE_GEN_THRESHOLD} and populated arrays; aborting."
        )
        return

    rows: list[dict[str, Any]] = []
    for variant in sorted(df["variant"].unique().to_list()):
        sub_df = df.filter(pl.col("variant") == variant)
        targets = np.asarray(sub_df["target"].to_list(), dtype=float)
        estimates = np.asarray(sub_df["estimated"].to_list(), dtype=float)
        target_mean = targets.mean(axis=0)
        est_mean = estimates.mean(axis=0)
        for i, met in enumerate(metabolite_names):
            rows.append(
                {
                    "variant": int(variant),
                    "metabolite": met,
                    "target_mean": float(target_mean[i]),
                    "estimated_mean": float(est_mean[i]),
                }
            )

    out_df = pl.DataFrame(rows)
    csv_path = os.path.join(outdir, "homeostatic_target_vs_estimated.csv")
    out_df.write_csv(csv_path)
    print(f"Wrote {csv_path}")

    # dmdt_conc can be negative; log scale needs positive values, so plot
    # |target| vs |estimated| and surface the sign in tooltip / color.
    plot_df = out_df.with_columns(
        [
            pl.col("variant").cast(pl.Utf8).alias("variant_str"),
            pl.col("target_mean").abs().alias("abs_target"),
            pl.col("estimated_mean").abs().alias("abs_estimated"),
            pl.when(
                (pl.col("target_mean").sign() == pl.col("estimated_mean").sign())
                | (pl.col("target_mean") == 0)
                | (pl.col("estimated_mean") == 0)
            )
            .then(pl.lit("same/zero"))
            .otherwise(pl.lit("opposite"))
            .alias("sign_match"),
        ]
    ).filter((pl.col("abs_target") > 0) & (pl.col("abs_estimated") > 0))

    # Shared y=x reference line spanning the positive range
    pos_vals = pl.concat(
        [
            plot_df.select("abs_target").rename({"abs_target": "v"}),
            plot_df.select("abs_estimated").rename({"abs_estimated": "v"}),
        ]
    )
    lo = float(pos_vals["v"].min())
    hi = float(pos_vals["v"].max())
    diag = pl.DataFrame({"v": [lo, hi]})

    base_diag = (
        alt.Chart(diag)
        .mark_line(strokeDash=[2, 2], color="gray")
        .encode(
            x=alt.X("v:Q", scale=alt.Scale(type="log"), title=None),
            y=alt.Y("v:Q", scale=alt.Scale(type="log"), title=None),
        )
    )

    combined = (
        alt.Chart(plot_df)
        .mark_circle(size=40, opacity=0.6)
        .encode(
            x=alt.X(
                "abs_target:Q",
                scale=alt.Scale(type="log"),
                title="|Mean target dmdt_conc| (log)",
            ),
            y=alt.Y(
                "abs_estimated:Q",
                scale=alt.Scale(type="log"),
                title="|Mean estimated dmdt_conc| (log)",
            ),
            color=alt.Color("variant_str:N", legend=alt.Legend(title="Variant")),
            tooltip=[
                "variant",
                "metabolite",
                "target_mean",
                "estimated_mean",
                "sign_match",
            ],
        )
        .properties(
            title=f"Target vs estimated homeostatic dmdt_conc (gen >= {LATE_GEN_THRESHOLD}, log-log)",
            width=500,
            height=500,
        )
    )
    combined_chart = (combined + base_diag).interactive()
    combined_path = f"{outdir}/homeostatic_target_vs_estimated_combined.html"
    combined_chart.save(combined_path)
    print(f"Wrote {combined_path}")

    per_variant_charts = []
    for variant in sorted(plot_df["variant"].unique().to_list()):
        v_df = plot_df.filter(pl.col("variant") == variant)
        scatter = (
            alt.Chart(v_df)
            .mark_circle(size=40, opacity=0.7)
            .encode(
                x=alt.X(
                    "abs_target:Q",
                    scale=alt.Scale(type="log"),
                    title="|Mean target dmdt_conc| (log)",
                ),
                y=alt.Y(
                    "abs_estimated:Q",
                    scale=alt.Scale(type="log"),
                    title="|Mean estimated dmdt_conc| (log)",
                ),
                color=alt.Color("variant_str:N", legend=None),
                tooltip=[
                    "metabolite",
                    "target_mean",
                    "estimated_mean",
                    "sign_match",
                ],
            )
            .properties(title=f"Variant {variant}", width=300, height=300)
        )
        per_variant_charts.append((scatter + base_diag).interactive())

    if per_variant_charts:
        # Two-column grid
        rows_of_charts = []
        for i in range(0, len(per_variant_charts), 2):
            rows_of_charts.append(alt.hconcat(*per_variant_charts[i : i + 2]))
        grid = alt.vconcat(*rows_of_charts).properties(
            title=f"Per-variant target vs estimated (gen >= {LATE_GEN_THRESHOLD})"
        )
        per_path = f"{outdir}/homeostatic_target_vs_estimated_per_variant.html"
        grid.save(per_path)
        print(f"Wrote {per_path}")
