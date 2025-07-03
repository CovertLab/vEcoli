import os
from typing import Any
import altair as alt
import pickle
import polars as pl
import numpy as np
from duckdb import DuckDBPyConnection
import pandas as pd

from ecoli.library.parquet_emitter import open_arbitrary_sim_data, named_idx
from ecoli.library.schema import bulk_name_to_idx


# ----------------------------------------- #


def calc_rna_doubling_time(
    produced_col: str, count_col: str, borderline: float
) -> pl.Expr:
    """
    Calculate rRNA doubling time with sanitation.
    """
    production_rate = pl.col(produced_col) / pl.col("time_step_sec")
    growth_rate = production_rate / pl.col(count_col)
    dt_min = float(np.log(2)) / growth_rate / 60
    valid = (
        (pl.col(produced_col) >= 0)
        & (pl.col(count_col) > 0)
        & (growth_rate > 0)
        & dt_min.is_finite()
        & (dt_min > 0)
        & (dt_min < 2 * borderline)
    )
    return pl.when(valid).then(dt_min).otherwise(None)


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
    """Visualize ribosome production metrics for E. coli simulation."""
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    sim_doubling_time = sim_data.doubling_time.asNumber()

    # define rRNA groups and bulk IDs
    s30_16s = list(sim_data.molecule_groups.s30_16s_rRNA) + [
        sim_data.molecule_ids.s30_full_complex
    ]
    s50_23s = list(sim_data.molecule_groups.s50_23s_rRNA) + [
        sim_data.molecule_ids.s50_full_complex
    ]
    s50_5s = list(sim_data.molecule_groups.s50_5s_rRNA) + [
        sim_data.molecule_ids.s50_full_complex
    ]
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()

    # precompute indices as Python ints
    idx_16s = [int(i) for i in np.atleast_1d(bulk_name_to_idx(s30_16s, bulk_ids))]
    idx_23s = [int(i) for i in np.atleast_1d(bulk_name_to_idx(s50_23s, bulk_ids))]
    idx_5s = [int(i) for i in np.atleast_1d(bulk_name_to_idx(s50_5s, bulk_ids))]

    required_columns = [
        "time",
        "variant",
        "generation",
        "agent_id",
        "listeners__mass__instantaneous_growth_rate",
        "listeners__mass__dry_mass",
        "listeners__ribosome_data__rRNA16S_initiated",
        "listeners__ribosome_data__rRNA23S_initiated",
        "listeners__ribosome_data__rRNA5S_initiated",
        "listeners__ribosome_data__rRNA16S_init_prob",
        "listeners__ribosome_data__rRNA23S_init_prob",
        "listeners__ribosome_data__rRNA5S_init_prob",
        "listeners__ribosome_data__effective_elongation_rate",
        "listeners__unique_molecule_counts__active_ribosome",
    ]

    # load data
    # Create the bulk index expressions
    bulk_16s_expr = named_idx("bulk", [f"bulk_{i}" for i in idx_16s], [idx_16s])
    bulk_23s_expr = named_idx("bulk", [f"bulk_{i}" for i in idx_23s], [idx_23s])
    bulk_5s_expr = named_idx("bulk", [f"bulk_{i}" for i in idx_5s], [idx_5s])

    # Combine all columns and expressions
    all_columns = ", ".join(required_columns)
    bulk_expressions = ", ".join([bulk_16s_expr, bulk_23s_expr, bulk_5s_expr])

    # Build the SQL query
    sql = f"""
    SELECT {all_columns}, {bulk_expressions}
    FROM ({history_sql})
    WHERE agent_id = 0
    ORDER BY generation, time
    """

    df = conn.sql(sql).pl()

    # time
    df = df.with_columns((pl.col("time") / 60).alias("time_min"))
    df = df.with_columns(
        pl.col("time")
        .diff()
        .over(["variant", "generation", "agent_id"])
        .alias("time_step_sec")
    )
    df = df.with_columns(
        time_step_sec=pl.when(pl.col("time_step_sec").is_null())
        .then(pl.col("time"))
        .otherwise(pl.col("time_step_sec"))
    )

    # cell doubling time
    if "listeners__mass__instantaneous_growth_rate" in df.columns:
        val = (
            float(np.log(2)) / pl.col("listeners__mass__instantaneous_growth_rate") / 60
        )
        df = df.with_columns(
            pl.when(val.is_between(0, 2 * sim_doubling_time, closed="both"))
            .then(val)
            .otherwise(None)
            .alias("cell_doubling_time_min")
        )

    df = df.with_columns(
        [
            pl.sum_horizontal([pl.col(f"bulk_{i}") for i in idx_16s]).alias(
                "bulk_16s_count"
            ),
            pl.sum_horizontal([pl.col(f"bulk_{i}") for i in idx_23s]).alias(
                "bulk_23s_count"
            ),
            pl.sum_horizontal([pl.col(f"bulk_{i}") for i in idx_5s]).alias(
                "bulk_5s_count"
            ),
            pl.col("listeners__unique_molecule_counts__active_ribosome")
            .fill_null(0)
            .alias("ribosome_count"),
        ]
    )

    # total rRNA
    df = df.with_columns(
        [
            (pl.col("bulk_16s_count") + pl.col("ribosome_count")).alias("rrn16s_count"),
            (pl.col("bulk_23s_count") + pl.col("ribosome_count")).alias("rrn23s_count"),
            (pl.col("bulk_5s_count") + pl.col("ribosome_count")).alias("rrn5s_count"),
        ]
    )

    # rRNA doubling times
    if "listeners__ribosome_data__rRNA16S_initiated" in df.columns:
        df = df.with_columns(
            rrn16S_doubling_time_min=calc_rna_doubling_time(
                "listeners__ribosome_data__rRNA16S_initiated",
                "rrn16s_count",
                sim_doubling_time,
            )
        )
    if "listeners__ribosome_data__rRNA23S_initiated" in df.columns:
        df = df.with_columns(
            rrn23S_doubling_time_min=calc_rna_doubling_time(
                "listeners__ribosome_data__rRNA23S_initiated",
                "rrn23s_count",
                sim_doubling_time,
            )
        )
    if "listeners__ribosome_data__rRNA5S_initiated" in df.columns:
        df = df.with_columns(
            rrn5S_doubling_time_min=calc_rna_doubling_time(
                "listeners__ribosome_data__rRNA5S_initiated",
                "rrn5s_count",
                sim_doubling_time,
            )
        )

    # reference probabilities
    cond = sim_data.condition
    trans = sim_data.process.transcription
    synth_probs = trans.cistron_tu_mapping_matrix.dot(trans.rna_synth_prob[cond])

    def fit_prob(group_ids):
        cistrons = [rid[:-3] for rid in group_ids]
        idxs = np.where(np.isin(trans.cistron_data["id"], cistrons))[0]
        return synth_probs[idxs].sum() if idxs.size else 0.0

    ref_probs = {
        "16S": fit_prob(sim_data.molecule_groups.s30_16s_rRNA),
        "23S": fit_prob(sim_data.molecule_groups.s50_23s_rRNA),
        "5S": fit_prob(sim_data.molecule_groups.s50_5s_rRNA),
    }

    # ----------------------------------------- #
    # prepare for plotting
    plot_cols = ["time_min", "variant", "generation"]

    for c in [
        "listeners__mass__dry_mass",
        "cell_doubling_time_min",
        "rrn16S_doubling_time_min",
        "rrn23S_doubling_time_min",
        "rrn5S_doubling_time_min",
        "rrn16S_init_prob",
        "rrn23S_init_prob",
        "rrn5S_init_prob",
        "listeners__ribosome_data__effective_elongation_rate",
    ]:
        if c in df.columns:
            plot_cols.append(c)

    plot_df = df.select(plot_cols)

    init_dm = (
        plot_df.filter(pl.col("time_min") == 0)
        .select(["variant", "listeners__mass__dry_mass"])
        .rename({"listeners__mass__dry_mass": "initial_dry_mass"})
    )
    plot_df = plot_df.join(init_dm, on=["variant"], how="left")
    plot_df = plot_df.with_columns(
        (pl.col("listeners__mass__dry_mass") / pl.col("initial_dry_mass")).alias(
            "dry_mass_normalized"
        )
    )

    # generate Altair charts
    def create_line_chart(y, title, y_title, ref=None):
        base = alt.Chart(plot_df)
        line = (
            base.mark_line()
            .encode(
                x=alt.X("time_min:Q", title="Time (min)"),
                y=alt.Y(f"{y}:Q", title=y_title),
                color=alt.Color(
                    "generation:N",
                    legend=alt.Legend(title="Simulated Multigeneration Data"),
                ),
            )
            .properties(title=title, width=600, height=120)
        )
        if ref is not None:
            rule = (
                alt.Chart(pd.DataFrame({"y": [ref]}))
                .mark_rule(color="red", strokeDash=[5, 5])
                .encode(y="y:Q")
            )
            return line + rule
        return line

    def create_histogram(
        col: str, title: str, bins: int = 30, probability: bool = False
    ) -> alt.Chart:
        if probability:
            density = (
                alt.Chart(plot_df)
                .transform_density(col, as_=[col, "density"], counts=False, steps=bins)
                .mark_area(opacity=0.6)
                .encode(
                    x=alt.X(f"{col}:Q", title=f"bin={bins}"),
                    y=alt.Y("density:Q", title="Density"),
                )
                .properties(width=200, height=120, title=title)
            )
            return density
        else:
            hist = (
                alt.Chart(plot_df)
                .mark_bar(opacity=0.6)
                .encode(
                    x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=bins), title=f"bin={bins}"),
                    y=alt.Y("count():Q", title="Count"),
                    color=alt.value("steelblue"),
                )
                .properties(width=200, height=120, title=title)
            )
            return hist

    plots = []
    # Dry mass
    if "dry_mass_normalized" in plot_df.columns:
        line = create_line_chart(
            "dry_mass_normalized",
            "Normalized Dry Mass Over Time",
            "Dry mass (relative to t=0)",
        )
        hist = create_histogram(
            "dry_mass_normalized", "Normalized Dry Mass Distribution", probability=True
        )
        plots.append(alt.hconcat(line, hist))
    # Cell Doubling Time
    if "cell_doubling_time_min" in plot_df.columns:
        line = create_line_chart(
            "cell_doubling_time_min",
            "Cell Doubling Time",
            "Doubling Time (min)",
            sim_doubling_time,
        )
        hist = create_histogram(
            "cell_doubling_time_min",
            "Cell Doubling Time (min) Distribution",
            probability=True,
        )
        plots.append(alt.hconcat(line, hist))
    # rRNA Doubl;ing Time
    for suffix in ["16S", "23S", "5S"]:
        col = f"rrn{suffix}_doubling_time_min"
        if col in plot_df.columns:
            line = create_line_chart(
                col,
                f"{suffix} rRNA Doubling Time",
                "Doubling Time (min)",
                sim_doubling_time,
            )
            hist = create_histogram(
                col, f"{suffix} rRNA Doubling Time Distribution", probability=True
            )
            plots.append(alt.hconcat(line, hist))
    # rRNA Initiation Probability
    for suffix, ref in ref_probs.items():
        col = f"rrn{suffix}_init_prob"
        if col in plot_df.columns:
            line = create_line_chart(
                col, f"{suffix} rRNA Initiation Probability", "Probability", ref
            )
            hist = create_histogram(
                col,
                f"{suffix} rRNA Initiation Probability Distribution",
                probability=True,
            )
            plots.append(alt.hconcat(line, hist))
    # Ribosome Elongation Rate
    if "listeners__ribosome_data__effective_elongation_rate" in plot_df.columns:
        line = create_line_chart(
            "listeners__ribosome_data__effective_elongation_rate",
            "Ribosome Elongation Rate",
            "Amino acids/s",
        )
        hist = create_histogram(
            "listeners__ribosome_data__effective_elongation_rate",
            "Ribosome Elongation Rate Distribution",
            probability=True,
        )
        plots.append(alt.hconcat(line, hist))

    if not plots:
        fallback = pl.DataFrame({"message": ["No data available"], "x": [0], "y": [0]})
        plots.append(
            alt.Chart(fallback)
            .mark_text(size=20, color="red")
            .encode(x="x:Q", y="y:Q", text="message:N")
            .properties(width=600, height=400, title="No Data")
        )

    combined = (
        alt.vconcat(*plots)
        .resolve_scale(x="shared", y="independent")
        .properties(title="Ribosome Production Metrics")
    )
    out_path = os.path.join(outdir, "ribosome_production_report.html")
    combined.save(out_path)
    print(f"Saved visualization to: {out_path}")
    return combined
