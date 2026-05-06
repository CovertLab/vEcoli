import altair as alt

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import polars as pl
from typing import Any

from ecoli.library.parquet_emitter import read_stacked_columns


def _variant_scale_frame(
    variant_metadata: dict[str, dict[int, Any]],
) -> pl.DataFrame:
    """Flatten ``variant_metadata`` into a (variant, scale) polars DataFrame."""
    rows: list[dict[str, Any]] = []
    for _experiment_id, per_variant in variant_metadata.items():
        for variant_idx, meta in per_variant.items():
            scale: Any = None
            if isinstance(meta, dict):
                if "scale" in meta:
                    scale = meta["scale"]
                elif "homeostatic_target_scale" in meta:
                    nested = meta["homeostatic_target_scale"]
                    if isinstance(nested, dict) and "scale" in nested:
                        scale = nested["scale"]
                    else:
                        scale = nested
            rows.append({"variant": int(variant_idx), "scale": scale})
    return pl.DataFrame(rows).unique(subset=["variant"], keep="first")


def _attach_variant_label(df: pl.DataFrame, scale_df: pl.DataFrame) -> pl.DataFrame:
    joined = df.join(scale_df, on="variant", how="left")
    label = (
        pl.when(pl.col("scale").is_null())
        .then(pl.col("variant").cast(pl.Utf8))
        .otherwise(
            pl.col("variant").cast(pl.Utf8)
            + pl.lit(" (scale=")
            + pl.col("scale").cast(pl.Utf8)
            + pl.lit(")")
        )
        .alias("variant_label")
    )
    return joined.with_columns(label)


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
    """
    Line plot of doubling time vs generation for each lineage seed, plus a
    box plot of the doubling-time distribution per variant for generations
    >= 6. Legend labels include the homeostatic_target_scale value when
    available. Only works for lineage simulations with ``single_daughters``
    set to True.
    """
    doubling_time_sql = read_stacked_columns(
        history_sql,
        ["time"],
        order_results=False,
    )
    doubling_times = conn.sql(f"""
        SELECT (max(time) - min(time)) / 60 AS 'Doubling Time (min)', experiment_id, variant, lineage_seed, generation, agent_id
        FROM ({doubling_time_sql})
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
    """).pl()
    successful_sims = conn.sql(success_sql).pl()
    death_times = doubling_times.join(
        successful_sims,
        how="anti",
        on=["experiment_id", "variant", "lineage_seed", "agent_id"],
    )
    doubling_times = doubling_times.join(
        successful_sims,
        how="semi",
        on=["experiment_id", "variant", "lineage_seed", "agent_id"],
    )

    scale_df = _variant_scale_frame(variant_metadata)
    doubling_times = _attach_variant_label(doubling_times, scale_df)
    death_times = _attach_variant_label(death_times, scale_df)

    selection = alt.selection_point(fields=["variant_label"], bind="legend")

    chart = (
        alt.Chart(doubling_times)
        .mark_line()
        .encode(
            x="generation",
            y="Doubling Time (min)",
            color=alt.Color("variant_label:N", legend=alt.Legend(title="Variant")),
            detail="lineage_seed:N",
            tooltip=["Doubling Time (min)", "lineage_seed", "variant_label"],
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
        )
        .add_params(selection)
        .interactive()
    )

    death_points = (
        alt.Chart(death_times)
        .mark_point(shape="cross")
        .encode(
            x="generation",
            y="Doubling Time (min)",
            color=alt.Color("variant_label:N", legend=alt.Legend(title="Variant")),
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
            tooltip=["Doubling Time (min)", "lineage_seed", "variant_label"],
        )
    )

    exp_avg = alt.Chart().mark_rule(strokeDash=[2, 2]).encode(y=alt.datum(60 / 0.47))

    sim_avg_df = doubling_times.group_by(
        "experiment_id", "variant_label", "generation"
    ).agg(pl.mean("Doubling Time (min)"))
    sim_avg = (
        alt.Chart(sim_avg_df)
        .mark_line(strokeDash=[2, 2], strokeWidth=3)
        .encode(
            x="generation",
            y="Doubling Time (min)",
            color=alt.Color("variant_label:N", legend=alt.Legend(title="Variant")),
            tooltip=["Doubling Time (min)", "variant_label"],
        )
    )

    line_chart = chart + exp_avg + sim_avg + death_points

    box_df = doubling_times.filter(pl.col("generation") >= 6)
    box_chart = (
        alt.Chart(box_df)
        .mark_boxplot()
        .encode(
            x=alt.X("variant_label:N", title="Variant"),
            y=alt.Y("Doubling Time (min):Q"),
            color=alt.Color("variant_label:N", legend=alt.Legend(title="Variant")),
        )
        .properties(title="Doubling time distribution (generations >= 6)")
    )

    combined = alt.vconcat(line_chart, box_chart)
    combined.save(f"{outdir}/doubling_time.html")
