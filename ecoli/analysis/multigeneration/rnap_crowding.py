"""
Comparison of target vs actual transcription-unit synthesis probabilities
for TUs whose synthesis probabilities exceeded the limit set by the
physical size and the elongation rates of RNA polymerases.

Migrated from wcEcoli
``models/ecoli/analysis/multigen/rnap_crowding.py``.
"""

import os
import pickle
from typing import Any

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    open_arbitrary_sim_data,
)

MAX_NUMBER_OF_TUS_TO_PLOT = 300

# ----------------------------------------- #


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
    For every transcription unit flagged as ``tu_is_overcrowded`` at any
    point in the simulation, plot the target and actual synthesis
    probability traces over time.
    """

    # Load sim_data only to confirm we can access transcription tables if
    # ever needed; the TU display names come straight from the listener
    # field metadata, which matches the wcEcoli readAttribute("rnaIds").
    with open_arbitrary_sim_data(sim_data_dict) as f:
        _ = pickle.load(f)

    try:
        tu_ids = field_metadata(
            conn,
            config_sql,
            "listeners__rna_synth_prob__target_rna_synth_prob",
        )
    except Exception as e:
        print(f"[ERROR] Error getting field metadata: {e}")
        return

    # ---- Find overcrowded TU indices -------------------------------------------
    # A TU is "overcrowded" if tu_is_overcrowded was true at any timepoint.
    overcrowded_query = f"""
    WITH unnested AS (
        SELECT
            unnest(listeners__rna_synth_prob__tu_is_overcrowded)::INTEGER AS oc,
            generate_subscripts(listeners__rna_synth_prob__tu_is_overcrowded, 1) AS idx
        FROM ({history_sql})
    )
    SELECT idx
    FROM unnested
    GROUP BY idx
    HAVING sum(oc) > 0
    ORDER BY idx
    LIMIT {MAX_NUMBER_OF_TUS_TO_PLOT}
    """

    overcrowded_indices = [
        row[0] - 1 for row in conn.execute(overcrowded_query).fetchall()
    ]  # 1-based DuckDB → 0-based Python

    if not overcrowded_indices:
        print("[INFO] No overcrowded TUs found.")
        return

    n_overcrowded_tus = len(overcrowded_indices)
    n_overcrowded_tus_to_plot = min(n_overcrowded_tus, MAX_NUMBER_OF_TUS_TO_PLOT)

    print(f"[INFO] Found {n_overcrowded_tus} overcrowded TUs")

    # ---- Pull target/actual traces for those TUs -------------------------------
    actual_columns: list[str] = []
    target_columns: list[str] = []

    for i, idx in enumerate(overcrowded_indices):
        if i >= n_overcrowded_tus_to_plot:
            break
        if idx < len(tu_ids):
            # wcEcoli stripped a 3-char compartment tag ([c]); listener metadata
            # here uses the same convention, so do the same to keep the labels
            # tidy. Guard against ids that happen to be shorter.
            raw = tu_ids[idx]
            label = raw[:-3] if len(raw) > 3 else raw
            actual_columns.append(f"actual__{label}")
            target_columns.append(f"target__{label}")

    actual_expr = named_idx(
        "listeners__rna_synth_prob__actual_rna_synth_prob",
        actual_columns,
        [overcrowded_indices[: len(actual_columns)]],
    )
    target_expr = named_idx(
        "listeners__rna_synth_prob__target_rna_synth_prob",
        target_columns,
        [overcrowded_indices[: len(target_columns)]],
    )

    data_query = f"SELECT {actual_expr}, {target_expr}, time FROM ({history_sql})"
    df = conn.execute(data_query).fetchdf()

    # ---- Reshape for Altair ----------------------------------------------------
    pl_df = pl.DataFrame(df)

    prob_columns = actual_columns + target_columns
    plot_df = (
        pl_df.unpivot(
            index=["time"],
            on=prob_columns,
            variable_name="variable",
            value_name="Synthesis_Probability",
        )
        .with_columns(
            [
                pl.col("variable")
                .str.split_exact("__", 1)
                .struct.rename_fields(["Probability_Type", "TU_ID"]),
                (pl.col("time") / 60).alias("Time_min"),
            ]
        )
        .unnest("variable")
    )

    unique_tus = plot_df["TU_ID"].unique(maintain_order=True).to_list()

    # ---- One small-multiple per TU --------------------------------------------
    charts = []
    for i, tu_id in enumerate(unique_tus[:n_overcrowded_tus_to_plot]):
        tu_data = plot_df.filter(pl.col("TU_ID") == tu_id)
        if tu_data.height == 0:
            continue

        chart = (
            alt.Chart(tu_data)
            .mark_line(point=False, strokeWidth=2)
            .encode(
                x=alt.X("Time_min:Q", title="Time (min)", scale=alt.Scale(nice=True)),
                y=alt.Y(
                    "Synthesis_Probability:Q",
                    title=f"{tu_id} synthesis probability",
                    scale=alt.Scale(nice=True),
                ),
                color=alt.Color(
                    "Probability_Type:N",
                    scale=alt.Scale(
                        domain=["target", "actual"],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                    legend=alt.Legend(title="Type") if i == 0 else None,
                ),
                tooltip=[
                    alt.Tooltip("Time_min:Q", title="Time (min)", format=".2f"),
                    alt.Tooltip(
                        "Synthesis_Probability:Q",
                        title="Probability",
                        format=".4f",
                    ),
                    alt.Tooltip("Probability_Type:N", title="Type"),
                    alt.Tooltip("TU_ID:N", title="TU"),
                ],
            )
            .properties(
                width=600,
                height=150,
                title=alt.TitleParams(
                    text=[
                        f"TU {tu_id} - Synthesis Probability Comparison",
                        f"Total overcrowded TUs: {n_overcrowded_tus}"
                        + (
                            f" (showing first {MAX_NUMBER_OF_TUS_TO_PLOT})"
                            if n_overcrowded_tus > MAX_NUMBER_OF_TUS_TO_PLOT
                            else ""
                        )
                        if i == 0
                        else "",
                    ],
                    fontSize=12,
                    anchor="start",
                ),
            )
        )
        charts.append(chart)

    if not charts:
        print("[INFO] No charts created - no data to plot")
        return

    combined_chart = (
        alt.vconcat(*charts)
        .resolve_scale(color="independent")
        .add_params(alt.selection_interval(bind="scales"))
    )
    alt.data_transformers.enable("json")

    output_path = os.path.join(outdir, "rnap_crowding.html")
    combined_chart.save(output_path)
    print(f"[INFO] Generated rnap crowding plot for {len(charts)} overcrowded TUs")
    print(f"[INFO] Plot saved to: {output_path}")

    json_path = os.path.join(outdir, "rnap_crowding.json")
    combined_chart.save(json_path)
    print(f"[INFO] Chart specification saved to: {json_path}")
