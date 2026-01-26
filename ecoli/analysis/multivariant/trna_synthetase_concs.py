import pickle
import warnings
from datetime import datetime, timezone
from io import StringIO
from typing import Any

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
from scipy.constants import N_A
import polars as pl

from ecoli.library.parquet_emitter import (
    read_stacked_columns,
    field_metadata,
    skip_n_gens,
)
from reconstruction.ecoli.fit_sim_data_1 import SimulationDataEcoli


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
    Analysis script for calculating the mean, standard deviation, and minimum values
    of tRNA synthetase concentrations for a set of baseline simulations (no growth
    rate control or kinetic tRNA charging) in minimal and rich media. The minimal
    workflow required to generate this data is ``configs/no_grc.json``. The output
    of this analysis should be used to replace
    ``reconstruction/ecoli/flat/optimization/trna_synthetase_dynamic_range.tsv``
    every time changes are made that could affect tRNA synthetase concentrations.

    Args:
        params: Dictionary of parameters that can include the following keys::

            {
                "skip_gens": int  # Number of initial generations to skip
            }

    """
    # Skip n generations if provided
    skip_gens = params.get("skip_gens")
    if skip_gens is not None:
        history_sql = skip_n_gens(history_sql, skip_gens)
    # Figure out which variant(s) are minimal and which are rich media
    experiment_ids = list(variant_names.keys())
    assert len(experiment_ids) == 1, (
        "This analysis script is only intended to be run with one experiment ID "
        "at a time."
    )
    experiment_id = experiment_ids[0]
    variant_name = variant_names[experiment_id]
    assert variant_name == "condition", (
        "This analysis script is only intended to be run with the 'condition' variant."
    )
    variant_id_data: dict[str, Any] = {
        "basal": {
            "variant_ids": [],
            "bulk_ids": [],
            "synthetase_ids": [],
            "synthetase_idx": [],
            "summary_stats": [],
        },
        "with_aa": {
            "variant_ids": [],
            "bulk_ids": [],
            "synthetase_ids": [],
            "synthetase_idx": [],
            "summary_stats": [],
        },
    }
    for variant_id, variant_params in variant_metadata[experiment_id].items():
        condition_label: str = ""
        if isinstance(variant_params, str):
            if variant_params != "baseline":
                warnings.warn(
                    f"Ignoring variant {variant_id} with unexpected label: {variant_params}"
                )
                continue
            condition_label = "basal"
        elif isinstance(variant_params, dict):
            condition_label = variant_params.get("condition", "")
        else:
            warnings.warn(
                f"Ignoring variant {variant_id} with unsupported metadata type: {type(variant_params)}"
            )
            continue

        media_info = variant_id_data[condition_label]
        if variant_id not in media_info["variant_ids"]:
            media_info["variant_ids"].append(variant_id)

    for condition_label, media_info in variant_id_data.items():
        if len(media_info["variant_ids"]) == 0:
            raise ValueError(f"No variants found for media type: {condition_label}")
        ordered_bulk_ids: list[str] | None = None
        trna_synthetase_ids: list[str] | None = None
        for variant_id in media_info["variant_ids"]:
            var_subquery = f"FROM ({config_sql}) WHERE variant = {variant_id}"
            bulk_ids = field_metadata(conn, var_subquery, "bulk")
            if ordered_bulk_ids is None:
                ordered_bulk_ids = bulk_ids
            elif bulk_ids != ordered_bulk_ids:
                raise ValueError(
                    f"Bulk ID order mismatch for media type '{condition_label}'. Ensure all "
                    "variants list bulk IDs in the same order."
                )

            sim_data_path = sim_data_dict[experiment_id].get(variant_id)
            if sim_data_path is None:
                raise ValueError(
                    f"Missing sim_data path for experiment {experiment_id}, variant {variant_id}."
                )
            with open(sim_data_path, "rb") as f:
                sim_data: SimulationDataEcoli = pickle.load(f)
            variant_trna_synthetase_ids = list(
                sim_data.process.transcription.synthetase_names
            )
            if trna_synthetase_ids is None:
                trna_synthetase_ids = variant_trna_synthetase_ids
            elif variant_trna_synthetase_ids != trna_synthetase_ids:
                raise ValueError(
                    "tRNA synthetase ID mismatch between variants. Ensure all variants "
                    "have the same set of tRNA synthetases."
                )

        if ordered_bulk_ids is None or trna_synthetase_ids is None:
            raise ValueError(
                f"Unable to determine metadata for media type '{condition_label}'."
            )
        media_info["synthetase_ids"] = trna_synthetase_ids
        media_info["bulk_ids"] = ordered_bulk_ids

    # Get indices of tRNA synthetases in bulk IDs
    # and read concentration summary stats
    for condition_label, media_info in variant_id_data.items():
        synthetase_ids = media_info.get("synthetase_ids")
        bulk_ids = media_info.get("bulk_ids")
        if synthetase_ids is None or bulk_ids is None:
            raise ValueError(
                f"Missing bulk or synthetase metadata for media type '{condition_label}'."
            )
        media_info["synthetase_idx"] = []
        for synthetase_id in synthetase_ids:
            try:
                synthetase_idx = bulk_ids.index(synthetase_id)
            except ValueError:
                raise ValueError(
                    f"tRNA synthetase ID '{synthetase_id}' not found in bulk IDs for "
                    f"media type '{condition_label}'."
                )
            media_info["synthetase_idx"].append(synthetase_idx + 1)  # 1-based indexing
        subquery = read_stacked_columns(
            history_sql,
            ["bulk", "listeners__mass__volume AS volume"],
            order_results=False,
        )
        # Scale concs from counts / fL to umol / L
        # 1e15 fL / L * 1 mol / N_A counts * 1e6 umol / mol
        scale_factor = 1e15 / N_A * 1e6
        # Read each variant ID individually as DuckDB does not support filter
        # pushdown for IN statements (e.g. WHERE variant IN (...))
        for variant_id in media_info["variant_ids"]:
            variant_subquery = f"FROM ({subquery}) WHERE variant = {variant_id}"
            trna_synthetase_stats = conn.sql(f"""
                WITH synthetase_counts AS (
                    SELECT list_select(bulk, {media_info["synthetase_idx"]}) AS counts,
                    volume, variant
                    FROM ({variant_subquery})
                ),
                unnested_counts AS (
                    SELECT unnest(counts) AS counts, volume,
                        generate_subscripts(counts, 1) AS count_idx, variant
                    FROM synthetase_counts
                ),
                concs AS (
                    SELECT counts / volume * {scale_factor} AS bulk_concs, count_idx, variant
                    FROM unnested_counts
                )
                SELECT avg(bulk_concs) AS avg_concs, stddev(bulk_concs) AS std_concs,
                    min(bulk_concs) AS min_concs, count_idx, variant
                FROM concs
                GROUP BY variant, count_idx
                ORDER BY variant, count_idx
                """).pl()
            stats_column_names = {
                "avg_concs": "mean (units.umol / units.L)",
                "std_concs": "std (units.umol / units.L)",
                "min_concs": "min (units.umol / units.L)",
            }
            synthetase_condition = [
                f"{synth_id}__{condition_label}"
                for synth_id in media_info["synthetase_ids"]
            ]
            trna_synthetase_stats = (
                trna_synthetase_stats.rename(stats_column_names)
                .with_columns(
                    pl.col("count_idx")
                    .map_elements(
                        lambda idx: synthetase_condition[idx - 1],
                        return_dtype=pl.Utf8,
                    )
                    .alias("synthetase_condition")
                )
                .select(["synthetase_condition"] + list(stats_column_names.values()))
            )
            media_info["summary_stats"].append(trna_synthetase_stats)

    # Combine and save data
    final_df = pl.concat(
        [
            pl.concat(condition_data["summary_stats"])
            for condition_data in variant_id_data.values()
        ]
    ).sort("synthetase_condition")

    # Build metadata comments for reproducibility
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    metadata_lines = [
        "# Generated by ecoli.analysis.multivariant.trna_synthetase_concs",
        f"# Run timestamp (UTC): {timestamp_utc}",
        f"# Experiment ID: {experiment_id}",
        f"# Skip n gens: {skip_gens if skip_gens is not None else '0'}",
    ]

    output_path = f"{outdir}/trna_synthetase_dynamic_range.tsv"
    buffer = StringIO()
    final_df.write_csv(buffer, separator="\t", quote_style="non_numeric")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))
        f.write("\n")
        f.write(buffer.getvalue())
    print(f"Saved tRNA synthetase dynamic range data to: {output_path}")
