"""
Multivariant analysis for the homeostatic-target-scaling sweep.

Writes three CSVs to ``outdir``:

1. ``doubling_time_sweep.csv`` — per-generation doubling time (seconds) for
   every (variant, lineage_seed, generation, agent_id) tuple, annotated with
   the configured homeostatic target scale and whether the generation
   successfully divided.
2. ``doubling_time_sweep_summary.csv`` — per-variant statistics over the last
   four of ten generations (``generation >= 6``), successful gens only.
3. ``homeostatic_scale_check.csv`` — sanity check comparing the observed
   ratio of ``target_homeostatic_dmdt_conc`` magnitudes between each variant
   and variant 0 against the configured scale, for three reference
   metabolites.

Also prints PASS/FAIL for the homeostatic scale check and the list of failed
(variant, lineage_seed, generation) tuples.
"""

import os
from typing import Any

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl

from ecoli.library.parquet_emitter import (
    read_stacked_columns,
)


REFERENCE_METABOLITES = ["WATER[c]", "ATP[c]", "L-ALPHA-ALANINE[c]"]


def _variant_scale_frame(
    variant_metadata: dict[str, dict[int, Any]],
) -> pl.DataFrame:
    """Flatten ``variant_metadata`` into a (variant, scale) polars DataFrame."""
    rows: list[dict[str, Any]] = []
    # variant_metadata is keyed by experiment_id; each sub-dict maps
    # variant index -> metadata dict containing ``scale``.
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
    # Deduplicate — same (variant, scale) may appear under multiple experiments
    return pl.DataFrame(rows).unique(subset=["variant"], keep="first")


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

    # ------------------------------------------------------------------
    # 1. Per-generation doubling time CSV
    # ------------------------------------------------------------------
    doubling_time_sql = read_stacked_columns(
        history_sql,
        ["time"],
        order_results=False,
    )
    doubling_times = conn.sql(f"""
        SELECT (max(time) - min(time)) AS doubling_time_s,
               experiment_id, variant, lineage_seed, generation, agent_id
        FROM ({doubling_time_sql})
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
    """).pl()

    successful_sims = conn.sql(success_sql).pl()
    succeeded = doubling_times.join(
        successful_sims,
        how="semi",
        on=["experiment_id", "variant", "lineage_seed", "agent_id"],
    ).with_columns(pl.lit(True).alias("succeeded"))
    failed = doubling_times.join(
        successful_sims,
        how="anti",
        on=["experiment_id", "variant", "lineage_seed", "agent_id"],
    ).with_columns(pl.lit(False).alias("succeeded"))

    all_gens = pl.concat([succeeded, failed], how="vertical_relaxed")

    scale_df = _variant_scale_frame(variant_metadata)
    all_gens = all_gens.join(scale_df, on="variant", how="left")

    sweep_df = all_gens.select(
        [
            "variant",
            "scale",
            "lineage_seed",
            "generation",
            "agent_id",
            "doubling_time_s",
            "succeeded",
        ]
    ).sort(["variant", "lineage_seed", "generation", "agent_id"])
    sweep_path = os.path.join(outdir, "doubling_time_sweep.csv")
    sweep_df.write_csv(sweep_path)
    print(f"Wrote {sweep_path}")

    # Stdout: failed tuples
    failed_tuples = (
        failed.select(["variant", "lineage_seed", "generation"])
        .unique()
        .sort(["variant", "lineage_seed", "generation"])
    )
    if failed_tuples.height > 0:
        print("[doubling_time_sweep] Failed (variant, lineage_seed, generation):")
        for row in failed_tuples.iter_rows(named=True):
            print(f"  ({row['variant']}, {row['lineage_seed']}, {row['generation']})")
    else:
        print("[doubling_time_sweep] No failed generations.")

    # ------------------------------------------------------------------
    # 2. Per-variant summary over generations 6-9 (successful only)
    # ------------------------------------------------------------------
    # n_success / n_failed counted across ALL generations, not just 6-9,
    # so the user can see total failure counts per variant.
    success_counts = succeeded.group_by("variant").agg(pl.len().alias("n_success"))
    failed_counts = failed.group_by("variant").agg(pl.len().alias("n_failed"))

    gen69 = succeeded.filter(pl.col("generation") >= 6)
    gen69_stats = gen69.group_by("variant").agg(
        [
            pl.col("doubling_time_s").mean().alias("mean_gen6_9_s"),
            pl.col("doubling_time_s").std().alias("std_gen6_9_s"),
        ]
    )

    summary = (
        scale_df.join(success_counts, on="variant", how="left")
        .join(failed_counts, on="variant", how="left")
        .join(gen69_stats, on="variant", how="left")
        .with_columns(
            [
                pl.col("n_success").fill_null(0),
                pl.col("n_failed").fill_null(0),
            ]
        )
        .select(
            [
                "variant",
                "scale",
                "n_success",
                "n_failed",
                "mean_gen6_9_s",
                "std_gen6_9_s",
            ]
        )
        .sort("variant")
    )
    summary_path = os.path.join(outdir, "doubling_time_sweep_summary.csv")
    summary.write_csv(summary_path)
    print(f"Wrote {summary_path}")

    # ------------------------------------------------------------------
    # 3. Homeostatic scale sanity check
    # ------------------------------------------------------------------
    homeostatic_sql = read_stacked_columns(
        history_sql,
        [
            "listeners__fba_results__target_homeostatic_dmdt_conc",
            "listeners__fba_results__homeostatic_metabolites",
            "time",
        ],
        order_results=False,
    )
    # For each variant, grab the first emitted timestep for lineage_seed=0,
    # generation=0.
    first_ts = conn.sql(f"""
        WITH base AS (
            SELECT variant, time,
                   listeners__fba_results__target_homeostatic_dmdt_conc AS target,
                   listeners__fba_results__homeostatic_metabolites AS mets
            FROM ({homeostatic_sql})
            WHERE lineage_seed = 0 AND generation = 0
        ),
        ranked AS (
            SELECT variant, time, target, mets,
                   row_number() OVER (PARTITION BY variant ORDER BY time) AS rn
            FROM base
        )
        SELECT variant, target, mets
        FROM ranked
        WHERE rn = 1
        ORDER BY variant
    """).pl()

    # Build {variant -> {metabolite -> target}} dict.
    variant_targets: dict[int, dict[str, float]] = {}
    for row in first_ts.iter_rows(named=True):
        variant_idx = int(row["variant"])
        mets_list = list(row["mets"])
        target_arr = np.asarray(list(row["target"]), dtype=float)
        variant_targets[variant_idx] = {
            met: float(target_arr[i]) for i, met in enumerate(mets_list)
        }

    scale_lookup = {
        int(r["variant"]): r["scale"] for r in scale_df.iter_rows(named=True)
    }

    check_rows: list[dict[str, Any]] = []
    v0_targets = variant_targets.get(0, {})
    for variant_idx in sorted(variant_targets.keys()):
        configured_scale = scale_lookup.get(variant_idx)
        for met in REFERENCE_METABOLITES:
            target_v0 = v0_targets.get(met)
            target_vn = variant_targets[variant_idx].get(met)
            if target_v0 is None or target_vn is None or abs(target_v0) == 0.0:
                observed_scale = None
                abs_error = None
            else:
                observed_scale = abs(target_vn) / abs(target_v0)
                if configured_scale is None:
                    abs_error = None
                else:
                    abs_error = abs(observed_scale - float(configured_scale))
            check_rows.append(
                {
                    "variant": variant_idx,
                    "configured_scale": configured_scale,
                    "metabolite": met,
                    "target_v0": target_v0,
                    "target_vN": target_vn,
                    "observed_scale": observed_scale,
                    "abs_error": abs_error,
                }
            )

    check_df = pl.DataFrame(check_rows)
    check_path = os.path.join(outdir, "homeostatic_scale_check.csv")
    check_df.write_csv(check_path)
    print(f"Wrote {check_path}")

    # PASS/FAIL
    offenders = check_df.filter(
        (pl.col("abs_error").is_not_null()) & (pl.col("abs_error") >= 1e-6)
    )
    if offenders.height == 0:
        print("[homeostatic_scale_check] PASS")
    else:
        print("[homeostatic_scale_check] FAIL")
        for row in offenders.iter_rows(named=True):
            print(f"  {row}")
