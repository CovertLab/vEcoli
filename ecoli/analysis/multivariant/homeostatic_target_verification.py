"""
Verify the ``homeostatic_target_scale`` variant actually applies the expected
scale factor in each variant's sim_data.

For each variant, this loads the per-variant ``sim_data`` pickle and reads
``sim_data.homeostatic_target_scale``. That attribute is what
``LoadSimData.get_metabolism_redux_config`` (in
``ecoli/library/sim_data.py``) plumbs into the metabolism-redux process,
where it scales every entry of ``conc_dict`` before the homeostatic
objective is built. So if this attribute matches the configured scale,
the targets *will* be scaled at runtime.

The variant index 0 is the implicit baseline (no variant applied), so its
scale should default to 1.0.
"""

import os
import pickle
from typing import Any

import altair as alt

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import polars as pl


def _configured_scale(meta: Any) -> float | None:
    """Extract a configured scale from a single variant_metadata entry."""
    if isinstance(meta, dict):
        if "scale" in meta:
            try:
                return float(meta["scale"])
            except (TypeError, ValueError):
                return None
        if "homeostatic_target_scale" in meta:
            return _configured_scale(meta["homeostatic_target_scale"])
        return None
    if isinstance(meta, (int, float)):
        return float(meta)
    # Strings like "baseline" -> no configured scale; baseline defaults to 1.0
    return None


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

    rows: list[dict[str, Any]] = []
    for experiment_id, per_variant_paths in sim_data_dict.items():
        configured_per_variant = variant_metadata.get(experiment_id, {})
        for variant_idx, sim_data_path in per_variant_paths.items():
            configured = _configured_scale(configured_per_variant.get(variant_idx))
            # Variant 0 is the implicit baseline (no variant applied) and
            # will use the default scale of 1.0.
            if configured is None and int(variant_idx) == 0:
                configured = 1.0

            try:
                with open(sim_data_path, "rb") as f:
                    sim_data = pickle.load(f)
            except Exception as exc:
                rows.append(
                    {
                        "experiment_id": experiment_id,
                        "variant": int(variant_idx),
                        "configured_scale": configured,
                        "sim_data_scale": None,
                        "match": False,
                        "note": f"failed to load sim_data: {exc}",
                    }
                )
                continue

            actual = getattr(sim_data, "homeostatic_target_scale", 1.0)
            try:
                actual_f = float(actual)
            except (TypeError, ValueError):
                actual_f = None

            match = (
                configured is not None
                and actual_f is not None
                and abs(actual_f - configured) < 1e-9
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "variant": int(variant_idx),
                    "configured_scale": configured,
                    "sim_data_scale": actual_f,
                    "match": bool(match),
                    "note": "" if match else "MISMATCH",
                }
            )

    df = pl.DataFrame(rows).sort(["experiment_id", "variant"])
    csv_path = os.path.join(outdir, "homeostatic_target_scale_check.csv")
    df.write_csv(csv_path)
    print(f"Wrote {csv_path}")

    print("[homeostatic_target_verification] Per-variant scale check:")
    for row in df.iter_rows(named=True):
        print(
            f"  variant={row['variant']:>2}  "
            f"configured={row['configured_scale']}  "
            f"sim_data={row['sim_data_scale']}  "
            f"{'OK' if row['match'] else 'MISMATCH'}"
        )

    if df.filter(~pl.col("match")).height == 0:
        print(
            "[homeostatic_target_verification] PASS — every variant's "
            "sim_data.homeostatic_target_scale matches the configured value."
        )
    else:
        print("[homeostatic_target_verification] FAIL — see CSV for details.")

    plot_df = df.filter(
        pl.col("configured_scale").is_not_null()
        & pl.col("sim_data_scale").is_not_null()
    ).with_columns(pl.col("variant").cast(pl.Utf8).alias("variant_str"))
    if plot_df.height > 0:
        configured_pts = (
            alt.Chart(plot_df)
            .mark_point(shape="circle", size=120, color="steelblue", filled=True)
            .encode(
                x=alt.X("variant_str:N", title="Variant"),
                y=alt.Y("configured_scale:Q", title="Scale"),
                tooltip=["variant", "configured_scale"],
            )
        )
        actual_pts = (
            alt.Chart(plot_df)
            .mark_point(shape="diamond", size=120, color="orange", filled=True)
            .encode(
                x="variant_str:N",
                y="sim_data_scale:Q",
                tooltip=["variant", "sim_data_scale"],
            )
        )
        chart = (configured_pts + actual_pts).properties(
            title="Configured (blue circle) vs sim_data (orange diamond) homeostatic_target_scale"
        )
        chart.save(f"{outdir}/homeostatic_target_verification.html")
