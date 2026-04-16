"""
Cross-variant sensitivity-overview plots for sensitivity-campaign workflows.

For each variant (= one perturbed RNA-seq dataset = one ParCa run), compute:

* mean doubling time across all sims in that variant
* mean dry-mass at the *last* timestep of the *last* generation
* mass-drift slope (per-generation regression of mean dry-mass on generation idx)
* fraction of sims that reached the final generation (proxy for "healthy")

Then plot each of those vs. the sensitivity-axis value pulled from the
campaign sidecar (e.g. ``sigma`` for ``add_log_normal_noise``). Variants for
which the parca itself never produced sim_data are absent from the parquet
inputs; they show up as missing points in the plots and are listed in the
companion CSV with ``parca_status="missing"``.

Inputs (from the standard analysis API):
- ``params["campaign_sidecar"]``: path to ``<campaign>.campaign.json`` written
  by ``run_sensitivity_campaign.py``. Optional — if absent, plots fall back
  to "variant index" on the x-axis.
- ``params["axis_param"]``: which operator-param key to use as the x-axis
  (default: first scalar param in the spec's grid, falling back to "sigma").

Outputs:
- ``sensitivity_overview.html`` — composite Altair plot
- ``sensitivity_overview.tsv`` — one row per variant with metrics and
  campaign metadata
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import read_stacked_columns


def _load_campaign_sidecar(path_str: str | None) -> dict | None:
    """Load the campaign sidecar JSON written by run_sensitivity_campaign.py."""
    if not path_str:
        return None
    p = Path(path_str)
    if not p.is_absolute():
        # Resolve relative paths from the project root (cwd at run-time on the
        # sim container — the workflow_config.json lives next to it).
        candidates = [Path.cwd() / p, Path("/vEcoli") / p]
        p = next((c for c in candidates if c.exists()), p)
    if not p.exists():
        print(f"sensitivity_overview: campaign sidecar not found at {p}; "
              "x-axis will fall back to variant index")
        return None
    with open(p) as f:
        return cast(dict, json.load(f))


def _expand_param_grid(grid: dict[str, list]) -> list[dict]:
    """Cartesian product of param_grid (matching run_sensitivity_campaign)."""
    import itertools
    keys = list(grid)
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]


def _build_variant_label_table(sidecar: dict | None, axis_param: str | None) -> pl.DataFrame:
    """
    Build a polars table mapping variant index → (operator, axis_value, seed,
    dataset_id, parca_status). variant 0 is the source baseline when
    include_source_as_baseline; subsequent variants follow the cartesian
    expansion order.
    """
    if sidecar is None:
        return pl.DataFrame({
            "variant": [],
            "axis_value": [],
            "seed": [],
            "dataset_id": [],
            "operator": [],
        })

    spec = sidecar["spec"]
    operator = spec["operator"]
    grid = spec["param_grid"]
    expanded = _expand_param_grid(grid)
    include_baseline = sidecar.get("include_source_as_baseline", True)
    generated_ids = sidecar.get("generated_dataset_ids", [])

    # Pick axis param if not explicitly set: first scalar key in the grid,
    # excluding obvious bookkeeping (``seed``).
    if axis_param is None:
        for k in grid:
            if k != "seed":
                axis_param = k
                break
        if axis_param is None and "seed" in grid:
            axis_param = "seed"

    rows: list[dict] = []
    variant_offset = 0
    if include_baseline:
        rows.append({
            "variant": 0,
            "axis_value": 0.0,  # baseline gets axis=0 (sentinel: "no perturbation")
            "seed": None,
            "dataset_id": spec["source_dataset_id"],
            "operator": "baseline",
        })
        variant_offset = 1

    for i, params in enumerate(expanded):
        rows.append({
            "variant": variant_offset + i,
            "axis_value": float(params.get(axis_param, float("nan"))) if axis_param else None,
            "seed": params.get("seed"),
            "dataset_id": generated_ids[i] if i < len(generated_ids) else None,
            "operator": operator,
        })

    # Extra pre-existing datasets (Annabelle's exclusions, real alternate
    # datasets, overlay datasets) are appended after the operator grid.
    extra_ids = sidecar.get("extra_dataset_ids", [])
    extra_offset = variant_offset + len(expanded)
    for i, did in enumerate(extra_ids):
        rows.append({
            "variant": extra_offset + i,
            "axis_value": None,
            "seed": None,
            "dataset_id": did,
            "operator": "extra",
        })

    return pl.DataFrame(rows)


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
    sidecar = _load_campaign_sidecar(params.get("campaign_sidecar"))
    axis_param = params.get("axis_param")
    axis_label = axis_param or "axis"

    # --- Per-(variant, lineage_seed, generation, agent) doubling time ---
    dt_subquery = cast(str, read_stacked_columns(
        history_sql, ["time"], order_results=False, success_sql=success_sql,
    ))
    sim_dt = conn.sql(f"""
        SELECT variant, lineage_seed, generation, agent_id,
               (max(time) - min(time)) / 60.0 AS doubling_time_min
        FROM ({dt_subquery})
        GROUP BY variant, lineage_seed, generation, agent_id
    """).pl()

    # --- End-of-life mass per sim ---
    mass_subquery = cast(str, read_stacked_columns(
        history_sql,
        ["time", "listeners__mass__dry_mass"],
        order_results=False,
        success_sql=success_sql,
    ))
    final_mass = conn.sql(f"""
        SELECT variant, lineage_seed, generation, agent_id,
               last(listeners__mass__dry_mass ORDER BY time) AS final_dry_mass_fg
        FROM ({mass_subquery})
        GROUP BY variant, lineage_seed, generation, agent_id
    """).pl()

    # --- Combine + variant-level aggregation ---
    per_sim = sim_dt.join(
        final_mass,
        on=["variant", "lineage_seed", "generation", "agent_id"],
        how="full",
        coalesce=True,
    )

    # Mean dry mass at the final generation per variant.
    last_gen_per_variant = per_sim.group_by("variant").agg(
        pl.col("generation").max().alias("max_generation")
    )
    last_gen_mass = (
        per_sim.join(last_gen_per_variant, on="variant")
        .filter(pl.col("generation") == pl.col("max_generation"))
        .group_by("variant").agg(
            pl.col("final_dry_mass_fg").mean().alias("final_dry_mass_fg"),
        )
    )

    # Mass drift: regression slope of mean(final_dry_mass) on generation.
    per_gen_mass = (
        per_sim.group_by(["variant", "generation"])
        .agg(pl.col("final_dry_mass_fg").mean().alias("mean_mass_fg"))
        .sort(["variant", "generation"])
    )
    drift_rows: list[dict] = []
    for variant in per_gen_mass.select("variant").unique().to_series().to_list():
        sub = per_gen_mass.filter(pl.col("variant") == variant)
        if len(sub) < 2:
            slope = None
        else:
            xs = sub["generation"].to_numpy().astype(float)
            ys = sub["mean_mass_fg"].to_numpy().astype(float)
            x_mean = xs.mean()
            y_mean = ys.mean()
            denom = ((xs - x_mean) ** 2).sum()
            slope = float(((xs - x_mean) * (ys - y_mean)).sum() / denom) if denom > 0 else None
        drift_rows.append({"variant": variant, "mass_drift_per_gen_fg": slope})
    mass_drift = pl.DataFrame(drift_rows, schema={"variant": pl.Int64, "mass_drift_per_gen_fg": pl.Float64})

    # Variant-level doubling-time aggregate.
    dt_per_variant = per_sim.group_by("variant").agg(
        pl.col("doubling_time_min").mean().alias("mean_doubling_time_min"),
        pl.col("doubling_time_min").count().alias("n_sims"),
    )

    # Healthiness: fraction of sims that made it to the final requested generation.
    # We can't infer the "requested" generation count from history alone, so use
    # the max generation observed for each variant divided by the global max.
    global_max_gen = int(per_sim["generation"].max() or 0)
    health = (
        per_sim.group_by("variant").agg(
            pl.col("generation").max().alias("max_gen_reached"),
        )
        .with_columns(
            (pl.col("max_gen_reached") / max(global_max_gen, 1)).alias("frac_max_gen")
        )
    )

    summary = (
        dt_per_variant
        .join(last_gen_mass, on="variant", how="full", coalesce=True)
        .join(mass_drift, on="variant", how="full", coalesce=True)
        .join(health, on="variant", how="full", coalesce=True)
    )

    # Join campaign metadata (axis values, dataset_ids).
    label_table = _build_variant_label_table(sidecar, axis_param)
    if not label_table.is_empty():
        summary = summary.join(label_table, on="variant", how="full", coalesce=True)
    else:
        summary = summary.with_columns(
            pl.col("variant").cast(pl.Float64).alias("axis_value"),
            pl.lit(None, dtype=pl.Int64).alias("seed"),
            pl.lit(None, dtype=pl.Utf8).alias("dataset_id"),
            pl.lit("unknown", dtype=pl.Utf8).alias("operator"),
        )

    summary = summary.sort("variant")

    # --- Write CSV ---
    tsv_path = os.path.join(outdir, "sensitivity_overview.tsv")
    summary.write_csv(tsv_path, separator="\t")
    print(f"sensitivity_overview: wrote {tsv_path}")

    # --- Plots ---
    summary_pd = summary.to_pandas()

    def _scatter(y_col: str, y_title: str) -> alt.Chart:
        return (
            alt.Chart(summary_pd)
            .mark_point(size=80, filled=True)
            .encode(
                x=alt.X("axis_value:Q", title=f"{axis_label} (operator parameter)"),
                y=alt.Y(f"{y_col}:Q", title=y_title),
                color=alt.Color("operator:N", legend=alt.Legend(title="operator")),
                tooltip=[
                    "variant:O",
                    "operator:N",
                    "axis_value:Q",
                    "seed:N",
                    "n_sims:Q",
                    "mean_doubling_time_min:Q",
                    "final_dry_mass_fg:Q",
                    "mass_drift_per_gen_fg:Q",
                    "frac_max_gen:Q",
                    "dataset_id:N",
                ],
            )
            .properties(width=480, height=240, title=y_title)
        )

    composite = alt.vconcat(
        alt.hconcat(
            _scatter("mean_doubling_time_min", "Mean doubling time (min)"),
            _scatter("final_dry_mass_fg", "Final-generation mean dry mass (fg)"),
        ),
        alt.hconcat(
            _scatter("mass_drift_per_gen_fg", "Mass drift per generation (fg/gen)"),
            _scatter("frac_max_gen", "Fraction of max generation reached"),
        ),
    ).properties(
        title=f"Sensitivity overview: {len(summary_pd)} variants"
        + (f" (operator: {sidecar['spec']['operator']})" if sidecar else "")
    )

    html_path = os.path.join(outdir, "sensitivity_overview.html")
    composite.save(html_path)
    print(f"sensitivity_overview: wrote {html_path}")
