"""
Plot dynamic traces of genes with high expression (> 20 counts of mRNA)

EG10367_RNA	24.8	gapA	Glyceraldehyde 3-phosphate dehydrogenase
EG11036_RNA	25.2	tufA	Elongation factor Tu
EG50002_RNA	26.2	rpmA	50S Ribosomal subunit protein L27
EG10671_RNA	30.1	ompF	Outer membrane protein F
EG50003_RNA	38.7	acpP	Apo-[acyl carrier protein]
EG10669_RNA	41.1	ompA	Outer membrane protein A
EG10873_RNA	44.7	rplL	50S Ribosomal subunit protein L7/L12 dimer
EG12179_RNA	46.2	cspE	Transcription antiterminator and regulator of RNA stability
EG10321_RNA	53.2	fliC	Flagellin
EG10544_RNA	97.5	lpp		Murein lipoprotein
"""

import altair as alt
import os
from typing import Any
import pickle
import polars as pl
import numpy as np

from duckdb import DuckDBPyConnection
from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
    named_idx,
    read_stacked_columns,
)


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
    """Plot dynamic traces of genes with high expression (> 20 counts of mRNA)"""
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    cistron_array = sim_data.process.transcription.cistron_data.struct_array
    all_ids = list(cistron_array["id"])
    deg_rates = {row["id"]: row["deg_rate"] for row in cistron_array}

    # Define high-expression cistrons
    target_ids = [
        "EG10367_RNA",
        "EG11036_RNA",
        "EG50002_RNA",
        "EG10671_RNA",
        "EG50003_RNA",
        "EG10669_RNA",
        "EG10873_RNA",
        "EG12179_RNA",
        "EG10321_RNA",
        "EG10544_RNA",
    ]
    valid_ids = [cid for cid in target_ids if cid in all_ids]
    if not valid_ids:
        print("[ERROR] No matching cistrons in sim_data")
        return

    # Retrieve metadata for degradation and counts
    deg_field = "listeners__rna_degradation_listener__count_RNA_degraded_per_cistron"
    cnt_field = "listeners__rna_counts__mRNA_cistron_counts"
    try:
        deg_meta = field_metadata(conn, config_sql, deg_field)
        cnt_meta = field_metadata(conn, config_sql, cnt_field)
    except Exception as e:
        print(f"[ERROR] field_metadata failed: {e}")
        return

    # Find indices for valid cistrons
    deg_indices = [deg_meta.index(cid) for cid in valid_ids]
    cnt_indices = [cnt_meta.index(cid) for cid in valid_ids]

    # Build named_idx structures
    deg_named = named_idx(deg_field, valid_ids, [deg_indices])
    cnt_named = named_idx(cnt_field, [f"{i}_cnt" for i in valid_ids], [cnt_indices])

    # Read stacked columns
    try:
        data_dict = read_stacked_columns(
            history_sql,
            [deg_named, cnt_named],
            conn=conn,
        )
    except Exception as e:
        print(f"[ERROR] read_stacked_columns failed: {e}")
        return

    # Convert to Polars DataFrame
    df = pl.DataFrame(data_dict)
    # convert to minutes
    if "time" in df.columns:
        df = df.with_columns((pl.col("time") / 60).alias("time_min"))

    # Melt degradation and counts
    deg_cols = valid_ids
    cnt_cols = [f"{i}_cnt" for i in valid_ids]
    deg_df = df.select(["time_min"] + deg_cols).melt(
        "time_min", variable_name="cistron", value_name="degraded"
    )
    cnt_df = (
        df.select(["time_min"] + cnt_cols)
        .melt("time_min", variable_name="cistron", value_name="counts")
        .with_columns(pl.col("cistron").str.replace("_cnt", "", literal=True))
    )
    joined = deg_df.join(cnt_df, on=["time_min", "cistron"])

    # Smooth and fit per cistron
    charts = []
    window = 100
    for cid in valid_ids[:9]:
        sub = joined.filter(pl.col("cistron") == cid).sort("time_min")
        if sub.height < 2 * window:
            continue
        counts = sub["counts"].to_numpy()
        degraded = sub["degraded"].to_numpy()
        # smoothing
        smooth_c = np.convolve(counts, np.ones(window) / window, mode="same")
        dt = np.gradient(sub["time_min"].to_numpy() * 60)
        rate = degraded / np.maximum(dt, 1e-10)
        smooth_r = np.convolve(rate, np.ones(window) / window, mode="same")
        mask = (
            np.isfinite(smooth_c)
            & (smooth_c > 0)
            & np.isfinite(smooth_r)
            & (smooth_r >= 0)
        )
        A = smooth_c[mask]
        y = smooth_r[mask]
        if len(A) < 10:
            continue
        kdeg = np.linalg.lstsq(A[:, None], y, rcond=None)[0][0]

        # Prepare data for plotting
        plot_df = pl.DataFrame({"RNA_counts": A, "RNA_degraded": y})
        # Regression line data
        line_x = np.linspace(A.min(), A.max(), 100)
        line_y = kdeg * line_x

        # Scatter
        scatter = (
            alt.Chart(plot_df)
            .mark_circle(size=20, opacity=0.6, color="blue")
            .encode(x="RNA_counts:Q", y="RNA_degraded:Q")
        )
        # Regression line
        line = (
            alt.Chart(pl.DataFrame({"RNA_counts": line_x, "RNA_degraded": line_y}))
            .mark_line(color="red", strokeWidth=0.5)
            .encode(x="RNA_counts:Q", y="RNA_degraded:Q")
        )

        # Combine and style
        title = f"{cid} kdeg meas: {kdeg:.1e} s⁻¹ | kdeg exp: {deg_rates[cid]:.1e} s⁻¹"
        charts.append((scatter + line).properties(title=title, width=250, height=200))

    if charts:
        rows = [alt.hconcat(*charts[i : i + 3]) for i in range(0, len(charts), 3)]
        combined = alt.vconcat(*rows).properties(
            title="RNA Decay - High Expression Genes"
        )
        output = os.path.join(outdir, "rna_decay_03_high.html")
        combined.save(output)
        print(f"[INFO] Saved visualization to: {output}")
        return combined
    else:
        print("[ERROR] No charts generated")
        return None
