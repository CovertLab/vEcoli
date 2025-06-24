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
from typing import Any, cast
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
    # Load sim_data
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    all_cistron_ids = sim_data.process.transcription.cistron_data["id"].tolist()

    cistron_ids = [
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

    names = [
        "gapA - Glyceraldehyde 3-phosphate dehydrogenase",
        "tufA - Elongation factor Tu",
        "rpmA - 50S Ribosomal subunit protein L27",
        "ompF - Outer membrane protein F",
        "acpP - Apo-[acyl carrier protein]",
        "ompA - Outer membrane protein A",
        "rplL - 50S Ribosomal subunit protein L7/L12 dimer",
        "cspE - Transcription antiterminator and regulator of RNA stability",
        "fliC - Flagellin",
        "lpp - Murein lipoprotein",
    ]

    cistron_idxs = [all_cistron_ids.index(x) for x in cistron_ids]
    deg_rates = sim_data.process.transcription.cistron_data["deg_rate"][cistron_idxs]

    # Get indexes for the specific cistrons we want to track
    rna_degradation_idx_dict = {
        cistron: i
        for i, cistron in enumerate(
            field_metadata(
                conn,
                config_sql,
                "listenersrna_degradation__count_RNA_degraded_per_cistron",
            )
        )
    }

    rna_counts_idx_dict = {
        cistron: i
        for i, cistron in enumerate(
            field_metadata(conn, config_sql, "listenersrna_counts__mRNA_cistron_counts")
        )
    }

    cistron_degradation_indexes = [
        cast(int, rna_degradation_idx_dict.get(cistron_id))
        for cistron_id in cistron_ids
    ]

    cistron_counts_indexes = [
        cast(int, rna_counts_idx_dict.get(cistron_id)) for cistron_id in cistron_ids
    ]

    # Load data using vEcoli pattern
    degradation_columns = named_idx(
        "listenersrna_degradation__count_RNA_degraded_per_cistron",
        cistron_ids,
        cistron_degradation_indexes,
    )

    counts_columns = named_idx(
        "listenersrna_counts__mRNA_cistron_counts", cistron_ids, cistron_counts_indexes
    )

    # Read data
    data = read_stacked_columns(
        history_sql,
        [degradation_columns, counts_columns, "time", "timeStepSec"],
        conn=conn,
    )

    df = pl.DataFrame(data)

    # Convert to numpy arrays for processing (similar to original logic)
    N = 100  # smoothing window

    # Group by simulation and process each separately
    processed_data = []

    for sim_data_group in df.group_by(["variant", "seed", "generation"]):
        sim_df = sim_data_group[1].sort("time")

        # Extract arrays for this simulation
        dt = sim_df["timeStepSec"].to_numpy()

        # Process degradation counts
        degraded_counts = np.column_stack(
            [
                sim_df[
                    f"listenersrna_degradation__count_RNA_degraded_per_cistron__{cistron_id}"
                ].to_numpy()
                for cistron_id in cistron_ids
            ]
        )

        # Process RNA counts
        rna_counts = np.column_stack(
            [
                sim_df[
                    f"listenersrna_counts__mRNA_cistron_counts__{cistron_id}"
                ].to_numpy()
                for cistron_id in cistron_ids
            ]
        )

        # Apply smoothing (similar to original)
        if len(dt) > 2 * N:
            degraded_smoothed = np.nan * np.ones_like(degraded_counts)
            counts_smoothed = np.nan * np.ones_like(rna_counts)

            for col_idx in range(degraded_counts.shape[1]):
                # Smooth degradation rates
                degraded_smoothed[:, col_idx] = np.convolve(
                    degraded_counts[:, col_idx] / dt, np.ones(N) / N, mode="same"
                )
                # Smooth counts
                counts_smoothed[:, col_idx] = np.convolve(
                    rna_counts[:, col_idx], np.ones(N) / N, mode="same"
                )

            # Trim edges
            degraded_trimmed = degraded_smoothed[N:-N, :]
            counts_trimmed = counts_smoothed[N:-N, :]

            processed_data.append(
                {
                    "degraded": degraded_trimmed,
                    "counts": counts_trimmed,
                    "variant": sim_data_group[1]["variant"].iloc[0],
                    "seed": sim_data_group[1]["seed"].iloc[0],
                    "generation": sim_data_group[1]["generation"].iloc[0],
                }
            )

    if not processed_data:
        print("No data available for processing")
        return

    # Combine all processed data
    all_degraded = np.vstack([d["degraded"] for d in processed_data])
    all_counts = np.vstack([d["counts"] for d in processed_data])

    # Create subplot charts using Altair
    charts = []

    for subplot_idx in range(
        min(9, len(cistron_ids))
    ):  # Limit to 9 subplots like original
        if subplot_idx >= len(cistron_ids):
            break

        y = all_degraded[:, subplot_idx]
        A = all_counts[:, subplot_idx]

        try:
            # Calculate degradation rate using least squares
            kdeg, _, _, _ = np.linalg.lstsq(A[:, np.newaxis], y, rcond=None)
            kdeg = kdeg[0]
        except (ValueError, np.linalg.LinAlgError):
            print(f"Skipping subplot {subplot_idx} because not enough data")
            continue

        # Subsample data for plotting (similar to original ::N)
        plot_data = pl.DataFrame({"RNA_counts": A[::N], "RNA_degraded": y[::N]})

        chart = (
            alt.Chart(plot_data)
            .mark_circle()
            .encode(
                x=alt.X("RNA_counts:Q", title="RNA (counts)"),
                y=alt.Y("RNA_degraded:Q", title="RNA degraded (counts)"),
            )
            .properties(
                title=f"{names[subplot_idx].split(' - ')[0]}\n"
                f"kdeg meas: {kdeg:.1e}\n"
                f"kdeg exp: {deg_rates[subplot_idx]:.1e}",
                width=250,
                height=200,
            )
        )

        charts.append(chart)

    # Arrange charts in 3x3 grid
    if charts:
        # Group charts into rows of 3
        rows = []
        for i in range(0, len(charts), 3):
            row_charts = charts[i : i + 3]
            if len(row_charts) == 1:
                rows.append(row_charts[0])
            else:
                rows.append(alt.hconcat(*row_charts))

        # Combine rows vertically
        if len(rows) == 1:
            combined_plot = rows[0]
        else:
            combined_plot = alt.vconcat(*rows)

        combined_plot.save(os.path.join(outdir, "rna_decay_03_high.html"))
    else:
        print("No charts were generated due to insufficient data")
