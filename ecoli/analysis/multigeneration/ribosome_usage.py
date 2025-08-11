"""
Record several things:
1. cell volume over time
2. total / active ribosome count and concentration
3. active ribosome molar / mass fraction
4. Ribosome activation / deactivation count
5. # of AA. be translated
6. the effective ribosome elongation rate
"""

import altair as alt
import os
from typing import Any
import pickle

import polars as pl
from duckdb import DuckDBPyConnection
import pandas as pd
import numpy as np

from ecoli.library.parquet_emitter import open_arbitrary_sim_data, named_idx
from ecoli.library.schema import bulk_name_to_idx

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
    """Visualize ribosome usage statistics for E. coli simulation."""
    # Load sim_data
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Get molecular IDs for ribosome subunits
    complex_ids_30s = [sim_data.molecule_ids.s30_full_complex]
    complex_ids_50s = [sim_data.molecule_ids.s50_full_complex]
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()

    # precompute indices as Python ints (following ribosome_production.py pattern)
    idx_30s = [
        int(i) for i in np.atleast_1d(bulk_name_to_idx(complex_ids_30s, bulk_ids))
    ]
    idx_50s = [
        int(i) for i in np.atleast_1d(bulk_name_to_idx(complex_ids_50s, bulk_ids))
    ]

    # Get molecular weights
    n_avogadro = sim_data.constants.n_avogadro
    mw_30s = sim_data.getter.get_masses(complex_ids_30s)
    mw_50s = sim_data.getter.get_masses(complex_ids_50s)
    mw_70s = mw_30s + mw_50s

    required_columns = [
        "time",
        "variant",
        "generation",
        "agent_id",
        "experiment_id",
        "lineage_seed",
        "listeners__mass__instantaneous_growth_rate",
        "listeners__mass__cell_mass",
        "listeners__mass__volume",
        "listeners__ribosome_data__did_initialize",
        "listeners__ribosome_data__actual_elongations",
        "listeners__ribosome_data__did_terminate",
        "listeners__ribosome_data__effective_elongation_rate",
        "listeners__unique_molecule_counts__active_ribosome",
    ]

    # Create the bulk index expressions
    expr_30s = named_idx("bulk", [f"bulk_30s_{i}" for i in idx_30s], [idx_30s])
    expr_50s = named_idx("bulk", [f"bulk_50s_{i}" for i in idx_50s], [idx_50s])

    # load data
    sql = f"""
    SELECT
        {", ".join(required_columns)},
        {expr_30s},
        {expr_50s}
    FROM ({history_sql})
    WHERE agent_id = 0
    ORDER BY generation, time
    """

    df = conn.sql(sql).pl()

    # Convert time
    if "time" in df.columns:
        df = df.with_columns((pl.col("time") / 60).alias("time_min"))
        df = df.with_columns([(pl.col("time") + 1).alias("time_step_sec")])

    # Calculate ribosome subunit counts
    cols_30s = [c for c in df.columns if c.startswith("bulk_30s_")]
    cols_50s = [c for c in df.columns if c.startswith("bulk_50s_")]
    df = df.with_columns(
        [
            # compute bulk ribosome subunit counts
            pl.sum_horizontal(cols_30s).alias("counts_30s"),
            pl.sum_horizontal(cols_50s).alias("counts_50s"),
            # compute unique ribosomes
            pl.col("listeners__unique_molecule_counts__active_ribosome")
            .fill_null(0)
            .alias("active_ribosome_counts"),
        ]
    )

    # Calculate total ribosome counts and fractions
    df = df.with_columns(
        [
            (
                pl.col("active_ribosome_counts")
                + pl.min_horizontal(pl.col("counts_30s"), pl.col("counts_50s"))
            ).alias("total_ribosome_counts"),
            (
                pl.col("active_ribosome_counts").cast(pl.Float64)
                / (
                    pl.col("active_ribosome_counts")
                    + pl.min_horizontal(pl.col("counts_30s"), pl.col("counts_50s"))
                )
            ).alias("molar_fraction_active"),
        ]
    )

    if "listeners__mass__cell_mass" in df.columns:
        cell_density = sim_data.constants.cell_density.asNumber()
        df = df.with_columns(
            (1e-15 * pl.col("listeners__mass__cell_mass") / cell_density).alias(
                "cell_volume"
            )
        )

    # Calculate concentrations
    df = df.with_columns(
        [
            (
                pl.col("total_ribosome_counts")
                / n_avogadro.asNumber()
                / pl.col("cell_volume")
            ).alias("total_ribosome_concentration_mM"),
            (
                pl.col("active_ribosome_counts")
                / n_avogadro.asNumber()
                / pl.col("cell_volume")
            ).alias("active_ribosome_concentration_mM"),
        ]
    )

    # Calculate masses
    mw_30s_value = mw_30s.asNumber() if hasattr(mw_30s, "asNumber") else float(mw_30s)
    mw_50s_value = mw_50s.asNumber() if hasattr(mw_50s, "asNumber") else float(mw_50s)
    mw_70s_value = mw_70s.asNumber() if hasattr(mw_70s, "asNumber") else float(mw_70s)

    df = df.with_columns(
        [
            (pl.col("counts_30s") / n_avogadro.asNumber() * mw_30s_value).alias(
                "mass_30s"
            ),
            (pl.col("counts_50s") / n_avogadro.asNumber() * mw_50s_value).alias(
                "mass_50s"
            ),
            (
                pl.col("active_ribosome_counts") / n_avogadro.asNumber() * mw_70s_value
            ).alias("active_ribosome_mass"),
        ]
    )

    df = df.with_columns(
        [
            (
                pl.col("active_ribosome_mass") + pl.col("mass_30s") + pl.col("mass_50s")
            ).alias("total_ribosome_mass"),
            (
                pl.col("active_ribosome_mass")
                / (
                    pl.col("active_ribosome_mass")
                    + pl.col("mass_30s")
                    + pl.col("mass_50s")
                )
            ).alias("mass_fraction_active"),
        ]
    )

    # Calculate rates per time and volume
    if "time_step_sec" in df.columns and "cell_volume" in df.columns:
        df = df.with_columns(
            [
                (
                    pl.col("listeners__ribosome_data__did_initialize")
                    / (pl.col("cell_volume") / 1e-15)
                ).alias("activations_per_volume"),
                (
                    pl.col("listeners__ribosome_data__did_terminate")
                    / (pl.col("cell_volume") / 1e-15)
                ).alias("deactivations_per_volume"),
            ]
        )

    # Select columns for plotting
    plot_columns = ["time_min", "variant", "generation"]

    # Add other columns that exist
    for col in [
        "time_step_sec",
        "cell_volume",
        "total_ribosome_counts",
        "total_ribosome_concentration_mM",
        "active_ribosome_counts",
        "active_ribosome_concentration_mM",
        "molar_fraction_active",
        "mass_fraction_active",
        "listeners__ribosome_data__did_initialize",
        "listeners__ribosome_data__did_terminate",
        "activations_per_volume",
        "deactivations_per_volume",
        "listeners__ribosome_data__actual_elongations",
        "listeners__ribosome_data__effective_elongation_rate",
    ]:
        if col in df.columns:
            plot_columns.append(col)

    plot_df = df.select(plot_columns)

    # ----------------------------------------- #

    def create_line_chart(y_field, title, y_title, skip_first_point=False):
        """Create line chart with optional skipping of first data point."""
        data = plot_df.to_pandas()
        if skip_first_point:
            # Group by variant and generation, skip first point of each group
            filtered_data = []
            for (variant, generation), group in data.groupby(["variant", "generation"]):
                if len(group) > 1:
                    filtered_data.append(group.iloc[1:])
                else:
                    filtered_data.append(group)
            data = (
                pd.concat(filtered_data, ignore_index=True) if filtered_data else data
            )

        chart = (
            alt.Chart(data)
            .mark_line()
            .encode(
                x=alt.X("time_min:Q", title="Time (min)"),
                y=alt.Y(f"{y_field}:Q", title=y_title),
                color=alt.Color("generation:N", legend=alt.Legend(title="Generation")),
            )
            .properties(title=title, width=600, height=120)
        )

        return chart

    # ----------------------------------------- #
    plots = []

    # Create all 14 plots following the original order
    if "time_step_sec" in plot_df.columns:
        plots.append(
            create_line_chart(
                "time_step_sec", "Length of Time Step", "Length of time step (s)"
            )
        )

    if "cell_volume" in plot_df.columns:
        plots.append(create_line_chart("cell_volume", "Cell Volume", "Cell volume (L)"))

    if "total_ribosome_counts" in plot_df.columns:
        plots.append(
            create_line_chart(
                "total_ribosome_counts", "Total Ribosome Count", "Total ribosome count"
            )
        )

    if "total_ribosome_concentration_mM" in plot_df.columns:
        plots.append(
            create_line_chart(
                "total_ribosome_concentration_mM",
                "Total Ribosome Concentration",
                "[Total ribosome] (mM)",
            )
        )

    if "active_ribosome_counts" in plot_df.columns:
        plots.append(
            create_line_chart(
                "active_ribosome_counts",
                "Active Ribosome Count",
                "Active ribosome count",
                skip_first_point=True,
            )
        )

    if "active_ribosome_concentration_mM" in plot_df.columns:
        plots.append(
            create_line_chart(
                "active_ribosome_concentration_mM",
                "Active Ribosome Concentration",
                "[Active ribosome] (mM)",
                skip_first_point=True,
            )
        )

    if "molar_fraction_active" in plot_df.columns:
        plots.append(
            create_line_chart(
                "molar_fraction_active",
                "Molar Fraction Active Ribosomes",
                "Molar fraction active ribosomes",
                skip_first_point=True,
            )
        )

    if "mass_fraction_active" in plot_df.columns:
        plots.append(
            create_line_chart(
                "mass_fraction_active",
                "Mass Fraction Active Ribosomes",
                "Mass fraction active ribosomes",
                skip_first_point=True,
            )
        )

    if "listeners__ribosome_data__did_initialize" in plot_df.columns:
        plots.append(
            create_line_chart(
                "listeners__ribosome_data__did_initialize",
                "Ribosome Activations",
                "Activations per timestep",
            )
        )

    if "listeners__ribosome_data__did_terminate" in plot_df.columns:
        plots.append(
            create_line_chart(
                "listeners__ribosome_data__did_terminate",
                "Ribosome Deactivations",
                "Deactivations per timestep",
            )
        )

    if "activations_per_volume" in plot_df.columns:
        plots.append(
            create_line_chart(
                "activations_per_volume",
                "Activations per Volume (fL)",
                "Activations per Volume (fL)",
            )
        )

    if "deactivations_per_volume" in plot_df.columns:
        plots.append(
            create_line_chart(
                "deactivations_per_volume",
                "Deactivations per Volume (fL)",
                "Deactivations per Volume (fL)",
            )
        )

    if "listeners__ribosome_data__actual_elongations" in plot_df.columns:
        plots.append(
            create_line_chart(
                "listeners__ribosome_data__actual_elongations",
                "Amino Acids Translated",
                "AA translated",
            )
        )

    if "listeners__ribosome_data__effective_elongation_rate" in plot_df.columns:
        plots.append(
            create_line_chart(
                "listeners__ribosome_data__effective_elongation_rate",
                "Effective Ribosome Elongation Rate",
                "Effective elongation rate",
            )
        )

    if not plots:
        fallback_df = pl.DataFrame(
            {
                "message": ["No data available for ribosome usage visualization"],
                "x": [0],
                "y": [0],
            }
        )
        fallback_plot = (
            alt.Chart(fallback_df)
            .mark_text(size=20, color="red")
            .encode(x="x:Q", y="y:Q", text="message:N")
            .properties(
                width=600,
                height=400,
                title="Ribosome Usage Statistics - No Data Available",
            )
        )
        plots.append(fallback_plot)

    # Arrange plots in 2 columns as in original
    left_plots = plots[::2]  # Even indices (0, 2, 4, ...)
    right_plots = plots[1::2]  # Odd indices (1, 3, 5, ...)

    # Ensure both columns have same length by adding empty chart if needed
    if len(left_plots) > len(right_plots):
        empty_chart = (
            alt.Chart(pl.DataFrame({"x": [0], "y": [0]}))
            .mark_point(opacity=0)
            .encode(x="x:Q", y="y:Q")
            .properties(width=600, height=120)
        )
        right_plots.append(empty_chart)
    elif len(right_plots) > len(left_plots):
        empty_chart = (
            alt.Chart(pl.DataFrame({"x": [0], "y": [0]}))
            .mark_point(opacity=0)
            .encode(x="x:Q", y="y:Q")
            .properties(width=600, height=120)
        )
        left_plots.append(empty_chart)

    # Create two column layout
    left_column = alt.vconcat(*left_plots)
    right_column = alt.vconcat(*right_plots)
    combined_plot = (
        alt.hconcat(left_column, right_column)
        .resolve_scale(x="shared", y="independent")
        .properties(title="Ribosome Usage Statistics")
    )

    output_path = os.path.join(outdir, "ribosome_usage_report.html")
    combined_plot.save(output_path)
    print(f"Saved visualization to: {output_path}")

    return combined_plot
