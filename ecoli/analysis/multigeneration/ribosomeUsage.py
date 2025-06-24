import altair as alt
import os
from typing import Any

from duckdb import DuckDBPyConnection
import pickle
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
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
    # Get sim data from pickle file
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Get ids for 30S and 50S subunits
    complexIds30S = [sim_data.molecule_ids.s30_full_complex]
    complexIds50S = [sim_data.molecule_ids.s50_full_complex]

    # Get molecular weights for 30S and 50S subunits, and add these two for 70S
    nAvogadro = sim_data.constants.n_avogadro
    mw30S = sim_data.getter.get_masses(complexIds30S)
    mw50S = sim_data.getter.get_masses(complexIds50S)
    mw70S = mw30S + mw50S

    # Load data
    bulk_molecule_data = read_stacked_columns(
        history_sql,
        [
            "listeners__bulk_molecules__counts",
            "listeners__unique_molecule_counts__unique_molecule_counts",
            "listeners__ribosome_data__did_initialize",
            "listeners__ribosome_data__actual_elongations",
            "listeners__ribosome_data__did_terminate",
            "listeners__ribosome_data__effective_elongation_rate",
            "listeners__mass__cell_mass",
            "time",
            "time_step_sec",
        ],
        conn=conn,
    )

    # Convert to DataFrame
    df = pl.DataFrame(bulk_molecule_data).with_columns(
        **{
            "Time (min)": pl.col("time") / 60,
            "Cell Volume (L)": (pl.col("listeners__mass__cell_mass") * 1e-15)
            / sim_data.constants.cell_density,
        }
    )

    # Get indexes for 30S and 50S subunits based on ids
    bulk_molecule_ids = field_metadata(
        conn, config_sql, "listeners__bulk_molecules__counts"
    )
    complexIndexes30S = [bulk_molecule_ids.index(comp) for comp in complexIds30S]
    complexIndexes50S = [bulk_molecule_ids.index(comp) for comp in complexIds50S]

    # Get indexes for active ribosomes
    unique_molecule_ids = field_metadata(
        conn, config_sql, "listeners__unique_molecule_counts__unique_molecule_counts"
    )
    ribosomeIndex = unique_molecule_ids.index("active_ribosome")

    # Extract specific columns from arrays
    df = df.with_columns(
        [
            pl.col("listeners__bulk_molecules__counts")
            .list.get(complexIndexes30S[0])
            .alias("counts_30S"),
            pl.col("listeners__bulk_molecules__counts")
            .list.get(complexIndexes50S[0])
            .alias("counts_50S"),
            pl.col("listeners__unique_molecule_counts__unique_molecule_counts")
            .list.get(ribosomeIndex)
            .alias("active_ribosome_counts"),
        ]
    )

    # Calculate ribosome statistics
    df = df.with_columns(
        [
            # Total ribosome counts
            (
                pl.col("active_ribosome_counts")
                + pl.min_horizontal([pl.col("counts_30S"), pl.col("counts_50S")])
            ).alias("total_ribosome_counts"),
            # Concentrations
            (
                (1 / nAvogadro)
                * pl.col("active_ribosome_counts")
                / pl.col("Cell Volume (L)")
            ).alias("active_ribosome_concentration_M"),
            # Masses
            ((1 / nAvogadro) * pl.col("counts_30S") * mw30S).alias("mass_30S"),
            ((1 / nAvogadro) * pl.col("counts_50S") * mw50S).alias("mass_50S"),
            ((1 / nAvogadro) * pl.col("active_ribosome_counts") * mw70S).alias(
                "active_ribosome_mass"
            ),
            # Rates per time*volume
            (
                pl.col("listeners__ribosome_data__did_initialize")
                / (pl.col("time_step_sec") * pl.col("Cell Volume (L)"))
            ).alias("activations_per_time_volume"),
            (
                pl.col("listeners__ribosome_data__did_terminate")
                / (pl.col("time_step_sec") * pl.col("Cell Volume (L)"))
            ).alias("deactivations_per_time_volume"),
        ]
    )

    # Calculate additional derived columns
    df = df.with_columns(
        [
            # Total ribosome concentration
            (
                (1 / nAvogadro)
                * pl.col("total_ribosome_counts")
                / pl.col("Cell Volume (L)")
            ).alias("total_ribosome_concentration_M"),
            # Molar fraction active
            (
                pl.col("active_ribosome_counts").cast(pl.Float64)
                / pl.col("total_ribosome_counts")
            ).alias("molar_fraction_active"),
            # Total ribosome mass and mass fraction
            (
                pl.col("active_ribosome_mass") + pl.col("mass_30S") + pl.col("mass_50S")
            ).alias("total_ribosome_mass"),
        ]
    )

    # Calculate mass fraction active
    df = df.with_columns(
        [
            (pl.col("active_ribosome_mass") / pl.col("total_ribosome_mass")).alias(
                "mass_fraction_active"
            ),
        ]
    )

    # Convert concentrations to mM
    df = df.with_columns(
        [
            (pl.col("active_ribosome_concentration_M") * 1000).alias(
                "active_ribosome_concentration_mM"
            ),
            (pl.col("total_ribosome_concentration_M") * 1000).alias(
                "total_ribosome_concentration_mM"
            ),
        ]
    )

    # Create individual plots
    plots = []

    # Time step plot
    timestep_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("time_step_sec:Q", title="Length of time step (s)"),
        )
        .properties(title="Time Step", width=300, height=150)
    )
    plots.append(timestep_plot)

    # Cell volume plot
    volume_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("Cell Volume (L):Q", title="Cell volume (L)"),
        )
        .properties(title="Cell Volume", width=300, height=150)
    )
    plots.append(volume_plot)

    # Total ribosome counts
    total_counts_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("total_ribosome_counts:Q", title="Total ribosome count"),
        )
        .properties(title="Total Ribosome Count", width=300, height=150)
    )
    plots.append(total_counts_plot)

    # Total ribosome concentration
    total_conc_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("total_ribosome_concentration_mM:Q", title="[Total ribosome] (mM)"),
        )
        .properties(title="Total Ribosome Concentration", width=300, height=150)
    )
    plots.append(total_conc_plot)

    # Active ribosome counts
    active_counts_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("active_ribosome_counts:Q", title="Active ribosome count"),
        )
        .properties(title="Active Ribosome Count", width=300, height=150)
    )
    plots.append(active_counts_plot)

    # Active ribosome concentration
    active_conc_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "active_ribosome_concentration_mM:Q", title="[Active ribosome] (mM)"
            ),
        )
        .properties(title="Active Ribosome Concentration", width=300, height=150)
    )
    plots.append(active_conc_plot)

    # Molar fraction active
    molar_fraction_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("molar_fraction_active:Q", title="Molar fraction active ribosomes"),
        )
        .properties(title="Molar Fraction Active Ribosomes", width=300, height=150)
    )
    plots.append(molar_fraction_plot)

    # Mass fraction active
    mass_fraction_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y("mass_fraction_active:Q", title="Mass fraction active ribosomes"),
        )
        .properties(title="Mass Fraction Active Ribosomes", width=300, height=150)
    )
    plots.append(mass_fraction_plot)

    # Activations per timestep
    activations_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "listeners__ribosome_data__did_initialize:Q",
                title="Activations per timestep",
            ),
        )
        .properties(title="Activations per Timestep", width=300, height=150)
    )
    plots.append(activations_plot)

    # Deactivations per timestep
    deactivations_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "listeners__ribosome_data__did_terminate:Q",
                title="Deactivations per timestep",
            ),
        )
        .properties(title="Deactivations per Timestep", width=300, height=150)
    )
    plots.append(deactivations_plot)

    # Activations per time*volume
    activations_tv_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "activations_per_time_volume:Q", title="Activations per time*volume"
            ),
        )
        .properties(title="Activations per Time*Volume", width=300, height=150)
    )
    plots.append(activations_tv_plot)

    # Deactivations per time*volume
    deactivations_tv_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "deactivations_per_time_volume:Q", title="Deactivations per time*volume"
            ),
        )
        .properties(title="Deactivations per Time*Volume", width=300, height=150)
    )
    plots.append(deactivations_tv_plot)

    # AA translated
    aa_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "listeners__ribosome_data__actual_elongations:Q", title="AA translated"
            ),
        )
        .properties(title="Amino Acids Translated", width=300, height=150)
    )
    plots.append(aa_plot)

    # Effective elongation rate
    elongation_plot = (
        alt.Chart(df.to_pandas())
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q", title="Time (min)"),
            y=alt.Y(
                "listeners__ribosome_data__effective_elongation_rate:Q",
                title="Effective elongation rate",
            ),
        )
        .properties(title="Effective Elongation Rate", width=300, height=150)
    )
    plots.append(elongation_plot)

    # Combine all plots in a grid layout (7 rows, 2 columns)
    left_column = alt.vconcat(*plots[:7])
    right_column = alt.vconcat(*plots[7:])
    combined_plot = alt.hconcat(left_column, right_column)

    # Save the plot
    combined_plot.save(os.path.join(outdir, "ribosome_usage.html"))
