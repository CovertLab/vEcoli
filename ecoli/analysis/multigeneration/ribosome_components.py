import altair as alt
import os
from typing import Any

from duckdb import DuckDBPyConnection
import pickle
import polars as pl
import numpy as np

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
    """
    Plots the timetrace of counts for each of the components of the ribosomal
    subunits (rRNAs and ribosomal proteins).
    """

    # Load sim_data
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Load IDs of ribosome components from sim_data
    s30_protein_ids = sim_data.molecule_groups.s30_proteins
    s30_16s_rRNA_ids = sim_data.molecule_groups.s30_16s_rRNA
    s30_full_complex_id = [sim_data.molecule_ids.s30_full_complex]
    s50_protein_ids = sim_data.molecule_groups.s50_proteins
    s50_23s_rRNA_ids = sim_data.molecule_groups.s50_23s_rRNA
    s50_5s_rRNA_ids = sim_data.molecule_groups.s50_5s_rRNA
    s50_full_complex_id = [sim_data.molecule_ids.s50_full_complex]

    # Get complexation stoichiometries of ribosomal proteins
    complexation = sim_data.process.complexation
    s30_monomers = complexation.get_monomers(s30_full_complex_id[0])
    s50_monomers = complexation.get_monomers(s50_full_complex_id[0])
    s30_subunit_id_to_stoich = {
        subunit_id: stoich
        for (subunit_id, stoich) in zip(
            s30_monomers["subunitIds"], s30_monomers["subunitStoich"]
        )
    }
    s50_subunit_id_to_stoich = {
        subunit_id: stoich
        for (subunit_id, stoich) in zip(
            s50_monomers["subunitIds"], s50_monomers["subunitStoich"]
        )
    }
    s30_protein_stoich = np.array(
        [s30_subunit_id_to_stoich[subunit_id] for subunit_id in s30_protein_ids]
    )
    s50_protein_stoich = np.array(
        [s50_subunit_id_to_stoich[subunit_id] for subunit_id in s50_protein_ids]
    )

    # Get metadata for extracting indexes
    unique_molecule_metadata = field_metadata(
        conn, config_sql, "listeners__unique_molecule_counts"
    )
    monomer_metadata = field_metadata(conn, config_sql, "listeners__monomer_counts")

    # Extract indexes
    active_ribosome_index = unique_molecule_metadata.index("active_ribosome")

    monomer_id_to_index = {
        monomer_id: i for (i, monomer_id) in enumerate(monomer_metadata)
    }
    s30_protein_indexes = [
        monomer_id_to_index[protein_id] for protein_id in s30_protein_ids
    ]
    s50_protein_indexes = [
        monomer_id_to_index[protein_id] for protein_id in s50_protein_ids
    ]

    # Define named indexes for bulk molecules
    s30_16s_rRNAs = named_idx(
        "listeners__bulk_molecules",
        s30_16s_rRNA_ids,
        list(range(len(s30_16s_rRNA_ids))),  # Will be resolved by named_idx
    )
    s50_23s_rRNAs = named_idx(
        "listeners__bulk_molecules",
        s50_23s_rRNA_ids,
        list(range(len(s50_23s_rRNA_ids))),
    )
    s50_5s_rRNAs = named_idx(
        "listeners__bulk_molecules", s50_5s_rRNA_ids, list(range(len(s50_5s_rRNA_ids)))
    )
    s30_full_complex = named_idx("listeners__bulk_molecules", s30_full_complex_id, [0])
    s50_full_complex = named_idx("listeners__bulk_molecules", s50_full_complex_id, [0])

    # Named indexes for monomer counts
    s30_proteins = named_idx(
        "listeners__monomer_counts", s30_protein_ids, s30_protein_indexes
    )
    s50_proteins = named_idx(
        "listeners__monomer_counts", s50_protein_ids, s50_protein_indexes
    )

    # Named index for active ribosomes
    active_ribosomes = named_idx(
        "listeners__unique_molecule_counts",
        ["active_ribosome"],
        [active_ribosome_index],
    )

    # Load data
    ribosome_data = read_stacked_columns(
        history_sql,
        [
            s30_16s_rRNAs,
            s50_23s_rRNAs,
            s50_5s_rRNAs,
            s30_full_complex,
            s50_full_complex,
            s30_proteins,
            s50_proteins,
            active_ribosomes,
        ],
        conn=conn,
    )

    # Convert to DataFrame and add time column
    df = pl.DataFrame(ribosome_data).with_columns(**{"Time (min)": pl.col("time") / 60})

    # Calculate protein counts divided by stoichiometry
    s30_protein_counts_cols = []
    for i, protein_id in enumerate(s30_protein_ids):
        col_name = f"s30_protein_{i}_normalized"
        df = df.with_columns(
            (pl.col(protein_id) / s30_protein_stoich[i]).alias(col_name)
        )
        s30_protein_counts_cols.append(col_name)

    s50_protein_counts_cols = []
    for i, protein_id in enumerate(s50_protein_ids):
        col_name = f"s50_protein_{i}_normalized"
        df = df.with_columns(
            (pl.col(protein_id) / s50_protein_stoich[i]).alias(col_name)
        )
        s50_protein_counts_cols.append(col_name)

    # Calculate limiting protein counts and total counts
    df = df.with_columns(
        [
            # S30 calculations
            pl.min_horizontal(s30_protein_counts_cols).alias(
                "s30_limiting_protein_counts"
            ),
            (
                pl.sum_horizontal(s30_16s_rRNA_ids)
                + pl.col(s30_full_complex_id[0])
                + pl.col("active_ribosome")
            ).alias("s30_16s_rRNA_total_counts"),
            (pl.col(s30_full_complex_id[0]) + pl.col("active_ribosome")).alias(
                "s30_total_counts"
            ),
            # S50 calculations
            pl.min_horizontal(s50_protein_counts_cols).alias(
                "s50_limiting_protein_counts"
            ),
            (
                pl.sum_horizontal(s50_23s_rRNA_ids)
                + pl.col(s50_full_complex_id[0])
                + pl.col("active_ribosome")
            ).alias("s50_23s_rRNA_total_counts"),
            (
                pl.sum_horizontal(s50_5s_rRNA_ids)
                + pl.col(s50_full_complex_id[0])
                + pl.col("active_ribosome")
            ).alias("s50_5s_rRNA_total_counts"),
            (pl.col(s50_full_complex_id[0]) + pl.col("active_ribosome")).alias(
                "s50_total_counts"
            ),
        ]
    )

    # Create plots
    # 30S components plot
    s30_plot = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q").title("Time (min)"),
            y=alt.Y("value:Q").title("30S component counts").scale(domain=[0, 60000]),
            color=alt.Color("variable:N").title("Component"),
            strokeDash=alt.StrokeDash("variable:N").scale(
                domain=[
                    "s30_limiting_protein_counts",
                    "s30_16s_rRNA_total_counts",
                    "s30_total_counts",
                ],
                range=[[5, 5], [0], [3, 3]],
            ),
        )
        .transform_fold(
            [
                "s30_limiting_protein_counts",
                "s30_16s_rRNA_total_counts",
                "s30_total_counts",
            ],
            as_=["variable", "value"],
        )
        .transform_calculate(
            variable_label="datum.variable === 's30_limiting_protein_counts' ? 'limiting r-protein' : "
            "datum.variable === 's30_16s_rRNA_total_counts' ? '16S rRNA' : '30S subunit'"
        )
        .encode(
            color=alt.Color("variable_label:N")
            .title("Component")
            .scale(
                domain=["limiting r-protein", "16S rRNA", "30S subunit"],
                range=["#cccccc", "#1f77b4", "#000000"],
            )
        )
        .properties(title="30S Ribosomal Subunit Components", width=600, height=200)
    )

    # 50S components plot
    s50_plot = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("Time (min):Q").title("Time (min)"),
            y=alt.Y("value:Q").title("50S component counts").scale(domain=[0, 60000]),
            color=alt.Color("variable:N").title("Component"),
            strokeDash=alt.StrokeDash("variable:N").scale(
                domain=[
                    "s50_limiting_protein_counts",
                    "s50_23s_rRNA_total_counts",
                    "s50_5s_rRNA_total_counts",
                    "s50_total_counts",
                ],
                range=[[5, 5], [0], [0], [3, 3]],
            ),
        )
        .transform_fold(
            [
                "s50_limiting_protein_counts",
                "s50_23s_rRNA_total_counts",
                "s50_5s_rRNA_total_counts",
                "s50_total_counts",
            ],
            as_=["variable", "value"],
        )
        .transform_calculate(
            variable_label="datum.variable === 's50_limiting_protein_counts' ? 'limiting r-protein' : "
            "datum.variable === 's50_23s_rRNA_total_counts' ? '23S rRNA' : "
            "datum.variable === 's50_5s_rRNA_total_counts' ? '5S rRNA' : '50S subunit'"
        )
        .encode(
            color=alt.Color("variable_label:N")
            .title("Component")
            .scale(
                domain=["limiting r-protein", "23S rRNA", "5S rRNA", "50S subunit"],
                range=["#cccccc", "#ff7f0e", "#2ca02c", "#000000"],
            )
        )
        .properties(title="50S Ribosomal Subunit Components", width=600, height=200)
    )

    # Combine plots vertically
    combined_plot = alt.vconcat(s30_plot, s50_plot).resolve_scale(color="independent")

    # Save the plot
    combined_plot.save(os.path.join(outdir, "ribosome_components.html"))
