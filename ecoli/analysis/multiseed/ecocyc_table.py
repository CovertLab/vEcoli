"""
Generates tables of data to share with EcoCyc for display on the "modeling" tab.

TODO:
    other values
        weighted average for counts (time step weighted and cell cycle progress weighted)
        max/min
"""

import csv
import json
import os
import pickle
import tempfile
from typing import TYPE_CHECKING, Any

from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl
from scipy.stats import pearsonr

from ecoli.library.parquet_emitter import (
    config_value,
    field_metadata,
    open_arbitrary_sim_data,
    open_output_file,
    num_cells,
    read_stacked_columns,
    skip_n_gens,
)
from ecoli.processes.metabolism import (
    COUNTS_UNITS,
    MASS_UNITS,
    TIME_UNITS,
    VOLUME_UNITS,
)
from wholecell.utils import units

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli

IGNORE_FIRST_N_GENS = 2

MEDIA_NAME_TO_ID = {
    "minimal": "MIX0-57",
    "minimal_minus_oxygen": "MIX0-57-ANAEROBIC",
    "minimal_plus_amino_acids": "MIX0-850",
    "minimal_acetate": "MIX0-58",
    "minimal_succinate": "MIX0-844",
}


def save_file(outdir, filename, columns, values: list[pl.Series]):
    outfile = os.path.join(outdir, filename)
    print(f"Saving data to {outfile}")
    # Data rows
    out_df = pl.DataFrame({k: s for k, s in zip(columns, values)})
    with open(outfile, "w") as f, tempfile.NamedTemporaryFile("w+") as temp_out:
        out_df.write_csv(temp_out.name, separator="\t")

        # Header for columns
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["# Column descriptions:"])
        for col, desc in columns.items():
            writer.writerow([f"# {col}", desc])

        for line in temp_out:
            f.write(line)


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
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data: "SimulationDataEcoli" = pickle.load(f)
    with open_output_file(validation_data_paths[0]) as f:
        validation_data = pickle.load(f)

    ignore_first_n_gens = params.get("ignore_first_n_gens", IGNORE_FIRST_N_GENS)

    media_name = sim_data.conditions[sim_data.condition]["nutrients"]
    media_id = MEDIA_NAME_TO_ID.get(media_name, media_name)

    # Ignore first N generations
    history_sql = skip_n_gens(history_sql, ignore_first_n_gens)
    config_sql = skip_n_gens(config_sql, ignore_first_n_gens)

    if num_cells(conn, config_sql) == 0:
        print("Skipping analysis -- not enough simulations run.")
        return

    # Load tables and attributes for mRNAs
    mRNA_ids = field_metadata(
        conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts"
    )
    # mass_unit = field_metadata(config_lf, 'listeners__mass__dry_mass')
    # assert mass_unit == 'fg'

    # Load tables and attributes for tRNAs and rRNAs
    bulk_ids = field_metadata(conn, config_sql, "bulk")
    bulk_id_to_idx = {bulk_id: i + 1 for i, bulk_id in enumerate(bulk_ids)}
    uncharged_tRNA_ids = sim_data.process.transcription.uncharged_trna_names
    uncharged_tRNA_idx = [bulk_id_to_idx[trna] for trna in uncharged_tRNA_ids]
    charged_tRNA_ids = sim_data.process.transcription.charged_trna_names
    charged_tRNA_idx = [bulk_id_to_idx[trna] for trna in charged_tRNA_ids]
    tRNA_cistron_ids = [tRNA_id[:-3] for tRNA_id in uncharged_tRNA_ids]
    rRNA_ids = [
        sim_data.molecule_groups.s30_16s_rRNA[0],
        sim_data.molecule_groups.s50_23s_rRNA[0],
        sim_data.molecule_groups.s50_5s_rRNA[0],
    ]
    rRNA_idx = [bulk_id_to_idx[trna] for trna in rRNA_ids]
    rRNA_cistron_ids = [rRNA_id[:-3] for rRNA_id in rRNA_ids]
    ribosomal_subunit_ids = [
        sim_data.molecule_ids.s30_full_complex,
        sim_data.molecule_ids.s50_full_complex,
    ]
    ribo_subunit_idx = [bulk_id_to_idx[ribo] for ribo in ribosomal_subunit_ids]
    # Filter out first timestep for each cell because counts_to_molar is 0
    rna_subquery = read_stacked_columns(
        history_sql,
        [
            # Extract only necessary bulk counts to reduce RAM usage
            f"list_select(bulk, {charged_tRNA_idx}) AS charged_tRNAs, "
            f"list_select(bulk, {uncharged_tRNA_idx}) AS uncharged_tRNAs, "
            f"list_select(bulk, {rRNA_idx}) AS rRNAs, "
            f"list_select(bulk, {ribo_subunit_idx}) AS ribo_subunits",
            "listeners__unique_molecule_counts__active_ribosome",
            "listeners__enzyme_kinetics__counts_to_molar",
            "listeners__mass__dry_mass",
            "listeners__rna_counts__mRNA_cistron_counts",
        ],
        remove_first=True,
        order_results=False,
    )
    rna_data = conn.sql(
        f"""
        WITH rna_list AS (
            SELECT
                -- Create RNA counts list of mRNAs, tRNAs, and rRNAs (in order)
                (
                    -- mRNA
                    listeners__rna_counts__mRNA_cistron_counts +
                    -- tRNA = charged + uncharged
                    [
                        trna[1] + trna[2]
                        FOR trna IN list_zip(charged_tRNAs, uncharged_tRNAs)
                    ] +
                    -- First rRNA = bulk + active ribosome + small subunit
                    [
                        rRNAs[1] + 
                        listeners__unique_molecule_counts__active_ribosome +
                        ribo_subunits[1]
                    ] +
                    -- Remaining rRNAs = bulk + active ribosome + large subunit
                    [
                        rrna_count +
                        listeners__unique_molecule_counts__active_ribosome +
                        ribo_subunits[2]
                        FOR rrna_count IN rRNAs[2:]
                    ]
                ) AS rna_counts,
                listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                listeners__mass__dry_mass AS dry_masses
            FROM ({rna_subquery})
        ),
        unnested_counts AS (
            SELECT
                -- Unnest RNA counts to perform aggregations (e.g. mean, stddev)
                unnest(rna_counts) AS rna_counts,
                -- Track list index for each unnested row for grouped aggregations
                generate_subscripts(rna_counts, 1) AS rna_idx,
                counts_to_molar, dry_masses
            FROM rna_list
        )
        SELECT
            avg(rna_counts) AS "rna-count-avg",
            stddev(rna_counts) AS "rna-count-std",
            avg(rna_counts * counts_to_molar) AS "rna-concentration-avg",
            stddev(rna_counts * counts_to_molar) AS "rna-concentration-std",
            avg(dry_masses) AS "dry-masses-avg"
        FROM unnested_counts
        GROUP BY rna_idx
        ORDER BY rna_idx
        """
    ).pl()

    # Filter out first timestep for each cell because counts_to_molar is 0
    gene_copy_num_subquery = read_stacked_columns(
        history_sql,
        ["listeners__rna_synth_prob__gene_copy_number"],
        remove_first=True,
        order_results=False,
    )
    gene_copy_data = conn.sql(
        f"""
        WITH unnested_counts AS (
            SELECT
                -- Unnest gene copy number to perform aggregations (e.g. mean, stddev)
                unnest(listeners__rna_synth_prob__gene_copy_number) AS gene_copy_numbers,
                -- Track list index for each unnested row for grouped aggregations
                generate_subscripts(listeners__rna_synth_prob__gene_copy_number, 1) AS gene_idx
            FROM ({gene_copy_num_subquery})
        )
        SELECT
            avg(gene_copy_numbers) AS "gene-copy-number-avg",
            stddev(gene_copy_numbers) AS "gene-copy-number-std"
        FROM unnested_counts
        GROUP BY gene_idx
        ORDER BY gene_idx
        """
    ).pl()

    # Retrieve gene copy numbers in order of RNA counts
    cistron_id_to_gene_id = {
        cistron["id"]: cistron["gene_id"]
        for cistron in sim_data.process.transcription.cistron_data
    }
    gene_ids = [
        cistron_id_to_gene_id[rna_id]
        for rna_id in mRNA_ids + tRNA_cistron_ids + rRNA_cistron_ids
    ]
    gene_ids_rna_synth_prob = field_metadata(
        conn, config_sql, "listeners__rna_synth_prob__gene_copy_number"
    )
    gene_id_to_idx = {gene: i for i, gene in enumerate(gene_ids_rna_synth_prob)}
    gene_to_rna_order_idx = [gene_id_to_idx[gene] for gene in gene_ids]

    # Get RNA molecular weights
    gene_id_to_mw = {
        gene_id: cistron_mw
        for (gene_id, cistron_mw) in zip(
            sim_data.process.transcription.cistron_data["gene_id"],
            sim_data.process.transcription.cistron_data["mw"].asNumber(
                units.fg / units.count
            ),
        )
    }
    rna_mw = np.array([gene_id_to_mw[gene] for gene in gene_ids])

    # Calculate relative statistics
    rel_to_type_count = np.zeros(len(rna_data))
    rel_to_type_mass = np.zeros(len(rna_data))
    start_idx = 0
    for add_idx in [len(mRNA_ids), len(tRNA_cistron_ids), len(rRNA_cistron_ids)]:
        rna_slice_mw = rna_mw[start_idx : start_idx + add_idx]
        rel_to_type_count[start_idx : start_idx + add_idx] = rna_data.select(
            pl.col("rna-count-avg").slice(start_idx, add_idx)
            / pl.col("rna-count-avg").slice(start_idx, add_idx).sum()
        )["rna-count-avg"]
        rel_to_type_mass[start_idx : start_idx + add_idx] = rna_data.select(
            pl.col("rna-count-avg").slice(start_idx, add_idx)
            * rna_slice_mw
            / (pl.col("rna-count-avg").slice(start_idx, add_idx) * rna_slice_mw).sum()
        )["rna-count-avg"]
        start_idx += add_idx
    rna_data = rna_data.with_columns(
        **{  # type: ignore[arg-type]
            "relative-rna-count-to-total-rna-type-counts": rel_to_type_count,
            "relative-rna-mass-to-total-rna-type-mass": rel_to_type_mass,
            "relative-rna-count-to-total-rna-counts": rna_data.select(
                pl.col("rna-count-avg") / pl.col("rna-count-avg").sum()
            )["rna-count-avg"],
            "relative-rna-mass-to-total-rna-mass": rna_data.select(
                pl.col("rna-count-avg")
                * rna_mw
                / (pl.col("rna-count-avg") * rna_mw).sum()
            )["rna-count-avg"],
            "relative-rna-mass-to-total-cell-dry-mass": rna_data.select(
                pl.col("rna-count-avg") * rna_mw / pl.col("dry-masses-avg").sum()
            )["rna-count-avg"],
            "id": pl.Series(gene_ids),
            "gene-copy-number-avg": gene_copy_data["gene-copy-number-avg"][
                gene_to_rna_order_idx
            ],
            "gene-copy-number-std": gene_copy_data["gene-copy-number-std"][
                gene_to_rna_order_idx
            ],
        }
    )

    columns = {
        "id": "Object ID, according to EcoCyc",
        "gene-copy-number-avg": "A floating point number",
        "gene-copy-number-std": "A floating point number",
        "rna-count-avg": "A floating point number",
        "rna-count-std": "A floating point number",
        "rna-concentration-avg": "A floating point number in mM units",
        "rna-concentration-std": "A floating point number in mM units",
        "relative-rna-count-to-total-rna-counts": "A floating point number",
        "relative-rna-count-to-total-rna-type-counts": "A floating point number",
        "relative-rna-mass-to-total-rna-mass": "A floating point number",
        "relative-rna-mass-to-total-rna-type-mass": "A floating point number",
        "relative-rna-mass-to-total-cell-dry-mass": "A floating point number",
    }
    values = [rna_data[k] for k in columns]

    save_file(outdir, f"wcm_rnas_{media_id}.tsv", columns, values)

    # Build dictionary for metadata
    ecocyc_metadata = {
        "git_hash": config_value(conn, config_sql, "git_hash"),
        "n_ignored_generations": ignore_first_n_gens,
        "n_total_generations": config_value(conn, config_sql, "generations"),
        "n_seeds": config_value(conn, config_sql, "n_init_sims"),
        "n_cells": num_cells(conn, config_sql),
        "n_timesteps": len(rna_data),
    }

    # Filter out first timestep for each cell because counts_to_molar is 0
    monomer_subquery = read_stacked_columns(
        history_sql,
        [
            "listeners__enzyme_kinetics__counts_to_molar",
            "listeners__mass__dry_mass",
            "listeners__monomer_counts",
        ],
        remove_first=True,
        order_results=False,
    )
    # Load tables and attributes for proteins
    monomer_data = conn.sql(
        f"""
        WITH unnested_counts AS (
            SELECT
                listeners__mass__dry_mass AS dry_masses,
                listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                -- Unnest monomer counts to calculate aggregations (e.g. mean)
                unnest(listeners__monomer_counts) AS monomer_counts,
                -- Track list index for each unnested row for grouped aggregations
                generate_subscripts(listeners__monomer_counts, 1) AS monomer_idx,
            FROM ({monomer_subquery})
        )
        SELECT
            avg(monomer_counts) AS "protein-count-avg",
            stddev(monomer_counts) AS "protein-count-std",
            avg(monomer_counts * counts_to_molar) AS "protein-concentration-avg",
            stddev(monomer_counts * counts_to_molar) AS "protein-concentration-std",
            avg(dry_masses) AS "dry-masses-avg"
        FROM unnested_counts
        GROUP BY monomer_idx
        ORDER BY monomer_idx
        """
    ).pl()
    monomer_ids = field_metadata(conn, config_sql, "listeners__monomer_counts")
    monomer_ecocyc_ids = [monomer[:-3] for monomer in monomer_ids]  # strip [*]
    monomer_mw = sim_data.getter.get_masses(monomer_ids).asNumber(
        units.fg / units.count
    )
    monomer_sim_data = sim_data.process.translation.monomer_data.struct_array
    monomer_to_gene_id = {
        monomer_id: cistron_id_to_gene_id[cistron_id]
        for cistron_id, monomer_id in zip(
            monomer_sim_data["cistron_id"], monomer_sim_data["id"]
        )
    }
    # Calculate relative statistics
    monomer_data = monomer_data.with_columns(
        **{
            "relative-protein-count-to-total-protein-counts": monomer_data.select(
                pl.col("protein-count-avg") / pl.col("protein-count-avg").sum()
            )["protein-count-avg"],
            "relative-protein-mass-to-total-protein-mass": monomer_data.select(
                pl.col("protein-count-avg")
                * monomer_mw
                / (pl.col("protein-count-avg") * monomer_mw).sum()
            )["protein-count-avg"],
            "relative-protein-mass-to-total-cell-dry-mass": monomer_data.select(
                pl.col("protein-count-avg") * monomer_mw / pl.col("dry-masses-avg")
            )["protein-count-avg"],
            "gene-id": pl.Series(
                [monomer_to_gene_id[monomer_id] for monomer_id in monomer_ids]
            ),
            "id": pl.Series(monomer_ecocyc_ids),
        }
    )
    monomer_data = monomer_data.join(
        rna_data.select(**{"gene-id": "id", "rna-count-avg": "rna-count-avg"}),
        on="gene-id",
    )
    monomer_data = monomer_data.with_columns(
        **{
            "relative-protein-count-to-protein-rna-counts": monomer_data.select(
                pl.col("protein-count-avg") / pl.col("rna-count-avg")
            )["protein-count-avg"]
        }
    )

    columns = {
        "id": "Object ID, according to EcoCyc",
        "protein-count-avg": "A floating point number",
        "protein-count-std": "A floating point number",
        "protein-concentration-avg": "A floating point number in mM units",
        "protein-concentration-std": "A floating point number in mM units",
        "relative-protein-count-to-protein-rna-counts": "A floating point number",
        "relative-protein-mass-to-total-protein-mass": "A floating point number",
        "relative-protein-mass-to-total-cell-dry-mass": "A floating point number",
    }
    values = [monomer_data[k] for k in columns]

    # Add validation data if sims used minimal glucose media
    if media_name == "minimal":
        protein_id_to_schmidt_counts = {
            item[0]: item[1] for item in validation_data.protein.schmidt2015Data
        }
        protein_counts_val = np.array(
            [
                protein_id_to_schmidt_counts.get(protein_id, np.nan)
                for protein_id in monomer_ids
            ]
        )

        columns["validation-count"] = "A floating point number"
        values.append(pl.Series(protein_counts_val))

        protein_val_exists = np.logical_not(np.isnan(protein_counts_val))
        r, _ = pearsonr(
            monomer_data["protein-count-avg"].filter(pl.Series(protein_val_exists)),
            protein_counts_val[protein_val_exists],
        )

        ecocyc_metadata["protein_validation_r_squared"] = r**2

    save_file(outdir, f"wcm_monomers_{media_id}.tsv", columns, values)

    # Read data and load attributes for complexes
    complex_ids = sim_data.process.complexation.ids_complexes
    complex_idx = [bulk_id_to_idx[cplx] for cplx in complex_ids]
    complex_subquery = read_stacked_columns(
        history_sql,
        [
            # Extract only complex bulk counts to reduce RAM usage
            f"list_select(bulk, {complex_idx}) AS complex_counts",
            "listeners__enzyme_kinetics__counts_to_molar",
            "listeners__mass__dry_mass",
        ],
        remove_first=True,
        order_results=False,
    )
    complex_data = conn.sql(
        f"""
        WITH unnested_counts AS (
            SELECT
                -- Unnest complex counts to calculate aggregations (e.g. mean)
                unnest(complex_counts) AS complex_counts,
                -- Track list index for each unnested row for grouped aggregations
                generate_subscripts(complex_counts, 1) AS complex_idx,
                listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                listeners__mass__dry_mass AS dry_masses,
            FROM ({complex_subquery})
        )
        SELECT
            avg(complex_counts) AS "complex-count-avg",
            stddev(complex_counts) AS "complex-count-std",
            avg(complex_counts * counts_to_molar) AS "complex-concentration-avg",
            stddev(complex_counts * counts_to_molar) AS "complex-concentration-std",
            avg(dry_masses) AS "dry-masses-avg"
        FROM unnested_counts
        GROUP BY complex_idx
        ORDER BY complex_idx
        """
    ).pl()

    # Calculate derived protein values
    complex_mw = sim_data.getter.get_masses(complex_ids).asNumber(
        units.fg / units.count
    )
    complex_data = complex_data.with_columns(
        **{
            "relative-complex-mass-to-total-protein-mass": complex_data.select(
                pl.col("complex-count-avg")
                * complex_mw
                / (pl.col("complex-count-avg") * complex_mw).sum()
            )["complex-count-avg"],
            "relative-complex-mass-to-total-cell-dry-mass": complex_data.select(
                pl.col("complex-count-avg") * complex_mw / pl.col("dry-masses-avg")
            )["complex-count-avg"],
            "id": pl.Series(
                [complex_id[:-3] for complex_id in complex_ids]
            ),  # strip [*]
        }
    )
    # Save complex data in table
    columns = {
        "id": "Object ID, according to EcoCyc",
        "complex-count-avg": "A floating point number",
        "complex-count-std": "A floating point number",
        "complex-concentration-avg": "A floating point number in mM units",
        "complex-concentration-std": "A floating point number in mM units",
        "relative-complex-mass-to-total-protein-mass": "A floating point number",
        "relative-complex-mass-to-total-cell-dry-mass": "A floating point number",
    }
    values = [complex_data[k] for k in columns]

    save_file(outdir, f"wcm_complexes_{media_id}.tsv", columns, values)

    # Load attributes for metabolic fluxes
    cell_density = sim_data.constants.cell_density
    cell_density = cell_density.asNumber(MASS_UNITS / VOLUME_UNITS)
    reaction_ids = sim_data.process.metabolism.base_reaction_ids

    # Read fluxes
    flux_subquery = read_stacked_columns(
        history_sql,
        [
            "listeners__fba_results__base_reaction_fluxes",
            "listeners__mass__cell_mass",
            "listeners__mass__dry_mass",
        ],
        order_results=False,
    )
    flux_data = conn.sql(
        f"""
        WITH unnest_fluxes AS (
            SELECT listeners__mass__dry_mass / 
                listeners__mass__cell_mass * {cell_density} AS conversion_coeffs,
                -- Unnest monomer counts to calculate aggregations (e.g. mean)
                unnest(listeners__fba_results__base_reaction_fluxes) AS fluxes,
                generate_subscripts(listeners__fba_results__base_reaction_fluxes, 1) AS idx,
            FROM ({flux_subquery})
        )
        SELECT 
            avg(fluxes / conversion_coeffs) AS "flux-avg",
            stddev(fluxes / conversion_coeffs) AS "flux-std"
        FROM unnest_fluxes
        GROUP BY idx
        ORDER BY idx
        """
    ).pl()

    flux_data = flux_data.with_columns(
        **{
            "id": pl.Series(reaction_ids),
            "flux-avg": (
                (COUNTS_UNITS / MASS_UNITS / TIME_UNITS) * pl.col("flux-avg")
            ).asNumber(units.mmol / units.g / units.h),
            "flux-std": (
                (COUNTS_UNITS / MASS_UNITS / TIME_UNITS) * pl.col("flux-std")
            ).asNumber(units.mmol / units.g / units.h),
        }
    )

    columns = {
        "id": "Object ID, according to EcoCyc",
        "flux-avg": "A floating point number in mmol/g DCW/h units",
        "flux-std": "A floating point number in mmol/g DCW/h units",
    }
    values = [flux_data[k] for k in columns]

    save_file(outdir, f"wcm_metabolic_reactions_{media_id}.tsv", columns, values)

    metadata_file = os.path.join(outdir, f"wcm_metadata_{media_id}.json")
    with open(metadata_file, "w") as f:
        print(f"Saving data to {metadata_file}")
        json.dump(ecocyc_metadata, f, indent=4)
