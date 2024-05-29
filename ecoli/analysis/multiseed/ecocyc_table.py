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

import duckdb
import numpy as np
import polars as pl
from scipy.stats import pearsonr

from ecoli.analysis.template import get_config_value, get_field_metadata, num_cells
from ecoli.processes.metabolism import (
    COUNTS_UNITS,
    MASS_UNITS,
    TIME_UNITS,
    VOLUME_UNITS,
)
from wholecell.utils import units

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli

IGNORE_FIRST_N_GENS = 0

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
    configuration: duckdb.DuckDBPyRelation,
    history: duckdb.DuckDBPyRelation,
    sim_data_paths: list[str],
    validation_data_paths: list[str],
    outdir: str,
):
    with open(sim_data_paths[0], "rb") as f:
        sim_data: "SimulationDataEcoli" = pickle.load(f)
    with open(validation_data_paths[0], "rb") as f:
        validation_data = pickle.load(f)

    media_name = sim_data.conditions[sim_data.condition]["nutrients"]
    media_id = MEDIA_NAME_TO_ID.get(media_name, media_name)

    # Ignore first N generations
    history_skip_n_gens = duckdb.sql(
        f"SELECT * FROM history WHERE generation >= {IGNORE_FIRST_N_GENS}")
    config_skip_n_gens = duckdb.sql(
        f"SELECT * FROM configuration WHERE generation >= {IGNORE_FIRST_N_GENS}")

    if duckdb.sql("SELECT count(time) AS m FROM config_skip_n_gens").arrow()[
        'm'][0].as_py() == 0:
        print("Skipping analysis -- not enough simulations run.")
        return

    # Load tables and attributes for mRNAs
    mRNA_ids = get_field_metadata(
        config_skip_n_gens, "listeners__rna_counts__mRNA_cistron_counts"
    )
    # mass_unit = get_field_metadata(config_lf, 'listeners__mass__dry_mass')
    # assert mass_unit == 'fg'

    # Load tables and attributes for tRNAs and rRNAs
    bulk_ids = get_field_metadata(config_skip_n_gens, "bulk")
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

    # Reorder gene copy number listener to match RNA counts
    cistron_id_to_gene_id = {
        cistron["id"]: cistron["gene_id"]
        for cistron in sim_data.process.transcription.cistron_data
    }
    gene_ids = [cistron_id_to_gene_id[rna_id]
        for rna_id in mRNA_ids + tRNA_cistron_ids + rRNA_cistron_ids]
    gene_ids_rna_synth_prob = get_field_metadata(
        config_skip_n_gens, "listeners__rna_synth_prob__gene_copy_number"
    )
    gene_id_to_idx = {
        gene: i + 1 for i, gene in enumerate(gene_ids_rna_synth_prob)
    }
    gene_to_rna_order_idx = pl.DataFrame({
        "rna_idx": [gene_id_to_idx[gene] for gene in gene_ids],
        "row_num": list(range(1, len(gene_ids) + 1))
    })
    rna_data = duckdb.sql(
        f"""
        WITH filter_first AS (
            SELECT
                -- Create RNA counts list of mRNAs, tRNAs, and rRNAs (in
                -- that order), then unnest to perform aggregations (e.g. mean).
                unnest(list_concat(
                    listeners__rna_counts__mRNA_cistron_counts,
                    list_concat(
                        -- tRNA = charged + uncharged
                        [
                            trna[1] + trna[2]
                            FOR trna IN list_zip(
                                list_select(bulk, {charged_tRNA_idx}),
                                list_select(bulk, {uncharged_tRNA_idx})
                            )
                        ],
                        list_concat(
                            -- First rRNA = bulk + active ribosome + small subunit
                            [
                                bulk[{rRNA_idx[0]}] + 
                                listeners__unique_molecule_counts__active_ribosome +
                                bulk[{ribo_subunit_idx[0]}]
                            ],
                            -- Remaining rRNAs = bulk + active ribosome + large subunit
                            [
                                rrna_count +
                                listeners__unique_molecule_counts__active_ribosome +
                                bulk[{ribo_subunit_idx[1]}]
                                FOR rrna_count IN list_select(bulk, {rRNA_idx[1:]})
                            ]
                        )  
                    )
                )) AS rna_counts,
                -- Unnest gene copy number to perform aggregations (e.g. mean, stddev)
                unnest(listeners__rna_synth_prob__gene_copy_number) AS gene_copy_numbers,
                -- Track list index for each unnested row for grouped aggregations
                generate_subscripts(listeners__rna_synth_prob__gene_copy_number, 1) as gene_idx,
                listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                listeners__mass__dry_mass AS dry_masses
            FROM history_skip_n_gens
            -- Filter out first timestep for each cell because counts_to_molar is 0
            QUALIFY row_number() OVER (
                PARTITION BY lineage_seed, generation, agent_id
                ORDER BY time
            ) != 1
        ),
        -- Materialize aggregation CTE so can reuse without recomputing
        aggregates AS MATERIALIZED (
            SELECT
                avg(rna_counts) AS "rna-count-avg",
                stddev(rna_counts) AS "rna-count-std",
                avg(rna_counts * counts_to_molar) AS "rna-concentration-avg",
                stddev(rna_counts * counts_to_molar) AS "rna-concentration-std",
                avg(dry_masses) AS "dry-masses-avg",
                avg(gene_copy_numbers) AS "gene-copy-number-avg",
                stddev(gene_copy_numbers) AS "gene-copy-number-std"
            FROM filter_first
            GROUP BY gene_idx
            ORDER BY gene_idx
        )
        SELECT
            "rna-count-avg", "rna-count-std", "rna-concentration-avg",
            "rna-concentration-std", "dry-masses-avg", "gene-copy-number-avg",
            "gene-copy-number-std"
        -- Sort gene copy number aggregates by joining with gene_to_rna_order_idx,
        -- a Polars DataFrame that maps the ordering of genes in the gene copy number
        -- listener to the ordering of genes in the RNA count aggregates.
        FROM (
            SELECT
                "rna-count-avg", "rna-count-std", "rna-concentration-avg",
                "rna-concentration-std", "dry-masses-avg",
                row_number() OVER () AS rna_idx
            FROM aggregates
        )
        JOIN (
            SELECT
                "gene-copy-number-avg", "gene-copy-number-std", rna_idx
            FROM gene_to_rna_order_idx
            JOIN (
                SELECT
                    "gene-copy-number-avg", "gene-copy-number-std", 
                    row_number() OVER () AS row_num
                FROM aggregates
            )
            USING (row_num)
        )
        USING (rna_idx)
        ORDER BY rna_idx
        """).pl()

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
        rna_slice_mw = rna_mw[start_idx:start_idx + add_idx]
        rel_to_type_count[
            start_idx:start_idx + add_idx] = rna_data.select(
            pl.col("rna-count-avg").slice(start_idx, add_idx) / 
            pl.col("rna-count-avg").slice(start_idx, add_idx).sum()
        )["rna-count-avg"]
        rel_to_type_mass[
            start_idx:start_idx + add_idx] = rna_data.select(
            pl.col("rna-count-avg").slice(start_idx, add_idx) * rna_slice_mw /
            (pl.col("rna-count-avg").slice(start_idx, add_idx) * rna_slice_mw).sum()
        )["rna-count-avg"]
        start_idx += add_idx
    rna_data = rna_data.with_columns(**{
        "relative-rna-count-to-total-rna-type-counts": rel_to_type_count,
        "relative-rna-mass-to-total-rna-type-mass": rel_to_type_mass,
        "relative-rna-count-to-total-rna-counts": rna_data.select(
            pl.col("rna-count-avg") / pl.col("rna-count-avg").sum()
        )["rna-count-avg"],
        "relative-rna-mass-to-total-rna-mass": rna_data.select(
            pl.col("rna-count-avg") * rna_mw / 
            (pl.col("rna-count-avg") * rna_mw).sum()
        )["rna-count-avg"],
        "relative-rna-mass-to-total-cell-dry-mass": rna_data.select(
            pl.col("rna-count-avg") * rna_mw / 
            pl.col("dry-masses-avg").sum()
        )["rna-count-avg"],
        "id": pl.Series(gene_ids),
    })

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
        "git_hash": get_config_value(config_skip_n_gens, "git_hash"),
        "n_ignored_generations": IGNORE_FIRST_N_GENS,
        "n_total_generations": get_config_value(config_skip_n_gens, "generations"),
        "n_seeds": get_config_value(config_skip_n_gens, "n_init_sims"),
        "n_cells": num_cells(config_skip_n_gens),
        "n_timesteps": len(rna_data),
    }

    # Load tables and attributes for proteins
    monomer_data = duckdb.sql(
        """
        WITH filter_first AS (
            SELECT
                listeners__mass__dry_mass AS dry_masses,
                listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                -- Unnest monomer counts to calculate aggregations (e.g. mean)
                unnest(listeners__monomer_counts) AS monomer_counts,
                -- Track list index for each unnested row for grouped aggregations
                generate_subscripts(listeners__monomer_counts, 1) AS monomer_idx,
            FROM history_skip_n_gens
            -- Filter out first timestep for each cell because counts_to_molar is 0
            QUALIFY row_number() OVER (
                PARTITION BY lineage_seed, generation, agent_id
                ORDER BY time
            ) != 1
        )
        SELECT
            avg(monomer_counts) AS "protein-count-avg",
            stddev(monomer_counts) AS "protein-count-std",
            avg(monomer_counts * counts_to_molar) AS "protein-concentration-avg",
            stddev(monomer_counts * counts_to_molar) AS "protein-concentration-std",
            avg(dry_masses) AS "dry-masses-avg"
        FROM filter_first
        GROUP BY monomer_idx
        ORDER BY monomer_idx
        """).pl()
    monomer_ids = get_field_metadata(config_skip_n_gens, "listeners__monomer_counts")
    monomer_ecocyc_ids = [monomer[:-3] for monomer in monomer_ids]  # strip [*]
    monomer_mw = sim_data.getter.get_masses(monomer_ids).asNumber(
        units.fg / units.count
    )
    monomer_sim_data = sim_data.process.translation.monomer_data.struct_array
    monomer_to_gene_id = {
        monomer_id: cistron_id_to_gene_id[cistron_id]
        for cistron_id, monomer_id in
        zip(monomer_sim_data["cistron_id"], monomer_sim_data["id"])
    }
    # Calculate relative statistics
    monomer_data = monomer_data.with_columns(**{
        "relative-protein-count-to-total-protein-counts": monomer_data.select(
            pl.col("protein-count-avg") / pl.col("protein-count-avg").sum()
        )["protein-count-avg"],
        "relative-protein-mass-to-total-protein-mass": monomer_data.select(
            pl.col("protein-count-avg") * monomer_mw /
            (pl.col("protein-count-avg") * monomer_mw).sum()
        )["protein-count-avg"],
        "relative-protein-mass-to-total-cell-dry-mass": monomer_data.select(
            pl.col("protein-count-avg") * monomer_mw / pl.col("dry-masses-avg")
        )["protein-count-avg"],
        "gene-id": pl.Series([monomer_to_gene_id[monomer_id]
                              for monomer_id in monomer_ids]),
        "id": pl.Series(monomer_ecocyc_ids),
    })
    monomer_data = monomer_data.join(rna_data.select(
        **{"gene-id": "id", "rna-count-avg": "rna-count-avg"}),
        on="gene-id")
    monomer_data = monomer_data.with_columns(**{
        "relative-protein-count-to-protein-rna-counts": monomer_data.select(
            pl.col("protein-count-avg") / pl.col("rna-count-avg")
        )["protein-count-avg"]
    })

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
        values.append(protein_counts_val)

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
    complex_data = duckdb.sql(
        f"""
        WITH filter_first AS (
            SELECT
                -- Unnest complex counts to calculate aggregations (e.g. mean)
                unnest(list_select(bulk, {complex_idx})) AS complex_counts,
                -- Track list index for each unnested row for grouped aggregations
                generate_subscripts(list_select(bulk, {complex_idx}), 1) AS complex_idx,
                listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                listeners__mass__dry_mass AS dry_masses,
            FROM history_skip_n_gens
            -- Filter out first timestep for each cell because counts_to_molar is 0
            QUALIFY row_number() OVER (
                PARTITION BY lineage_seed, generation, agent_id
                ORDER BY time
            ) != 1
        )
        SELECT
            avg(complex_counts) AS "complex-count-avg",
            stddev(complex_counts) AS "complex-count-std",
            avg(complex_counts * counts_to_molar) AS "complex-concentration-avg",
            stddev(complex_counts * counts_to_molar) AS "complex-concentration-std",
            avg(dry_masses) AS "dry-masses-avg"
        FROM filter_first
        GROUP BY complex_idx
        ORDER BY complex_idx
        """).pl()

    # Calculate derived protein values
    complex_mw = sim_data.getter.get_masses(complex_ids).asNumber(
        units.fg / units.count
    )
    complex_data = complex_data.with_columns(**{
        "relative-complex-mass-to-total-protein-mass": complex_data.select(
            pl.col("complex-count-avg") * complex_mw /
            (pl.col("complex-count-avg") * complex_mw).sum()
        )["complex-count-avg"],
        "relative-complex-mass-to-total-cell-dry-mass": complex_data.select(
            pl.col("complex-count-avg") * complex_mw / pl.col("dry-masses-avg")
        )["complex-count-avg"],
        "id": pl.Series([complex_id[:-3] for complex_id in complex_ids]),  # strip [*]
    })
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
    data = duckdb.sql(
        f"""
        WITH unnest_fluxes AS (
            SELECT listeners__mass__dry_mass / 
                listeners__mass__cell_mass * {cell_density} AS conversion_coeffs,
                -- Unnest monomer counts to calculate aggregations (e.g. mean)
                unnest(listeners__fba_results__base_reaction_fluxes) as fluxes,
                generate_subscripts(listeners__fba_results__base_reaction_fluxes, 1) AS idx,
            FROM history_skip_n_gens
        )
        SELECT 
            avg(fluxes / conversion_coeffs) AS "flux-avg",
            stddev(fluxes / conversion_coeffs) AS "flux-std"
        FROM unnest_fluxes
        GROUP BY idx
        ORDER BY idx
        """).pl()
    
    data = data.with_columns(**{
        "id": pl.Series(reaction_ids),
        "flux-avg": ((COUNTS_UNITS / MASS_UNITS / TIME_UNITS)
            * pl.col("flux-avg")).asNumber(units.mmol / units.g / units.h),
        "flux-std": ((COUNTS_UNITS / MASS_UNITS / TIME_UNITS)
            * pl.col("flux-std")).asNumber(units.mmol / units.g / units.h)
    })

    columns = {
        "id": "Object ID, according to EcoCyc",
        "flux-avg": "A floating point number in mmol/g DCW/h units",
        "flux-std": "A floating point number in mmol/g DCW/h units",
    }
    values = [data[k] for k in columns]

    save_file(outdir, f"wcm_metabolic_reactions_{media_id}.tsv", columns, values)

    metadata_file = os.path.join(outdir, f"wcm_metadata_{media_id}.json")
    with open(metadata_file, "w") as f:
        print(f"Saving data to {metadata_file}")
        json.dump(ecocyc_metadata, f, indent=4)
