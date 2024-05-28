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
    gene_ids_rna_synth_prob = get_field_metadata(
        config_skip_n_gens, "listeners__rna_synth_prob__gene_copy_number"
    )

    # Load tables and attributes for tRNAs and rRNAs
    bulk_ids = get_field_metadata(config_skip_n_gens, "bulk")
    bulk_id_to_idx = {bulk_id: i + 1 for i, bulk_id in enumerate(bulk_ids)}
    uncharged_tRNA_ids = sim_data.process.transcription.uncharged_trna_names
    charged_tRNA_ids = sim_data.process.transcription.charged_trna_names
    tRNA_cistron_ids = [tRNA_id[:-3] for tRNA_id in uncharged_tRNA_ids]
    rRNA_ids = [
        sim_data.molecule_groups.s30_16s_rRNA[0],
        sim_data.molecule_groups.s50_23s_rRNA[0],
        sim_data.molecule_groups.s50_5s_rRNA[0],
    ]
    rRNA_cistron_ids = [rRNA_id[:-3] for rRNA_id in rRNA_ids]
    ribosomal_subunit_ids = [
        sim_data.molecule_ids.s30_full_complex,
        sim_data.molecule_ids.s50_full_complex,
    ]

    # Express calculations in SQL
    avg_gene_copy_numbers_expr = ", ".join(
        f"avg(gene_copy_numbers[{i + 1}]) AS \"gene-copy-number-avg_{gene_id}\""
        for i, gene_id in enumerate(gene_ids_rna_synth_prob)
    )
    std_gene_copy_numbers_expr = ", ".join(
        f"stddev(gene_copy_numbers[{i + 1}]) AS \"gene-copy-number-std_{gene_id}\""
        for i, gene_id in enumerate(gene_ids_rna_synth_prob)
    )
    # Name everything with gene ID for easy matching later
    cistron_id_to_gene_id = {
        cistron["id"]: cistron["gene_id"]
        for cistron in sim_data.process.transcription.cistron_data
    }
    mRNA_counts_expr = ", ".join(
        f"mRNA_counts[{i + 1}] AS \"{cistron_id_to_gene_id[mrna_id]}\""
        for i, mrna_id in enumerate(mRNA_ids)
    )
    # Add charged and uncharged tRNA counts
    tRNA_counts_expr = ", ".join(
        f"bulk[{bulk_id_to_idx[uncharged_id]}] + bulk[{bulk_id_to_idx[charged_id]}] AS \"{cistron_id_to_gene_id[tRNA_cistron_ids[i]]}\""
        for i, (uncharged_id, charged_id) in enumerate(
            zip(uncharged_tRNA_ids, charged_tRNA_ids)
        )
    )
    # Add subunit and full ribosome counts to rRNA counts
    rRNA_counts_expr = ", ".join([
        f"bulk[{bulk_id_to_idx[rRNA_ids[0]]}] + "
        f"bulk[{bulk_id_to_idx[ribosomal_subunit_ids[0]]}] + "
        f"full_ribosome_counts AS \"{cistron_id_to_gene_id[rRNA_cistron_ids[0]]}\""
    ] + [
        f"bulk[{bulk_id_to_idx[rrna_id]}] + "
        f"bulk[{bulk_id_to_idx[ribosomal_subunit_ids[1]]}] + "
        f"full_ribosome_counts AS \"{cistron_id_to_gene_id[rRNA_cistron_ids[i + 1]]}\""
        for i, rrna_id in enumerate(rRNA_ids[1:])
    ])
    gene_ids = [cistron_id_to_gene_id[rna_id]
        for rna_id in mRNA_ids + tRNA_cistron_ids + rRNA_cistron_ids]
    rna_counts_avg_expr = ", ".join(
        f"avg(\"{gene_id}\") AS \"rna-count-avg_{gene_id}\"" for gene_id in gene_ids)
    rna_counts_std_expr = ", ".join(
        f"stddev(\"{gene_id}\") AS \"rna-count-std_{gene_id}\"" for gene_id in gene_ids)
    rna_conc_avg_expr = ", ".join(
        f"avg(\"{gene_id}\" * counts_to_molar) AS \"rna-concentration-avg_{gene_id}\""
        for gene_id in gene_ids)
    rna_conc_std_expr = ", ".join(
        f"stddev(\"{gene_id}\" * counts_to_molar) AS \"rna-concentration-std_{gene_id}\""
        for gene_id in gene_ids)
    gene_id_to_mw = {
        gene_id: cistron_mw
        for (gene_id, cistron_mw) in zip(
            sim_data.process.transcription.cistron_data["gene_id"],
            sim_data.process.transcription.cistron_data["mw"].asNumber(
                units.fg / units.count
            ),
        )
    }
    rna_masses_avg_expr = ", ".join(
        f"avg(\"{gene_id}\" * {gene_id_to_mw[gene_id]}) AS \"rna-mass-avg_{gene_id}\""
        for gene_id in gene_ids)

    agg_prefixes = ["gene-copy-number-avg", "gene-copy-number-std",
                    "rna-count-avg", "rna-count-std", "rna-concentration-avg",
                    "rna-concentration-std", "rna-mass-avg"]
    # Each gene ID gets its own row in final view
    pivoted_table = ", ".join("unnest([" + ", ".join(
        f"\"{prefix}_{gene_id}\"" for gene_id in gene_ids) + f"]) AS \"{prefix}\""
        for prefix in agg_prefixes)
    escaped_gene_ids = "', '".join(gene_ids)
    rna_data = duckdb.sql(
        f"""WITH filter_first AS (
                SELECT
                    time, lineage_seed, generation, agent_id, bulk,
                    listeners__rna_synth_prob__gene_copy_number AS gene_copy_numbers,
                    listeners__rna_counts__mRNA_cistron_counts AS mRNA_counts,
                    listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                    listeners__mass__dry_mass AS dry_masses,
                    listeners__unique_molecule_counts__active_ribosome as full_ribosome_counts,
                FROM history_skip_n_gens
                -- Filter out first timestep for each cell because counts_to_molar is 0
                QUALIFY row_number() OVER (
                    PARTITION BY lineage_seed, generation, agent_id
                    ORDER BY time
                ) != 1
                ORDER BY time
            ),
            rna_counts AS (
                SELECT
                    gene_copy_numbers, counts_to_molar, dry_masses,
                    {mRNA_counts_expr}, {rRNA_counts_expr}, {tRNA_counts_expr}
                FROM filter_first
            ),
            averages AS (
                SELECT
                    {avg_gene_copy_numbers_expr}, {std_gene_copy_numbers_expr},
                    {rna_counts_avg_expr}, {rna_counts_std_expr},
                    {rna_conc_avg_expr}, {rna_conc_std_expr}, {rna_masses_avg_expr},
                    avg(dry_masses) AS \"dry-mass-avg\"
                FROM rna_counts
            )
            SELECT unnest(['{escaped_gene_ids}']) AS id, {pivoted_table}, \"dry-mass-avg\" from averages
        """).pl()

    # Calculate relative statistics
    rna_data["relative-rna-count-to-total-rna-counts"] = rna_data.select(
        pl.col("rna-count-avg") / pl.col("rna-count-avg").sum()
    )
    rna_data["relative-rna-mass-to-total-rna-mass"] = rna_data.select(
        pl.col("rna-mass-avg") / pl.col("rna-mass-avg").sum()
    )
    rna_data["relative-rna-mass-to-total-cell-dry-mass"] = rna_data.select(
        pl.col("rna-mass-avg") / pl.col("dry-mass-avg")
    )
    # Calculate relative statistics per RNA type
    rel_to_type_count = np.zeros(len(rna_data))
    rel_to_type_mass = np.zeros(len(rna_data))
    start_idx = 0
    for add_idx in [len(mRNA_ids), len(tRNA_cistron_ids), len(rRNA_cistron_ids)]:
        rel_to_type_count[
            start_idx:start_idx + add_idx] = rna_data.select(
            pl.col("rna-count-avg").slice(start_idx, add_idx) / 
            pl.col("rna-count-avg").slice(start_idx, add_idx).sum()
        )["rna-count-avg"]
        rel_to_type_mass[
            start_idx:start_idx + add_idx] = rna_data.select(
            pl.col("rna-mass-avg").slice(start_idx, add_idx) / 
            pl.col("rna-mass-avg").slice(start_idx, add_idx).sum()
        )["rna-mass-avg"]
        start_idx += add_idx
    rna_data["relative-rna-count-to-total-rna-type-counts"] = rel_to_type_count
    rna_data["relative-rna-mass-to-total-rna-type-mass"] = rel_to_type_mass

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
    monomer_ids = get_field_metadata(config_skip_n_gens, "listeners__monomer_counts")
    monomer_ecocyc_ids = [monomer[:-3] for monomer in monomer_ids]  # strip [*]
    escaped_monomer_ids = "', '".join(monomer_ecocyc_ids)
    monomer_mw = sim_data.getter.get_masses(monomer_ids).asNumber(
        units.fg / units.count
    )
    monomer_sim_data = sim_data.process.translation.monomer_data.struct_array
    monomer_to_gene_id = {
        monomer_id: cistron_id_to_gene_id[cistron_id]
        for cistron_id, monomer_id in
        zip(monomer_sim_data["cistron_id"], monomer_sim_data["id"])
    }
    gene_ids = [monomer_to_gene_id[monomer_id] for monomer_id in monomer_ids]

    avg_monomer_counts_expr = ", ".join(
        f"avg(monomer_counts[{i + 1}]) AS \"monomer-count-avg_{monomer_id}\""
        for i, monomer_id in enumerate(monomer_ids))
    std_monomer_counts_expr = ", ".join(
        f"stddev(monomer_counts[{i + 1}]) AS \"monomer-count-std_{monomer_id}\""
        for i, monomer_id in enumerate(monomer_ids))
    avg_monomer_conc_expr = ", ".join(
        f"avg(monomer_counts[{i + 1}] * counts_to_molar) AS \"monomer-conc-avg_{monomer_id}\""
        for i, monomer_id in enumerate(monomer_ids))
    std_monomer_conc_expr = ", ".join(
        f"stddev(monomer_counts[{i + 1}] * counts_to_molar) AS \"monomer-conc-std_{monomer_id}\""
        for i, monomer_id in enumerate(monomer_ids))
    avg_monomer_mass_expr = ", ".join(
        f"avg(monomer_counts[{i + 1}] * {monomer_mw[i]}) AS \"monomer-mass-avg_{monomer_id}\""
        for i, monomer_id in enumerate(monomer_ids))
    # Each monomer ID gets its own row in final view
    agg_prefixes = ["monomer-count-avg", "monomer-count-std",
                    "monomer-conc-avg", "monomer-conc-std"]
    pivoted_table = ", ".join("unnest([" + ", ".join(
        f"\"{prefix}_{monomer_id}\"" for monomer_id in monomer_ids) + f"]) AS \"{prefix}\""
        for prefix in agg_prefixes)
    monomer_data = duckdb.sql(
        f"""WITH filter_first AS (
                SELECT
                    time, lineage_seed, generation, agent_id,
                    listeners__mass__dry_mass AS dry_masses,
                    listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                    listeners__monomer_counts as monomer_counts,
                FROM history_skip_n_gens
                -- Filter out first timestep for each cell because counts_to_molar is 0
                QUALIFY row_number() OVER (
                    PARTITION BY lineage_seed, generation, agent_id
                    ORDER BY time
                ) != 1
                ORDER BY time
            )
            averages AS (
                SELECT
                    {avg_monomer_counts_expr}, {std_monomer_counts_expr},
                    {avg_monomer_conc_expr}, {std_monomer_conc_expr},
                    {avg_monomer_mass_expr},
                    avg(dry_masses) AS \"dry-mass-avg\"
                FROM rna_counts
            )
            SELECT unnest(['{escaped_monomer_ids}']) AS id, {pivoted_table},
                \"dry-mass-avg\" from averages
        """).pl()
    monomer_data = monomer_data.with_columns(**{
        "relative-protein-count-to-total-protein-counts": monomer_data.select(
            pl.col("monomer-count-avg") / pl.col("monomer-count-avg").sum()
        ),
        "relative-protein-mass-to-total-protein-mass": monomer_data.select(
            pl.col("monomer-mass-avg") / pl.col("monomer-mass-avg").sum()
        ),
        "relative-protein-mass-to-total-cell-dry-mass": monomer_data.select(
            pl.col("monomer-mass-avg") / pl.col("dry-mass-avg")
        ),
        "gene-ids": gene_ids,
    })
    monomer_data = monomer_data.join(rna_data.select(
        **{"gene-ids": "id", "rna-count-avg": "rna-count-avg"}),
        on="gene-ids")
    monomer_data = monomer_data.with_columns(**{
        "relative-protein-count-to-protein-rna-counts": monomer_data.select(
            pl.col("monomer-count-avg") / pl.col("rna-count-avg")
        )
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
            monomer_data["monomer-counts-avg"].filter(protein_val_exists)[:, 0],
            protein_counts_val[protein_val_exists],
        )

        ecocyc_metadata["protein_validation_r_squared"] = r**2

    save_file(outdir, f"wcm_monomers_{media_id}.tsv", columns, values)

    # Load attributes for complexes
    complex_ids = sim_data.process.complexation.ids_complexes
    complex_mw = sim_data.getter.get_masses(complex_ids).asNumber(
        units.fg / units.count
    )

    avg_complex_counts_expr = ", ".join(
        f"avg(bulk[{bulk_id_to_idx[complex_id]}]) AS \"complex-count-avg_{complex_id}\""
        for complex_id in complex_ids)
    std_complex_counts_expr = ", ".join(
        f"stddev(bulk[{bulk_id_to_idx[complex_id]}]]) AS \"complex-count-std_{complex_id}\""
        for complex_id in complex_ids)
    avg_complex_conc_expr = ", ".join(
        f"avg(bulk[{bulk_id_to_idx[complex_id]}] * counts_to_molar) AS \"complex-conc-avg_{complex_id}\""
        for complex_id in complex_ids)
    std_complex_conc_expr = ", ".join(
        f"stddev(bulk[{bulk_id_to_idx[complex_id]}] * counts_to_molar) AS \"complex-conc-std_{complex_id}\""
        for complex_id in complex_ids)
    avg_complex_mass_expr = ", ".join(
        f"avg(bulk[{bulk_id_to_idx[complex_id]}] * {complex_mw[i]}) AS \"complex-mass-avg_{complex_id}\""
        for i, complex_id in enumerate(complex_ids))
    # Each complex ID gets its own row in final view
    agg_prefixes = ["complex-count-avg", "complex-count-std",
                    "complex-conc-avg", "complex-conc-std"]
    pivoted_table = ", ".join("unnest([" + ", ".join(
        f"\"{prefix}_{complex_id}\"" for complex_id in complex_ids) + f"]) AS \"{prefix}\""
        for prefix in agg_prefixes)
    complex_data = duckdb.sql(
        f"""WITH filter_first AS (
                SELECT
                    time, lineage_seed, generation, agent_id, bulk,
                    listeners__mass__dry_mass AS dry_masses,
                    listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar,
                FROM history_skip_n_gens
                -- Filter out first timestep for each cell because counts_to_molar is 0
                QUALIFY row_number() OVER (
                    PARTITION BY lineage_seed, generation, agent_id
                    ORDER BY time
                ) != 1
                ORDER BY time
            )
            averages AS (
                SELECT
                    {avg_complex_counts_expr}, {std_complex_counts_expr},
                    {avg_complex_conc_expr}, {std_complex_conc_expr},
                    {avg_complex_mass_expr},
                    avg(dry_masses) AS \"dry-mass-avg\"
                FROM rna_counts
            )
            SELECT {pivoted_table}, \"dry-mass-avg\" from averages
        """).pl()

    # Calculate derived protein values
    complex_data = complex_data.with_columns(**{
        "relative-complex-mass-to-total-protein-mass": complex_data.select(
            pl.col("complex-mass-avg") / pl.col("complex-mass-avg").sum()
        ),
        "relative-complex-mass-to-total-cell-dry-mass": complex_data.select(
            pl.col("complex-mass-avg") / pl.col("dry-mass-avg")
        ),
        "id": [complex_id[:-3] for complex_id in complex_ids],  # strip [*]
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

    # Read columns
    avg_fluxes_expr = ", ".join(
        f"avg(listeners__fba_results__base_reaction_fluxes[{i}]) AS \"flux-avg_{flux_id}\""
        for i, flux_id in enumerate(reaction_ids))
    std_fluxes_expr = ", ".join(
        f"stddev(listeners__fba_results__base_reaction_fluxes[{i}]]) AS \"flux-std_{flux_id}\""
        for i, flux_id in enumerate(reaction_ids))
    # Each reaction ID gets its own row in final view
    agg_prefixes = ["flux-avg", "flux-std"]
    pivoted_table = ", ".join("unnest([" + ", ".join(
        f"\"{prefix}_{reaction_id}\" / conversion_coeff"
        for reaction_id in reaction_ids) + f"]) AS \"{prefix}\""
        for prefix in agg_prefixes)
    data = duckdb.sql(
        f"""
        WITH cte_1 AS (
            SELECT listeners__mass__dry_mass / 
                listeners__mass__cell_mass * {cell_density} AS conversion_coeffs,
                {avg_fluxes_expr}, {std_fluxes_expr}
            FROM history_skip_n_gens
        )
        SELECT {pivoted_table} FROM cte_1
        """).pl()
    
    data = data.with_columns(**{
        "id": reaction_ids,
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
