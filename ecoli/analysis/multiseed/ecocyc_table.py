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

import numpy as np
import polars as pl
from scipy.stats import pearsonr

from ecoli.analysis.template import get_field_metadata, named_idx, num_cells
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
    config_lf: pl.LazyFrame,
    history_lf: pl.LazyFrame,
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
    history_lf = history_lf.filter(pl.col("generation") >= IGNORE_FIRST_N_GENS)
    config_lf = config_lf.filter(pl.col("generation") >= IGNORE_FIRST_N_GENS)

    if config_lf.select("time").count().collect(streaming=True)["time"][0] == 0:
        print("Skipping analysis -- not enough simulations run.")
        return

    # Load tables and attributes for mRNAs
    mRNA_ids = get_field_metadata(
        config_lf, "listeners__rna_counts__mRNA_cistron_counts"
    )
    # mass_unit = get_field_metadata(config_lf, 'listeners__mass__dry_mass')
    # assert mass_unit == 'fg'
    gene_ids_rna_synth_prob = get_field_metadata(
        config_lf, "listeners__rna_synth_prob__gene_copy_number"
    )

    # Load tables and attributes for tRNAs and rRNAs
    bulk_ids = get_field_metadata(config_lf, "bulk")
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

    # Load attributes for complexes
    complex_ids = sim_data.process.complexation.ids_complexes

    # Read columns
    data = (
        history_lf.select(
            **{
                "time": "time",
                "lineage_seed": "lineage_seed",
                "generation": "generation",
                "agent_id": "agent_id",
                "gene_copy_numbers": "listeners__rna_synth_prob__gene_copy_number",
                "mRNA_counts": "listeners__rna_counts__mRNA_cistron_counts",
                "counts_to_molar": "listeners__enzyme_kinetics__counts_to_molar",
                "dry_masses": "listeners__mass__dry_mass",
                "full_ribosome_counts": "listeners__unique_molecule_counts__active_ribosome",
                "monomer_counts": "listeners__monomer_counts",
                **named_idx(
                    "bulk",
                    [f"complex__{i}" for i in complex_ids],
                    [bulk_ids.index(i) for i in complex_ids],
                ),
                **named_idx(
                    "bulk",
                    [f"uncharged_tRNA__{i}" for i in uncharged_tRNA_ids],
                    [bulk_ids.index(i) for i in uncharged_tRNA_ids],
                ),
                **named_idx(
                    "bulk",
                    [f"charged_tRNA__{i}" for i in charged_tRNA_ids],
                    [bulk_ids.index(i) for i in charged_tRNA_ids],
                ),
                **named_idx(
                    "bulk",
                    [f"rRNA__{i}" for i in rRNA_ids],
                    [bulk_ids.index(i) for i in rRNA_ids],
                ),
                **named_idx(
                    "bulk",
                    [f"ribosomal_subunit__{i}" for i in ribosomal_subunit_ids],
                    [bulk_ids.index(i) for i in ribosomal_subunit_ids],
                ),
            }
        )
        .with_columns(
            [
                pl.col("time")
                .rank("min")
                .over(["lineage_seed", "generation", "agent_id"])
                .alias("rank")
                # Filter out first timestep for each cell because counts_to_molar is 0
            ]
        )
        .filter(pl.col("rank") != 1)
        .drop("rank")
        .collect(streaming=True)
    )

    # Calculate statistics for gene copy numbers
    gene_copy_numbers = (
        data["gene_copy_numbers"]
        .list.to_struct(fields=gene_ids_rna_synth_prob)
        .struct.unnest()
    )
    gene_copy_numbers_avg = gene_copy_numbers.mean().transpose()
    gene_copy_numbers_std = gene_copy_numbers.std().transpose()

    # Add up to total counts of tRNAs and rRNAs
    tRNA_counts = data.select(
        [f"bulk__uncharged_tRNA__{i}" for i in uncharged_tRNA_ids]
    ) + data.select([f"bulk__charged_tRNA__{i}" for i in charged_tRNA_ids])
    rRNA_counts = data.select([f"bulk__rRNA__{i}" for i in rRNA_ids]).with_columns(
        **{
            f"bulk__rRNA__{rRNA_ids[0]}": pl.col(f"bulk__rRNA__{rRNA_ids[0]}")
            + data[f"bulk__ribosomal_subunit__{ribosomal_subunit_ids[0]}"],
            **{
                f"bulk__rRNA__{rRNA_id}": pl.col(f"bulk__rRNA__{rRNA_id}")
                + data[f"bulk__ribosomal_subunit__{ribosomal_subunit_ids[1]}"]
                for rRNA_id in rRNA_ids[1:]
            },
        }
    )
    rRNA_counts += data["full_ribosome_counts"]
    mRNA_counts = (
        data["mRNA_counts"]
        .list.to_struct(fields=["mRNA_counts__" + i for i in mRNA_ids])
        .struct.unnest()
    )

    # Calculate statistics for tRNAs, rRNAs, and mRNAs
    cistron_id_to_mw = {
        cistron_id: cistron_mw
        for (cistron_id, cistron_mw) in zip(
            sim_data.process.transcription.cistron_data["id"],
            sim_data.process.transcription.cistron_data["mw"].asNumber(
                units.fg / units.count
            ),
        )
    }
    rna_counts_avg: list[pl.DataFrame] = []
    rna_counts_std: list[pl.DataFrame] = []
    rna_counts_relative_to_total_rna_type_counts: list[pl.DataFrame] = []
    rna_conc_avg: list[pl.DataFrame] = []
    rna_conc_std: list[pl.DataFrame] = []
    rna_masses_avg: list[pl.DataFrame] = []
    rna_masses_relative_to_total_rna_type_mass: list[pl.DataFrame] = []
    for rna_ids, rna_counts in zip(
        [mRNA_ids, tRNA_cistron_ids, rRNA_cistron_ids],
        [mRNA_counts, tRNA_counts, rRNA_counts],
    ):
        rna_counts_avg.append(rna_counts.mean().transpose())
        rna_counts_std.append(rna_counts.std().transpose())
        rna_counts_relative_to_total_rna_type_counts.append(
            rna_counts_avg[-1] / rna_counts_avg[-1].sum()[0, 0]
        )
        rna_conc = rna_counts * data["counts_to_molar"]
        rna_conc_avg.append(rna_conc.mean().transpose())
        rna_conc_std.append(rna_conc.std().transpose())
        rna_mw = pl.Series([cistron_id_to_mw[cistron_id] for cistron_id in rna_ids])
        rna_masses_avg.append(rna_counts_avg[-1] * rna_mw)
        rna_masses_relative_to_total_rna_type_mass.append(
            rna_masses_avg[-1] / rna_masses_avg[-1].sum()[0, 0]
        )

    rna_counts_avg: pl.DataFrame = pl.concat(rna_counts_avg)
    rna_counts_std: pl.DataFrame = pl.concat(rna_counts_std)
    rna_counts_relative_to_total_rna_type_counts: pl.DataFrame = pl.concat(
        rna_counts_relative_to_total_rna_type_counts
    )
    rna_counts_relative_to_total_rna_counts = (
        rna_counts_avg / rna_counts_avg.sum()[0, 0]
    )
    rna_conc_avg: pl.DataFrame = pl.concat(rna_conc_avg)
    rna_conc_std: pl.DataFrame = pl.concat(rna_conc_std)
    rna_masses_relative_to_total_rna_type_mass: pl.DataFrame = pl.concat(
        rna_masses_relative_to_total_rna_type_mass
    )
    rna_masses_avg: pl.DataFrame = pl.concat(rna_masses_avg)
    rna_masses_relative_to_total_rna_mass = rna_masses_avg / rna_masses_avg.sum()[0, 0]
    rna_masses_relative_to_total_dcw = rna_masses_avg / data["dry_masses"].mean()
    rna_ids = mRNA_ids + tRNA_cistron_ids + rRNA_cistron_ids

    # Save RNA data in table
    cistron_id_to_gene_id = {
        cistron["id"]: cistron["gene_id"]
        for cistron in sim_data.process.transcription.cistron_data
    }
    gene_ids = [cistron_id_to_gene_id[x] for x in rna_ids]

    gene_id_to_index = {gene_id: i for i, gene_id in enumerate(gene_ids_rna_synth_prob)}
    reordering_indexes = np.array([gene_id_to_index[gene_id] for gene_id in gene_ids])
    assert np.all(np.array(gene_ids_rna_synth_prob)[reordering_indexes] == gene_ids)
    gene_copy_numbers_avg = gene_copy_numbers_avg[reordering_indexes]
    gene_copy_numbers_std = gene_copy_numbers_std[reordering_indexes]

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
    values = [
        gene_ids,
        gene_copy_numbers_avg,
        gene_copy_numbers_std,
        rna_counts_avg,
        rna_counts_std,
        rna_conc_avg,
        rna_conc_std,
        rna_counts_relative_to_total_rna_counts,
        rna_counts_relative_to_total_rna_type_counts,
        rna_masses_relative_to_total_rna_mass,
        rna_masses_relative_to_total_rna_type_mass,
        rna_masses_relative_to_total_dcw,
    ]

    save_file(outdir, f"wcm_rnas_{media_id}.tsv", columns, values)

    # Build dictionary for metadata
    ecocyc_metadata = {
        "git_hash": config_lf.select("data__git_hash").first().collect()[0, 0],
        "n_ignored_generations": IGNORE_FIRST_N_GENS,
        "n_total_generations": config_lf.select("data__generations")
        .first()
        .collect()[0, 0],
        "n_seeds": config_lf.select("data__n_init_sims").first().collect()[0, 0],
        "n_cells": num_cells(config_lf)[0, 0],
        "n_timesteps": len(data["counts_to_molar"]),
    }

    # Load tables and attributes for proteins
    monomer_ids = get_field_metadata(config_lf, "listeners__monomer_counts")
    monomer_mw = sim_data.getter.get_masses(monomer_ids).asNumber(
        units.fg / units.count
    )

    # Calculate derived protein values
    monomer_counts = data["monomer_counts"].list.to_struct().struct.unnest()
    monomer_counts_avg = monomer_counts.mean().transpose()
    monomer_counts_std = monomer_counts.std().transpose()
    monomer_conc = monomer_counts * data["counts_to_molar"]
    monomer_conc_avg = monomer_conc.mean().transpose()
    monomer_conc_std = monomer_conc.std().transpose()
    monomer_counts_relative_to_total_monomer_counts = (
        monomer_counts_avg / monomer_counts_avg.sum()[0, 0]
    )
    monomer_mass_avg = monomer_counts_avg * pl.Series(monomer_mw)
    monomer_mass_relative_to_total_monomer_mass = (
        monomer_mass_avg / monomer_mass_avg.sum()[0, 0]
    )
    monomer_mass_relative_to_total_dcw = monomer_mass_avg / data["dry_masses"].mean()

    # Save monomer data in table
    monomer_ecocyc_ids = [monomer[:-3] for monomer in monomer_ids]  # strip [*]

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
    values = [
        monomer_ecocyc_ids,
        monomer_counts_avg,
        monomer_counts_std,
        monomer_conc_avg,
        monomer_conc_std,
        monomer_counts_relative_to_total_monomer_counts,
        monomer_mass_relative_to_total_monomer_mass,
        monomer_mass_relative_to_total_dcw,
    ]

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
            monomer_counts_avg.filter(protein_val_exists)[:, 0],
            protein_counts_val[protein_val_exists],
        )

        ecocyc_metadata["protein_validation_r_squared"] = r**2

    save_file(outdir, f"wcm_monomers_{media_id}.tsv", columns, values)

    # Load attributes for complexes
    complex_mw = sim_data.getter.get_masses(complex_ids).asNumber(
        units.fg / units.count
    )

    # Calculate derived protein values
    complex_counts = data.select([f"bulk__complex__{i}" for i in complex_ids])
    complex_counts_avg = complex_counts.mean().transpose()
    complex_counts_std = complex_counts.std().transpose()
    complex_conc = complex_counts * data["counts_to_molar"]
    complex_conc_avg = complex_conc.mean().transpose()
    complex_conc_std = complex_conc.std().transpose()
    complex_mass_avg = complex_counts_avg * pl.Series(complex_mw)
    complex_mass_relative_to_total_protein_mass = (
        complex_mass_avg / monomer_mass_avg.sum()[0, 0]
    )
    complex_mass_relative_to_total_dcw = complex_mass_avg / data["dry_masses"].mean()

    # Save complex data in table
    complex_ecocyc_ids = [complex_id[:-3] for complex_id in complex_ids]  # strip [*]

    columns = {
        "id": "Object ID, according to EcoCyc",
        "complex-count-avg": "A floating point number",
        "complex-count-std": "A floating point number",
        "complex-concentration-avg": "A floating point number in mM units",
        "complex-concentration-std": "A floating point number in mM units",
        "relative-complex-mass-to-total-protein-mass": "A floating point number",
        "relative-complex-mass-to-total-cell-dry-mass": "A floating point number",
    }
    values = [
        complex_ecocyc_ids,
        complex_counts_avg,
        complex_counts_std,
        complex_conc_avg,
        complex_conc_std,
        complex_mass_relative_to_total_protein_mass,
        complex_mass_relative_to_total_dcw,
    ]

    save_file(outdir, f"wcm_complexes_{media_id}.tsv", columns, values)

    # Load attributes for metabolic fluxes
    cell_density = sim_data.constants.cell_density
    reaction_ids = sim_data.process.metabolism.base_reaction_ids

    # Read columns
    data = history_lf.select(
        **{
            "dry_mass": "listeners__mass__dry_mass",
            "cell_mass": "listeners__mass__cell_mass",
            "base_reaction_fluxes": "listeners__fba_results__base_reaction_fluxes",
        }
    ).collect(streaming=True)
    conversion_coeffs = (
        data["dry_mass"]
        / data["cell_mass"]
        * cell_density.asNumber(MASS_UNITS / VOLUME_UNITS)
    )

    # Calculate flux in units of mmol/g DCW/h
    fluxes: pl.DataFrame = (
        (COUNTS_UNITS / MASS_UNITS / TIME_UNITS)
        * (
            data["base_reaction_fluxes"].list.to_struct().struct.unnest()
            / conversion_coeffs
        )
    ).asNumber(units.mmol / units.g / units.h)

    # Calculate derived flux values
    fluxes_avg = fluxes.mean().transpose()
    fluxes_std = fluxes.std().transpose()

    columns = {
        "id": "Object ID, according to EcoCyc",
        "flux-avg": "A floating point number in mmol/g DCW/h units",
        "flux-std": "A floating point number in mmol/g DCW/h units",
    }
    values = [
        reaction_ids,
        fluxes_avg,
        fluxes_std,
    ]

    save_file(outdir, f"wcm_metabolic_reactions_{media_id}.tsv", columns, values)

    metadata_file = os.path.join(outdir, f"wcm_metadata_{media_id}.json")
    with open(metadata_file, "w") as f:
        print(f"Saving data to {metadata_file}")
        json.dump(ecocyc_metadata, f, indent=4)
