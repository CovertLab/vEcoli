import os
from typing import Any, cast

from duckdb import DuckDBPyConnection
import polars as pl
import pickle
import matplotlib.pyplot as plt
import numpy as np

from ecoli.library.parquet_emitter import (
    num_cells,
    read_stacked_columns,
    get_field_metadata,
    named_idx,
    open_arbitrary_sim_data,
    ndidx_to_duckdb_expr,
)
from wholecell.utils import units

def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_name: str,
):
    assert (
        num_cells(conn, config_sql) == 1
    ), "Mass fraction summary plot requires single-cell data."
    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)

    # Metabolism fluxes
    purF_kinetic_rxn_id = 'PRPPAMIDOTRANS-RXN (reverse)'

    purF_actual_flux_name = 'purF actual flux'
    actual_flux_dict = {
        rxn: i
        for i, rxn in enumerate(
            get_field_metadata(conn, config_sql, "listeners__enzyme_kinetics__actual_fluxes")
        )
    }
    purF_actual_flux_idx = cast(int, actual_flux_dict[purF_kinetic_rxn_id])
    purF_actual_flux = named_idx(
        "listeners__enzyme_kinetics__actual_fluxes", [purF_actual_flux_name], [purF_actual_flux_idx]
    )

    purF_kinetic_target_name = 'purF kinetic target'
    kinetic_target_dict = {
        rxn: i
        for i, rxn in enumerate(
            get_field_metadata(conn, config_sql, "listeners__enzyme_kinetics__target_fluxes")
        )
    }
    purF_kinetic_target_idx = cast(int, kinetic_target_dict[purF_kinetic_rxn_id])
    purF_kinetic_flux = named_idx(
        "listeners__enzyme_kinetics__target_fluxes", [purF_kinetic_target_name], [purF_kinetic_target_idx]
    )

    # Bulk molecule counts
    bulk_molecule_ids = ['ATP[c]', 'GTP[c]', 'ADP[c]', 'GDP[c]', 'AMP[c]', 'GMP[c]', 'IMP[c]', 'PRPP[c]',
                         'GLN[c]', 'PRPPAMIDOTRANS-CPLX[c]']
    bulk_dict = {
        mol: i
        for i, mol in enumerate(
            get_field_metadata(conn, config_sql, "bulk")
        )
    }
    bulk_idxs = [cast(int, bulk_dict[x]) for x in bulk_molecule_ids]
    bulk_molecules = named_idx(
        "bulk", bulk_molecule_ids, bulk_idxs
    )

    # mRNA counts
    cistron_ids = ['EG10794_RNA']
    cistron_dict = {
        cistron: i
        for i, cistron in enumerate(
            get_field_metadata(conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts")
        )
    }
    cistron_idxs = [cast(int, cistron_dict[x]) for x in cistron_ids]
    cistrons = named_idx(
        "listeners__rna_counts__mRNA_cistron_counts", cistron_ids, cistron_idxs
    )

    # Extract data
    metab_data = read_stacked_columns(
        history_sql,
        ['listeners__enzyme_kinetics__target_fluxes', 'listeners__enzyme_kinetics__actual_fluxes',
         'bulk', 'listeners__rna_counts__mRNA_cistron_counts'],
        [purF_kinetic_flux, purF_actual_flux, bulk_molecules, cistrons],
        conn=conn,
        remove_first=True
    )
    metab_data = pl.DataFrame(metab_data)

    # Mass data
    mass_columns = {
        "Protein": "listeners__mass__protein_mass",
        "tRNA": "listeners__mass__tRna_mass",
        "rRNA": "listeners__mass__rRna_mass",
        "mRNA": "listeners__mass__mRna_mass",
        "DNA": "listeners__mass__dna_mass",
        "Small Mol.s": "listeners__mass__smallMolecule_mass",
        "Dry": "listeners__mass__dry_mass",
        "Counts_to_molar": "listeners__enzyme_kinetics__counts_to_molar"
    }
    mass_data = read_stacked_columns(
        history_sql, list(mass_columns.values()), conn=conn,
        remove_first=True
    )
    mass_data = pl.DataFrame(mass_data)
    fractions = {
        k: (mass_data[v] / mass_data["listeners__mass__dry_mass"]).mean()
        for k, v in mass_columns.items()
    }
    new_columns = {
        "Time": mass_data["time"] - mass_data["time"].min(),
        **{
            k: mass_data[v]
            for k, v in mass_columns.items()
        },
    }
    mass_data = pl.DataFrame(new_columns)

    # Make plots
    num_plots = 19
    fig, axs = plt.subplots(num_plots, figsize=(60, 15*num_plots))
    purF_kinetic_target_data = metab_data[purF_kinetic_target_name].to_numpy()
    purF_actual_flux_data = metab_data[purF_actual_flux_name].to_numpy()
    purF_mRNA_counts = metab_data['EG10794_RNA'].to_numpy()
    purF_protein_counts = metab_data['PRPPAMIDOTRANS-CPLX[c]'].to_numpy()

    counts_to_molar = mass_data["Counts_to_molar"]
    purF_kinetic_flux_counts = purF_kinetic_target_data / counts_to_molar
    purF_actual_flux_counts = purF_actual_flux_data / counts_to_molar
    purF_mRNA_conc = purF_mRNA_counts * counts_to_molar
    purF_protein_conc = purF_protein_counts * counts_to_molar

    purF_v = purF_kinetic_flux_counts / purF_protein_counts

    axs[0].plot(metab_data["time"], purF_kinetic_target_data)
    axs[0].set_title("Kinetic conc flux")
    axs[1].plot(metab_data["time"], purF_actual_flux_data)
    axs[1].set_title("Actual conc flux")
    axs[2].plot(metab_data["time"], purF_kinetic_flux_counts)
    axs[2].set_title("Kinetic counts flux")
    axs[3].plot(metab_data["time"], purF_actual_flux_counts)
    axs[3].set_title("Actual counts flux")
    axs[4].plot(metab_data["time"], purF_mRNA_conc)
    axs[4].set_title("purF cistron conc")
    axs[5].plot(metab_data["time"], purF_mRNA_counts)
    axs[5].set_title("purF cistron counts")
    axs[6].plot(metab_data["time"], purF_protein_conc)
    axs[6].set_title("purF protein conc")
    axs[7].plot(metab_data["time"], purF_protein_counts)
    axs[7].set_title("purF protein counts")
    axs[8].plot(metab_data["time"], purF_v)
    axs[8].set_title("purF kinetic enzyme velocity")

    axs[9].plot(mass_data["Time"], mass_data["rRNA"])
    axs[9].set_title("rRNA mass")
    for i, x in enumerate(["ATP[c]", "GTP[c]", "ADP[c]",
              "GDP[c]", "AMP[c]", "GMP[c]", "IMP[c]", "PRPP[c]", "GLN[c]"]):
        axs[10+i].plot(metab_data["time"], metab_data[x] * counts_to_molar)
        axs[10+i].set_title(x+" conc")

    #axs[0].legend()

    # axs[1].plot(metab_data["time"], purF_kinetic_target_data / np.mean(purF_kinetic_target_data), label='target')
    # axs[1].plot(metab_data["time"], purF_actual_flux_data / np.mean(purF_actual_flux_data), label='actual')
    # axs[1].plot([0, 2000], [1, 1], label="mean")
    # axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "purine_metab_analysis.png"))
    plt.close('all')
