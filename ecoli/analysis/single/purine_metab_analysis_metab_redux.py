import os
from typing import Any, cast

from duckdb import DuckDBPyConnection
import polars as pl
import matplotlib.pyplot as plt

from ecoli.library.parquet_emitter import (
    num_cells,
    read_stacked_columns,
    get_field_metadata,
    named_idx,
)


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_name: str,
):
    assert num_cells(conn, config_sql) == 1, (
        "Mass fraction summary plot requires single-cell data."
    )

    # purF reaction
    purF_kinetic_rxn_id = "PRPPAMIDOTRANS-RXN (reverse)"
    purF_actual_flux_name = "purF actual flux"
    actual_flux_dict = {
        rxn: i
        for i, rxn in enumerate(
            get_field_metadata(
                conn, config_sql, "listeners__fba_results__estimated_fluxes"
            )
        )
    }
    purF_actual_flux_idx = cast(int, actual_flux_dict[purF_kinetic_rxn_id])
    purF_actual_flux = named_idx(
        "listeners__fba_results__estimated_fluxes",
        [purF_actual_flux_name],
        [purF_actual_flux_idx],
    )

    purF_kinetic_target_name = "purF kinetic target"
    kinetic_target_dict = {
        rxn: i
        for i, rxn in enumerate(
            get_field_metadata(
                conn, config_sql, "listeners__fba_results__target_kinetic_fluxes"
            )
        )
    }
    purF_kinetic_target_idx = cast(int, kinetic_target_dict[purF_kinetic_rxn_id])
    purF_kinetic_flux = named_idx(
        "listeners__fba_results__target_kinetic_fluxes",
        [purF_kinetic_target_name],
        [purF_kinetic_target_idx],
    )

    # Bulk molecule counts
    bulk_molecule_ids = [
        "ATP[c]",
        "GTP[c]",
        "ADP[c]",
        "GDP[c]",
        "AMP[c]",
        "GMP[c]",
        "IMP[c]",
        "PRPP[c]",
        "PRPPAMIDOTRANS-CPLX[c]",
        "GLN[c]",
    ]
    bulk_dict = {
        mol: i for i, mol in enumerate(get_field_metadata(conn, config_sql, "bulk"))
    }
    bulk_idxs = [cast(int, bulk_dict[x]) for x in bulk_molecule_ids]
    bulk_molecules = named_idx("bulk", bulk_molecule_ids, bulk_idxs)

    # mRNA counts
    cistron_ids = ["EG10794_RNA"]
    cistron_dict = {
        cistron: i
        for i, cistron in enumerate(
            get_field_metadata(
                conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts"
            )
        )
    }
    cistron_idxs = [cast(int, cistron_dict[x]) for x in cistron_ids]
    cistrons = named_idx(
        "listeners__rna_counts__mRNA_cistron_counts", cistron_ids, cistron_idxs
    )

    # Extract data
    metab_data = read_stacked_columns(
        history_sql,
        [purF_kinetic_flux, purF_actual_flux, bulk_molecules, cistrons],
        conn=conn,
    )
    metab_data = pl.DataFrame(metab_data)

    # Mass fractions
    mass_columns = {
        "Protein": "listeners__mass__protein_mass",
        "tRNA": "listeners__mass__tRna_mass",
        "rRNA": "listeners__mass__rRna_mass",
        "mRNA": "listeners__mass__mRna_mass",
        "DNA": "listeners__mass__dna_mass",
        "Small Mol.s": "listeners__mass__smallMolecule_mass",
        "Dry": "listeners__mass__dry_mass",
        "Counts_to_molar": "listeners__enzyme_kinetics__counts_to_molar",
    }
    mass_data = read_stacked_columns(
        history_sql, list(mass_columns.values()), conn=conn
    )
    mass_data = pl.DataFrame(mass_data)
    new_columns = {
        "Time (min)": (mass_data["time"] - mass_data["time"].min()) / 60,
        **{k: mass_data[v] / mass_data[v][0] for k, v in mass_columns.items()},
    }
    mass_fold_change = pl.DataFrame(new_columns)

    # TODO: purF mRNA and protein levels,
    # then also get the enzyme saturation (i.e. kinetic flux / enzyme conc),
    #
    num_plots = 19
    fig, axs = plt.subplots(num_plots, figsize=(60, 15 * num_plots))
    purF_kinetic_target_data = metab_data[purF_kinetic_target_name].to_numpy()
    purF_actual_flux_data = metab_data[purF_actual_flux_name].to_numpy()
    purF_mRNA_counts = metab_data["EG10794_RNA"].to_numpy()
    purF_protein_counts = metab_data["PRPPAMIDOTRANS-CPLX[c]"].to_numpy()

    relative_dry_mass = mass_fold_change["Dry"].to_numpy()
    purF_kinetic_flux_conc = purF_kinetic_target_data / relative_dry_mass
    purF_actual_flux_conc = purF_actual_flux_data / relative_dry_mass
    purF_mRNA_conc = purF_mRNA_counts / relative_dry_mass
    purF_protein_conc = purF_protein_counts / relative_dry_mass

    purF_v = purF_kinetic_target_data / purF_protein_counts

    axs[0].plot(metab_data["time"], purF_kinetic_target_data)
    axs[0].set_title("Kinetic count flux")
    axs[1].plot(metab_data["time"], purF_actual_flux_data)
    axs[1].set_title("Actual count flux")
    axs[2].plot(metab_data["time"], purF_kinetic_flux_conc)
    axs[2].set_title("Kinetic conc flux")
    axs[3].plot(metab_data["time"], purF_actual_flux_conc)
    axs[3].set_title("Actual conc flux")
    axs[4].plot(metab_data["time"], purF_mRNA_conc)
    axs[4].set_title("purF cistron conc")
    axs[5].plot(metab_data["time"], purF_protein_conc)
    axs[5].set_title("purF protein conc")
    axs[6].plot(metab_data["time"], purF_mRNA_counts)
    axs[6].set_title("purF cistron counts")
    axs[7].plot(metab_data["time"], purF_protein_counts)
    axs[7].set_title("purF protein counts")
    axs[8].plot(metab_data["time"], purF_v)
    axs[8].set_title("purF per-enzyme rate")

    axs[9].plot(metab_data["time"], mass_fold_change["rRNA"])
    axs[9].set_title("rRNA mass fold change")
    for i, x in enumerate(
        [
            "ATP[c]",
            "GTP[c]",
            "ADP[c]",
            "GDP[c]",
            "AMP[c]",
            "GMP[c]",
            "IMP[c]",
            "PRPP[c]",
            "GLN[c]",
        ]
    ):
        axs[10 + i].plot(metab_data["time"], metab_data[x] / relative_dry_mass)
        axs[10 + i].set_title(x + " conc")

    # axs[0].legend()

    # axs[1].plot(metab_data["time"], purF_kinetic_target_data / np.mean(purF_kinetic_target_data), label='target')
    # axs[1].plot(metab_data["time"], purF_actual_flux_data / np.mean(purF_actual_flux_data), label='actual')
    # axs[1].plot([0, 2000], [1, 1], label="mean")
    # axs[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "purine_metab_analysis.png"))
    plt.close("all")
