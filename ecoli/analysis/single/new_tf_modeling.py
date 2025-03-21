import os
from typing import Any, cast

from duckdb import DuckDBPyConnection
import polars as pl
import pickle
import matplotlib.pyplot as plt

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

    n_avogadro = sim_data.constants.n_avogadro
    cell_density = sim_data.constants.cell_density

    purC_monomer_name = "purC monomer"
    purC_rna_name = "purC TU"
    purC_gene_name = "purC gene"
    purC_rna_synth_prob_name = purC_rna_name + " synth prob"
    purC_rna_synth_aff_name = purC_rna_name + " synth aff"
    purC_per_copy_synth_prob_name = purC_rna_synth_prob_name + " per copy"
    purC_per_copy_synth_aff_name = purC_rna_synth_aff_name + " per copy"
    purR_bound_name = "purR dimer-hypoxanthine complex"
    purR_unbound_name = "free purR dimer"
    purR_free_name = "free purR monomer"
    purR_bulk_names = [purR_bound_name, purR_unbound_name, purR_free_name]
    purR_monomer_name = "total purR monomer"
    hypoxanthine_name = "hypoxanthine"
    purR_tf_name = "purR TF"

    purR_tf_binding_name = purR_tf_name + " binding"
    purR_tf_unbinding_name = purR_tf_name + " unbinding"
    purR_tf_bound_binding_sites_name = purR_tf_name + " bound binding sites"
    purR_tf_avail_binding_sites_name = purR_tf_name + " available binding sites"
    purR_bound_purC_name = "purR bound to purC"

    purC_monomer_id = "SAICARSYN-MONOMER"
    purC_RNA_id = "TU00055"
    purC_gene_id = "EG10791"
    purR_bound_id = "CPLX-123"
    purR_unbound_id = "PC00033"
    purR_monomer_id = "PD00219"
    purR_bulk_ids = [purR_bound_id, purR_unbound_id, purR_monomer_id]
    hypoxanthine_id = "HYPOXANTHINE"

    mRNA_idx_dict = {
        rna[:-3]: i
        for i, rna in enumerate(
            get_field_metadata(conn, config_sql, "listeners__rna_counts__mRNA_counts")
        )
    }
    purC_mRNA_idx = cast(int, mRNA_idx_dict[purC_RNA_id])

    rna_synth_prob_dict = {
        rna[:-3]: i
        for i, rna in enumerate(
            get_field_metadata(conn, config_sql, "listeners__rna_synth_prob__actual_rna_synth_prob")
        )
    }
    purC_tu_idx = cast(int, rna_synth_prob_dict[purC_RNA_id])

    monomer_idx_dict = {
        monomer[:-3]: i
        for i, monomer in enumerate(
            get_field_metadata(conn, config_sql, "listeners__monomer_counts")
        )
    }
    purC_monomer_idx = cast(int, monomer_idx_dict[purC_monomer_id])
    purR_monomer_idx = cast(int, monomer_idx_dict[purR_monomer_id])

    gene_idx_dict = {
        gene: i
        for i, gene in enumerate(
            get_field_metadata(conn, config_sql, "listeners__rna_synth_prob__gene_copy_number")
        )
    }
    purC_gene_idx = cast(int, gene_idx_dict[purC_gene_id])

    bulk_dict = {
        mol[:-3]: i
        for i, mol in enumerate(
            get_field_metadata(conn, config_sql, "bulk")
        )
    }
    purR_bulk_idxs = [cast(int, bulk_dict[x]) for x in purR_bulk_ids]

    metab_counts_dict = {
        mol[:-3]: i
        for i, mol in enumerate(
            get_field_metadata(conn, config_sql, "listeners__enzyme_kinetics__metabolite_counts_final")
        )
    }
    hypoxanthine_metab_idx = cast(int, metab_counts_dict[hypoxanthine_id])

    tf_binding_dict = {
        tf: i
        for i, tf in enumerate(
            get_field_metadata(conn, config_sql, "listeners__rna_synth_prob__n_binding_events")
        )
    }
    purR_tf_idx = cast(int, tf_binding_dict[purR_bound_id])

    purC_mRNA = named_idx(
        "listeners__rna_counts__mRNA_counts", [purC_rna_name], [purC_mRNA_idx]
    )
    purC_purR_monomers = named_idx(
        "listeners__monomer_counts", [purC_monomer_name, purR_monomer_name], [purC_monomer_idx, purR_monomer_idx]
    )
    purC_gene = named_idx(
        "listeners__rna_synth_prob__gene_copy_number", [purC_gene_name], [purC_gene_idx]
    )
    # TODO: could look at target rna synth prob too, if wanted.
    purC_rna_synth_prob = named_idx(
        "listeners__rna_synth_prob__actual_rna_synth_prob", [purC_rna_synth_prob_name], [purC_tu_idx]
    )
    purC_rna_synth_aff = named_idx(
        "listeners__rna_synth_prob__rna_synth_aff", [purC_rna_synth_aff_name], [purC_tu_idx]
    )
    purR_bulk = named_idx(
        "bulk", purR_bulk_names, purR_bulk_idxs
    )
    hypoxanthine_metab_counts = named_idx(
        "listeners__enzyme_kinetics__metabolite_counts_final", [hypoxanthine_name], [hypoxanthine_metab_idx]
    )
    purR_tf_binding = named_idx(
        "listeners__rna_synth_prob__n_binding_events", [purR_tf_binding_name], [purR_tf_idx]
    )
    purR_tf_unbinding = named_idx(
        "listeners__rna_synth_prob__n_unbinding_events", [purR_tf_unbinding_name], [purR_tf_idx]
    )
    purR_tf_bound_binding_sites = named_idx(
        "listeners__rna_synth_prob__n_bound_binding_sites", [purR_tf_bound_binding_sites_name], [purR_tf_idx]
    )
    purR_tf_available_binding_sites = named_idx(
        "listeners__rna_synth_prob__n_available_binding_sites", [purR_tf_avail_binding_sites_name], [purR_tf_idx]
    )
    purR_bound_to_purC = ndidx_to_duckdb_expr(
        "listeners__rna_synth_prob__n_bound_TF_per_TU", [purC_tu_idx, purR_tf_idx]
    )

    # Extract data
    purC_data = read_stacked_columns(
        history_sql,
        ["listeners__monomer_counts",
         "listeners__rna_counts__mRNA_counts",
         "listeners__rna_synth_prob__gene_copy_number",
         "listeners__rna_synth_prob__actual_rna_synth_prob",
         "listeners__rna_synth_prob__rna_synth_aff",
         "bulk",
         "listenners__enzyme_kinetics__metabolite_counts_final",
         "listeners__rna_synth_prob__n_binding_events",
         "listeners__rna_synth_prob__n_unbinding_events",
         "listeners__rna_synth_prob__n_bound_binding_sites",
         "listeners__rna_synth_prob__n_available_binding_sites"],
        [purC_mRNA, purC_purR_monomers, purC_gene, purC_rna_synth_prob, purC_rna_synth_aff,
         purR_bulk, hypoxanthine_metab_counts, purR_tf_binding, purR_tf_unbinding,
         purR_tf_bound_binding_sites, purR_tf_available_binding_sites],
        conn=conn,
    )
    # purR_bound_to_purC_data = read_stacked_columns(
    #     history_sql,
    #     ["listeners__rna_synth_prob__n_bound_TF_per_TU"],
    #     [purR_bound_to_purC],
    #     conn=conn,
    #     order_results=False
    # )
    cell_mass_data = read_stacked_columns(
        history_sql,
        ["listeners__mass__cell_mass"],
        conn=conn
    )

    purC_dataframe = pl.DataFrame(purC_data)
    cell_mass_dataframe = pl.DataFrame(cell_mass_data)
    #purR_bound_to_purC_dataframe = pl.DataFrame(purR_bound_to_purC_data)

    cell_mass_dataframe = cell_mass_dataframe.with_columns((cell_mass_dataframe["listeners__mass__cell_mass"] *
            (n_avogadro / cell_density).asNumber(units.L / (units.fg * units.mol))).alias("counts_to_mols"))
    # TODO: maybe a better way of doing units, cell_mass is in fg, but rn just converting the n_avo/cell_density to units.fg instead
    purC_dataframe = purC_dataframe.with_columns(purC_dataframe[hypoxanthine_name].cast(float) / cell_mass_dataframe["counts_to_mols"])
    purC_dataframe = purC_dataframe.with_columns((purC_dataframe[purC_rna_synth_aff_name] / purC_dataframe[purC_gene_name].cast(float)).alias(
        purC_per_copy_synth_aff_name
    ))
    purC_dataframe = purC_dataframe.with_columns((purC_dataframe[purC_rna_synth_prob_name] / purC_dataframe[purC_gene_name].cast(float)).alias(
        purC_per_copy_synth_prob_name
    ))

    fig, axs = plt.subplots(9, figsize=(45, 30))
    axs[0].plot(purC_dataframe["time"], purC_dataframe[purC_rna_name])
    axs[0].set_title(purC_rna_name+" counts")

    axs[1].plot(purC_dataframe["time"], purC_dataframe[purC_monomer_name])
    axs[1].set_title(purC_monomer_name+" counts")

    axs[2].plot(purC_dataframe["time"], purC_dataframe[purC_gene_name])
    axs[2].set_title(purC_gene_name+" copies")

    axs[3].plot(purC_dataframe["time"], purC_dataframe[purC_rna_synth_prob_name], label="total")
    axs[3].plot(purC_dataframe["time"], purC_dataframe[purC_per_copy_synth_prob_name], label="per gene copy")
    axs[3].set_title(purC_rna_synth_prob_name)
    axs[3].legend()

    axs[4].plot(purC_dataframe["time"], purC_dataframe[purC_rna_synth_aff_name], label="total")
    axs[4].plot(purC_dataframe["time"], purC_dataframe[purC_per_copy_synth_aff_name], label="per gene copy")
    axs[4].set_title(purC_rna_synth_aff_name)
    axs[4].legend()

    for x in purR_bulk_names:
        axs[5].plot(purC_dataframe["time"], purC_dataframe[x], label=x)
    axs[5].plot(purC_dataframe["time"], purC_dataframe[purR_monomer_name], label=purR_monomer_name)
    # TODO: check, if a purR is bound to DNA, is it still counted in bulk molecules?
    axs[5].plot(purC_dataframe["time"], purC_dataframe[purR_tf_bound_binding_sites_name], label=purR_tf_bound_binding_sites_name)
    axs[5].set_title("purR counts")
    axs[5].legend()

    axs[6].plot(purC_dataframe["time"], purC_dataframe[hypoxanthine_name])
    axs[6].set_title("Hypoxanthine concentration")

    axs[7].plot(purC_dataframe["time"], purC_dataframe[purR_tf_binding_name], label=purR_tf_binding_name)
    axs[7].plot(purC_dataframe["time"], purC_dataframe[purR_tf_unbinding_name], label=purR_tf_unbinding_name)
    axs[7].legend()
    axs[7].set_title("PurR total binding and unbinding events")

    axs[8].plot(purC_dataframe["time"], purC_dataframe[purR_tf_bound_binding_sites_name], label=purR_tf_bound_binding_sites_name)
    axs[8].plot(purC_dataframe["time"], purC_dataframe[purR_tf_avail_binding_sites_name], label=purR_tf_avail_binding_sites_name)
    axs[8].legend()
    axs[8].set_title("purR-binding promoters")

    # axs[8].plot(purC_dataframe["time"], purR_bound_to_purC_dataframe["listeners__rna_synth_prob__n_bound_TF_per_TU"])
    # axs[8].set_title("purR bound to purC")
    # TODO: add total number of purC, so we should be getting fraction of purC that are occupied

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "new_tf_modeling_test.pdf"))


    # "promoter_copy_number": ([0] * self.n_TU, self.rna_ids),
    # "gene_copy_number": ([0] * self.n_TU, self.gene_ids),
    # "bound_TF_indexes": ([], self.tf_ids),
    # "bound_TF_coordinates": [],
    # "bound_TF_domains": [],
    # "target_rna_synth_prob": ([0.0] * self.n_TU, self.rna_ids),
    # "actual_rna_synth_prob": ([0.0] * self.n_TU, self.rna_ids),
    # "actual_rna_synth_prob_per_cistron": (
    #     [0.0] * self.n_cistron,
    #     self.cistron_ids,
    # ),
    # "target_rna_synth_prob_per_cistron": (
    #     [0.0] * self.n_cistron,
    #     self.cistron_ids,
    # ),
    # "expected_rna_init_per_cistron": (
    #     [0.0] * self.n_cistron,
    #     self.cistron_ids,
    # ),
    # "n_bound_TF_per_TU": ([[0] * self.n_TF] * self.n_TU, self.rna_ids),
    # "n_bound_TF_per_cistron": ([], self.cistron_ids),
    # "total_rna_init"

    # mass_columns = {
    #     "Protein": "listeners__mass__protein_mass",
    #     "tRNA": "listeners__mass__tRna_mass",
    #     "rRNA": "listeners__mass__rRna_mass",
    #     "mRNA": "listeners__mass__mRna_mass",
    #     "DNA": "listeners__mass__dna_mass",
    #     "Small Mol.s": "listeners__mass__smallMolecule_mass",
    #     "Dry": "listeners__mass__dry_mass",
    # }
    # mass_data = read_stacked_columns(
    #     history_sql, list(mass_columns.values()), conn=conn
    # )
    # mass_data = pl.DataFrame(mass_data)
    # fractions = {
    #     k: (mass_data[v] / mass_data["listeners__mass__dry_mass"]).mean()
    #     for k, v in mass_columns.items()
    # }
    # new_columns = {
    #     "Time (min)": (mass_data["time"] - mass_data["time"].min()) / 60,
    #     **{
    #         f"{k} ({cast(float, fractions[k]):.3f})": mass_data[v] / mass_data[v][0]
    #         for k, v in mass_columns.items()
    #     },
    # }
    # mass_fold_change = pl.DataFrame(new_columns)
    #
    # fig, axs = plt.subplots(2)
    # axs[0].plot(mass_fold_change["Time (min)"], mass_fold_change[f"Protein ({cast(float, fractions['Protein']):.3f})"])
    # plt.savefig(os.path.join(outdir, "new_tf_modeling_test.pdf"))
