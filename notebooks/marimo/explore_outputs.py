import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import pickle
    import numpy as np
    import pandas as pd
    import sys
    import altair as alt
    import polars as pl
    from scipy.stats import pearsonr

    return alt, mo, np, os, pd, pearsonr, pickle, pl, sys


@app.cell
def _(os, pickle, sys):
    wd_root = os.getcwd().split("/notebooks")[0]

    sys.path.append(wd_root)

    from ecoli.library.sim_data import LoadSimData
    from ecoli.library.parquet_emitter import (
        dataset_sql,
        ndlist_to_ndarray,
        read_stacked_columns,
        create_duckdb_conn,
    )
    from wholecell.utils.protein_counts import get_simulated_validation_counts

    sim_data_path = os.path.join(
        wd_root, "reconstruction", "sim_data", "kb", "simData.cPickle"
    )

    validation_data_path = os.path.join(
        wd_root, "reconstruction", "sim_data", "kb", "validationData.cPickle"
    )

    sim_data = LoadSimData(sim_data_path).sim_data

    with open(validation_data_path, "rb") as f:
        validation_data = pickle.load(f)
    return (
        LoadSimData,
        create_duckdb_conn,
        dataset_sql,
        get_simulated_validation_counts,
        ndlist_to_ndarray,
        read_stacked_columns,
        sim_data,
        sim_data_path,
        validation_data,
        wd_root,
    )


@app.cell
def _(LoadSimData):
    def get_bulk_ids(sim_data_path):
        sim_data = LoadSimData(sim_data_path).sim_data
        bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
        return bulk_ids

    def get_rxn_ids(sim_data_path):
        sim_data = LoadSimData(sim_data_path).sim_data
        rxn_ids = sim_data.process.metabolism.base_reaction_ids
        return rxn_ids

    return get_bulk_ids, get_rxn_ids


@app.cell
def _(get_bulk_ids, get_rxn_ids, np, sim_data, sim_data_path):
    bulk_ids = get_bulk_ids(sim_data_path)
    bulk_ids_biocyc = [bulk_id[:-3] for bulk_id in bulk_ids]
    bulk_names_unique = list(np.unique(bulk_ids_biocyc))
    bulk_common_names = get_common_names(bulk_names_unique, sim_data)
    bulk_names2biocyc = {
        key: val for key, val in zip(bulk_common_names, bulk_names_unique)
    }
    rxn_ids = get_rxn_ids(sim_data_path)
    cistron_data = sim_data.process.transcription.cistron_data
    mrna_cistron_ids = cistron_data["id"][cistron_data["is_mRNA"]].tolist()
    mrna_gene_ids = [cistron_id.strip("_RNA") for cistron_id in mrna_cistron_ids]
    mrna_cistron_names = [
        sim_data.common_names.get_common_name(cistron_id)
        for cistron_id in mrna_cistron_ids
    ]
    monomer_ids = sim_data.process.translation.monomer_data["id"].tolist()
    monomer_ids = [id[:-3] for id in monomer_ids]
    monomer_names = get_common_names(monomer_ids, sim_data)
    return (
        bulk_common_names,
        bulk_ids_biocyc,
        bulk_names2biocyc,
        bulk_names_unique,
        monomer_ids,
        monomer_names,
        mrna_cistron_names,
        mrna_gene_ids,
        rxn_ids,
    )


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>
    Welcome to the vEcoli data explorer notebook! This notebook provides and interactive interface to explore, analyze and visualize the outputs of the E. Coli whole cell model simulations.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>

    By default, vEcoli uses the Parquet emitter. It saves simulation output in a tabular file format, inside a nested directory structure called Hive partitioning. The Hive partitioning structure represents a hierarchical classification to annotate individual single cell simulations uniquely. In this strcuture, each single cell output is saved to folder with the following pattern of path:



        <p style="font-family: 'Courier New', Courier, monospace;">
            experiment_id={}/variant={}/lineage_seed={}/generation={}/agent_id={}
        </p>

        <p style="font-family: Arial, sans-serif;">
        </p>

    This allows efficient organization, storage and retrieval of outputs from simulation workflows that run tasks with many variants,lineage seeds, generations and agent IDs.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>
    To proceed, please select the analysis type, which defines the manner in which simulation output will be aggregated. You may choose to retrieve output for a single cell, or aggregated output from multiple cells with groups defined by varying levels of hierarchy. For example, a multiseed analysis will aggregate all "agents" and "generations" for a selected "lineage_seed". The selected analysis type will determine how many and which parititions are to be specified by the user. For a "single" analysis, all the partitions required to define a single cell need to be specified, i.e., experiment_id, variant, lineage_seed, generation and agent_id.
    """
    )
    return


@app.cell
def _(mo):
    analysis_select = mo.ui.dropdown(
        options=["single", "multidaughter", "multigeneration", "multiseed"],
        value="single",
    )
    mo.hstack(["analysis type:", analysis_select], justify="start")
    return (analysis_select,)


@app.cell
def _(get_exp, mo, outdir_tree):
    exp_select = mo.ui.dropdown(options=get_exp(outdir_tree))
    y_scale = mo.ui.dropdown(options=["linear", "log", "symlog"], value="linear")

    return exp_select, y_scale


@app.cell
def _(analysis_select, mo, partition_groups, partitions_display):
    partitions_req = partition_groups[analysis_select.value]

    partitions_select_all = partitions_display()

    partition_selector = []

    for i in range(len(partitions_req)):
        partition_selector.append(str(partitions_req[i]) + ":")
        partition_selector.append(partitions_select_all[partitions_req[i]])

    mo.hstack(partition_selector, justify="start")
    return


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>

    Once all the paritions are correctly specified for the selected analysis type, outputs may be loaded for visualizations and further analysis. In the following series of visualizaitons, we present the time series plots of selected compound molecule counts, mRNA counts, protein monomer counts and metabolic reaction fluxes. Elements within each plotted dataset (i.e. RNA, protein, reaction) may be selected by the user from the attached dropdown menu. Alternatively, the user may select a pathway from the following menu which will specify the plotted elements based on the pathway components as defined by the EcoCyc database.
    """
    )
    return


@app.cell
def _(exp_select, get_variants, mo, outdir_tree):
    variant_select = mo.ui.dropdown(
        options=get_variants(outdir_tree, exp_id=exp_select.value)
    )
    return (variant_select,)


@app.cell
def _(exp_select, get_seeds, mo, outdir_tree, variant_select):
    seed_select = mo.ui.dropdown(
        options=get_seeds(
            outdir_tree, exp_id=exp_select.value, var_id=variant_select.value
        )
    )
    return (seed_select,)


@app.cell
def _(exp_select, get_gens, mo, outdir_tree, seed_select, variant_select):
    gen_select = mo.ui.dropdown(
        options=get_gens(
            outdir_tree,
            exp_id=exp_select.value,
            var_id=variant_select.value,
            seed_id=seed_select.value,
        )
    )
    return (gen_select,)


@app.cell
def _(
    exp_select,
    gen_select,
    get_agents,
    mo,
    outdir_tree,
    seed_select,
    variant_select,
):
    agent_select = mo.ui.dropdown(
        options=get_agents(
            outdir_tree,
            exp_id=exp_select.value,
            var_id=variant_select.value,
            seed_id=seed_select.value,
            gen_id=gen_select.value,
        )
    )
    return (agent_select,)


@app.cell
def _(analysis_select, get_db_filter, partitions_dict):
    dbf_dict = partitions_dict(analysis_select.value)
    db_filter = get_db_filter(dbf_dict)

    return (db_filter,)


@app.cell
def _(dataset_sql, exp_select, os, wd_root):
    datapoints_cap = 2000

    history_sql_base, _, _ = dataset_sql(
        os.path.join(wd_root, "out"), experiment_ids=[exp_select.value]
    )

    return datapoints_cap, history_sql_base


@app.cell
def _(get_pathways, mo, pathway_dir):
    select_pathway = mo.ui.dropdown(options=get_pathways(pathway_dir), searchable=True)
    return (select_pathway,)


@app.cell
def _(mo, select_pathway):
    mo.hstack(["pathway:", select_pathway], justify="start")
    return


@app.cell
def _(mo):
    molecule_id_type = mo.ui.radio(
        options=["Common name", "BioCyc ID"], value="Common name"
    )

    return (molecule_id_type,)


@app.cell
def _(
    bulk_common_names,
    bulk_names_unique,
    bulk_override,
    mo,
    molecule_id_type,
    select_pathway,
):
    if molecule_id_type.value == "Common name":
        molecule_id_options = bulk_common_names
    elif molecule_id_type.value == "BioCyc ID":
        molecule_id_options = bulk_names_unique

    bulk_sp_plot = mo.ui.multiselect(
        options=molecule_id_options,
        value=bulk_override(select_pathway.value),
        max_selections=500,
    )
    return (bulk_sp_plot,)


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>

    **Compound Molecule Counts:** The "bulk" store in the vEcoli model tracks individual molecule counts of modeled comopunds, namely, transcription units, RNAs, proteins, complexes, metabolites and small molecules. In this section, we generate time course plots of user selected compounds. If no pathway is selected, you may specify compounds to plot from the following menu, using their BioCyc IDs or display names.
    """
    )
    return


@app.cell
def _(bulk_sp_plot, mo, molecule_id_type, y_scale):
    bulk_select = [
        mo.ui.button(label="""Compound Molecule Counts """),
        "label type:",
        molecule_id_type,
    ]

    if molecule_id_type.value == "Common name":
        bulk_select.append("name:")
        bulk_select.append(bulk_sp_plot)
    elif molecule_id_type.value == "BioCyc ID":
        bulk_select.append("id:")
        bulk_select.append(bulk_sp_plot)

    bulk_select.append("Scale:")
    bulk_select.append(y_scale)

    mo.hstack(bulk_select, justify="start")
    return


@app.cell
def _(
    bulk_ids_biocyc,
    bulk_names2biocyc,
    bulk_sp_plot,
    conn,
    datapoints_cap,
    db_filter,
    get_plot_df_bulk,
    history_sql_base,
    molecule_id_type,
):
    plot_df_bulk = None
    if bulk_sp_plot.value:
        plot_df_bulk = get_plot_df_bulk(
            bulk_sp_plot,
            bulk_ids_biocyc,
            bulk_names2biocyc,
            history_sql_base,
            db_filter,
            datapoints_cap,
            conn,
            molecule_id_type,
        )
    return (plot_df_bulk,)


@app.cell
def _(alt, bulk_sp_plot, plot_df_bulk, y_scale):
    chart_compounds = None

    if bulk_sp_plot.value:
        chart_compounds = (
            alt.Chart(plot_df_bulk)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "counts:Q", scale=alt.Scale(type=y_scale.value), title="Counts"
                ),
                color=alt.Color("compound:N", legend=alt.Legend(title="Compound")),
            )
        )
    chart_compounds
    return


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>

    **mRNA Counts:** In this section, we generate time course plots of selected mRNA cistron counts. If no pathway is selected, mRNAs may be specified with gene names or their BioCyc IDs.
    """
    )
    return


@app.cell
def _(mo):
    rna_label_type = mo.ui.radio(options=["gene name", "BioCyc ID"], value="gene name")

    y_scale_mrna = mo.ui.dropdown(options=["linear", "log", "symlog"], value="linear")

    monomer_label_type = mo.ui.radio(
        options=["common name", "BioCyc ID"], value="common name"
    )

    y_scale_monomers = mo.ui.dropdown(
        options=["linear", "log", "symlog"], value="symlog"
    )
    return monomer_label_type, rna_label_type, y_scale_monomers, y_scale_mrna


@app.cell
def _(mo, mrna_select_plot, rna_label_type, y_scale_mrna):
    mrna_select_menu = [
        mo.ui.button(label="mRNA Counts"),
        "label type:",
        rna_label_type,
    ]

    if rna_label_type.value == "gene name":
        mrna_select_menu.append("name:")
        mrna_select_menu.append(mrna_select_plot)
    elif rna_label_type.value == "BioCyc ID":
        mrna_select_menu.append("ID:")
        mrna_select_menu.append(mrna_select_plot)

    mrna_select_menu.append("Scale:")
    mrna_select_menu.append(y_scale_mrna)

    mo.hstack(mrna_select_menu, justify="start")
    return


@app.cell
def _(
    mo,
    mrna_cistron_names,
    mrna_gene_ids,
    mrna_override,
    rna_label_type,
    select_pathway,
):
    if rna_label_type.value == "gene name":
        rna_label_options = mrna_cistron_names
    elif rna_label_type.value == "BioCyc ID":
        rna_label_options = mrna_gene_ids

    mrna_select_plot = mo.ui.multiselect(
        options=rna_label_options,
        value=mrna_override(select_pathway.value),
        max_selections=500,
    )

    return (mrna_select_plot,)


@app.cell
def _(
    mo,
    monomer_ids,
    monomer_label_type,
    monomer_names,
    protein_override,
    select_pathway,
):
    monomer_label_dict = {"common name": monomer_names, "BioCyc ID": monomer_ids}

    monomer_select_plot = mo.ui.multiselect(
        options=monomer_label_dict[monomer_label_type.value],
        value=protein_override(select_pathway.value),
        max_selections=500,
    )
    return (monomer_select_plot,)


@app.cell
def _(
    conn,
    datapoints_cap,
    db_filter,
    get_plot_df,
    history_sql_base,
    mrna_cistron_names,
    mrna_gene_ids,
    mrna_select_plot,
    rna_label_type,
):
    plot_df_mrna = None
    if mrna_select_plot.value:
        plot_df_mrna = get_plot_df(
            mrna_gene_ids,
            mrna_select_plot,
            "listeners__rna_counts__full_mRNA_cistron_counts",
            "mrna_counts",
            "Genes",
            "counts",
            history_sql_base,
            db_filter,
            datapoints_cap,
            conn,
            mrna_cistron_names,
            rna_label_type,
        )
    return (plot_df_mrna,)


@app.cell
def _(alt, mrna_select_plot, plot_df_mrna, y_scale):
    chart_mrna = None

    if mrna_select_plot.value:
        chart_mrna = (
            alt.Chart(plot_df_mrna)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "counts:Q", scale=alt.Scale(type=y_scale.value), title="Counts"
                ),
                color=alt.Color("Genes:N", legend=alt.Legend(title="Genes")),
            )
        )

    chart_mrna
    return


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>
    **Protein Monomer Counts:** This time course plot visualizes the protein content of the simulation output in terms of monomer counts. Monomers to plot can be specified with their BioCyc IDs or display names.
    """
    )
    return


@app.cell
def _(mo, monomer_label_type, monomer_select_plot, y_scale_monomers):
    monomer_menu_text = {"common name": "name: ", "BioCyc ID": "ID: "}
    monomer_select_menu = [
        mo.ui.button(label="protein monomer counts"),
        "label type:",
        monomer_label_type,
        monomer_menu_text[monomer_label_type.value],
        monomer_select_plot,
        "scale: ",
        y_scale_monomers,
    ]

    mo.hstack(monomer_select_menu, justify="start")
    return


@app.cell
def _(
    conn,
    datapoints_cap,
    db_filter,
    get_plot_df,
    history_sql_base,
    monomer_ids,
    monomer_label_type,
    monomer_names,
    monomer_select_plot,
):
    plot_df_monomers = None

    if monomer_select_plot.value:
        plot_df_monomers = get_plot_df(
            monomer_ids,
            monomer_select_plot,
            "listeners__monomer_counts",
            "monomer_counts",
            "protein names",
            "counts",
            history_sql_base,
            db_filter,
            datapoints_cap,
            conn,
            monomer_names,
            monomer_label_type,
        )

    return (plot_df_monomers,)


@app.cell
def _(alt, monomer_select_plot, plot_df_monomers, y_scale_monomers):
    chart_monomers = None

    if monomer_select_plot.value:
        chart_monomers = (
            alt.Chart(plot_df_monomers)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "counts:Q",
                    scale=alt.Scale(type=y_scale_monomers.value),
                    title="Counts",
                ),
                color=alt.Color("protein names:N", legend=alt.Legend(title="Proteins")),
            )
        )

    chart_monomers
    return


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>

    **Metabolic Reaction Fluxes:** In this plot, we visualize time course of metabolic reaction fluxes. Individual reactions can be selected using their BioCyc IDs
    """
    )
    return


@app.cell
def _(mo, rxn_ids, rxn_override, select_pathway):
    select_rxns = mo.ui.multiselect(
        options=rxn_ids, value=rxn_override(select_pathway.value), max_selections=500
    )
    y_scale_rxns = mo.ui.dropdown(options=["linear", "log", "symlog"], value="symlog")
    mo.hstack(
        [
            mo.ui.button(label="Reaction Fluxes"),
            "Reaction ID(s):",
            select_rxns,
            "scale:",
            y_scale_rxns,
        ],
        justify="start",
    )
    return select_rxns, y_scale_rxns


@app.cell
def _(
    conn,
    datapoints_cap,
    db_filter,
    get_plot_df,
    history_sql_base,
    rxn_ids,
    select_rxns,
):
    plot_df_rxns = None

    if select_rxns.value:
        plot_df_rxns = get_plot_df(
            rxn_ids,
            select_rxns,
            "listeners__fba_results__base_reaction_fluxes",
            "reaction_fluxes",
            "reaction_id",
            "flux",
            history_sql_base,
            db_filter,
            datapoints_cap,
            conn,
            dtype="FLOAT",
        )
    return (plot_df_rxns,)


@app.cell
def _(alt, plot_df_rxns, select_rxns, y_scale_rxns):
    chart_rxns = None

    if select_rxns.value:
        chart_rxns = (
            alt.Chart(plot_df_rxns)
            .mark_line()
            .encode(
                x=alt.X(
                    "time:Q",
                    scale=alt.Scale(type="linear"),
                    axis=alt.Axis(tickCount=4),
                    title="Time (s)",
                ),
                y=alt.Y(
                    "flux:Q",
                    scale=alt.Scale(type=y_scale_rxns.value),
                    title="Reaction Flux (mmol/s)",
                ),
                color=alt.Color(
                    "reaction_id:N", legend=alt.Legend(title="Reaction ID (BioCyc)")
                ),
            )
        )

    chart_rxns
    return


@app.cell
def _(create_duckdb_conn, os, wd_root):
    conn = create_duckdb_conn(os.path.join(wd_root, "out"), False, 1)

    return (conn,)


@app.cell
def _(
    conn,
    db_filter,
    history_sql_base,
    ndlist_to_ndarray,
    read_stacked_columns,
):
    history_sql_subquery = f"SELECT * FROM ({history_sql_base}) WHERE {db_filter}"

    subquery = read_stacked_columns(
        history_sql_subquery, ["listeners__monomer_counts"], order_results=False
    )

    sql_monomer_validation = f"""
            WITH unnested_counts AS (
                SELECT unnest(listeners__monomer_counts) AS counts,
                    generate_subscripts(listeners__monomer_counts, 1) AS idx,
                    experiment_id, variant, lineage_seed, generation, agent_id
                FROM ({subquery})
            ),
            avg_counts AS (
                SELECT avg(counts) AS avgCounts,
                    experiment_id, variant, lineage_seed,
                    generation, agent_id, idx
                FROM unnested_counts
                GROUP BY experiment_id, variant, lineage_seed,
                    generation, agent_id, idx
            )
            SELECT list(avgCounts ORDER BY idx) AS avgCounts
            FROM avg_counts
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
            """
    monomer_counts = conn.sql(sql_monomer_validation).pl()
    monomer_counts = ndlist_to_ndarray(monomer_counts["avgCounts"])
    return (monomer_counts,)


@app.cell
def _(
    get_simulated_validation_counts,
    get_val_ids,
    monomer_counts,
    sim_data,
    validation_data,
):
    sim_monomer_ids = sim_data.process.translation.monomer_data["id"]
    wisniewski_ids = validation_data.protein.wisniewski2014Data["monomerId"]
    schmidt_ids = validation_data.protein.schmidt2015Data["monomerId"]
    wisniewski_counts = validation_data.protein.wisniewski2014Data["avgCounts"]
    schmidt_counts = validation_data.protein.schmidt2015Data["glucoseCounts"]
    sim_wisniewski_counts, val_wisniewski_counts = get_simulated_validation_counts(
        wisniewski_counts, monomer_counts, wisniewski_ids, sim_monomer_ids
    )
    sim_schmidt_counts, val_schmidt_counts = get_simulated_validation_counts(
        schmidt_counts, monomer_counts, schmidt_ids, sim_monomer_ids
    )
    schmidt_val_ids = get_val_ids(schmidt_ids, sim_monomer_ids)
    wisniewski_val_ids = get_val_ids(wisniewski_ids, sim_monomer_ids)

    val_options = {
        "Schmidt 2015": {
            "id": schmidt_val_ids,
            "data": val_schmidt_counts,
            "sim": sim_schmidt_counts,
        },
        "Wisniewski 2014": {
            "id": wisniewski_val_ids,
            "data": val_wisniewski_counts,
            "sim": sim_wisniewski_counts,
        },
    }
    return (val_options,)


@app.cell
def _(mo):
    val_dataset_select = mo.ui.dropdown(
        options=["Schmidt 2015", "Wisniewski 2014"], value="Schmidt 2015"
    )
    val_label_type = mo.ui.dropdown(
        options=["Common Name", "BioCyc ID"], value="Common Name"
    )
    return val_dataset_select, val_label_type


@app.cell
def _(
    mo,
    protein_val_override,
    select_pathway,
    val_dataset_select,
    val_options,
):
    val_id_select = mo.ui.multiselect(
        options=val_options[val_dataset_select.value]["id"],
        value=protein_val_override(select_pathway.value),
    )
    return (val_id_select,)


@app.cell
def _(sim_data, val_label_type):
    def get_val_ids(data_ids, sim_ids):
        sim_ids_lst = sim_ids.tolist()
        data_ids_lst = data_ids.tolist()
        overlapping_ids_set = set(sim_ids_lst) & set(data_ids_lst)
        val_ids = list(overlapping_ids_set)
        val_ids = [id[:-3] for id in val_ids]
        val_ids_mapping = {
            "Common Name": get_common_names(val_ids, sim_data),
            "BioCyc ID": val_ids,
        }
        val_ids_final = val_ids_mapping[val_label_type.value]
        return val_ids_final

    return (get_val_ids,)


@app.cell
def _(mo):
    mo.md(
        """
    <p style="font-family: Arial, sans-serif;">
        </p>
    **Protein Count Validation:** This is a scatter plot comparing simulated average protein counts to experimental proteomics datasets. This is applicable to proteins overlapping the modeled proteins and either of the validation datasets. You may choose to visualize all available proteins or pathway specific proteins. Alternatively, the attached drop down menu can be used to select proteins using their BioCyc IDs or display names.
    """
    )
    return


@app.cell
def _(mo, val_dataset_select, val_id_select, val_label_type):
    val_menu_text = {"Common Name": "Name: ", "BioCyc ID": "ID: "}
    val_select_menu = [
        mo.ui.button(label="Protein Count Validation"),
        "Validation Dataset: ",
        val_dataset_select,
        "Label Type: ",
        val_label_type,
        val_menu_text[val_label_type.value],
        val_id_select,
    ]

    mo.hstack(val_select_menu, justify="start")
    return


@app.cell
def _(alt, np, pearsonr, pl, val_id_select, val_options):
    def val_chart(dataset_name):
        data_val = val_options[dataset_name]["data"]
        data_sim = val_options[dataset_name]["sim"]
        data_idxs = [
            val_options[dataset_name]["id"].index(name) for name in val_id_select.value
        ]
        data_val_filtered = data_val[data_idxs]
        data_sim_filtered = data_sim[data_idxs]

        chart = (
            alt.Chart(
                pl.DataFrame(
                    {
                        dataset_name: np.log10(data_val_filtered + 1),
                        "sim": np.log10(data_sim_filtered + 1),
                        "protein": val_id_select.value,
                    }
                )
            )
            .mark_point()
            .encode(
                x=alt.X(dataset_name, title=f"log10({dataset_name} Counts + 1)"),
                y=alt.Y("sim", title="log10(Simulation Average Counts + 1)"),
                tooltip=["protein:N"],
            )
            .properties(
                title="Pearson r: %0.2f"
                % pearsonr(
                    np.log10(data_sim_filtered + 1), np.log10(data_val_filtered + 1)
                )[0]
            )
        )

        max_val = max(
            np.log10(val_options["Schmidt 2015"]["data"] + 1).max(),
            np.log10(val_options["Wisniewski 2014"]["data"] + 1).max(),
            np.log10(val_options["Schmidt 2015"]["sim"] + 1).max(),
            np.log10(val_options["Wisniewski 2014"]["sim"] + 1).max(),
        )
        parity = (
            alt.Chart(pl.DataFrame({"x": np.arange(max_val)}))
            .mark_line()
            .encode(x="x", y="x", color=alt.value("red"), strokeDash=alt.value([5, 5]))
        )

        chart_final = chart + parity

        return chart_final

    return (val_chart,)


@app.cell
def _(val_chart, val_dataset_select, val_id_select):
    chart_val = None
    if val_id_select.value:
        chart_val = val_chart(val_dataset_select.value)
    chart_val
    return


@app.cell
def _(partition_groups, read_partitions):
    def partitions_dict(analysis_type):
        partitions_req = partition_groups[analysis_type]
        partitions_all = read_partitions()

        partitions_dict = {}
        for partition in partitions_req:
            partitions_dict[partition] = partitions_all[partition]
        partitions_dict["experiment_id"] = f"'{partitions_dict['experiment_id']}'"
        return partitions_dict

    def get_db_filter(partitions_dict):
        db_filter_list = []
        for key, value in partitions_dict.items():
            db_filter_list.append(str(key) + "=" + str(value))
        db_filter = " AND ".join(db_filter_list)

        return db_filter

    return get_db_filter, partitions_dict


@app.cell
def _(agent_select, exp_select, gen_select, seed_select, variant_select):
    partition_groups = {
        "multiseed": ["experiment_id", "variant"],
        "multigeneration": ["experiment_id", "variant", "lineage_seed"],
        "multidaughter": ["experiment_id", "variant", "lineage_seed", "generation"],
        "single": [
            "experiment_id",
            "variant",
            "lineage_seed",
            "generation",
            "agent_id",
        ],
    }

    def partitions_display():
        partitions_list = {
            "experiment_id": exp_select,
            "variant": variant_select,
            "lineage_seed": seed_select,
            "generation": gen_select,
            "agent_id": agent_select,
        }

        return partitions_list

    def read_partitions():
        partitions_selected = {
            "experiment_id": exp_select.value,
            "variant": variant_select.value,
            "lineage_seed": seed_select.value,
            "generation": gen_select.value,
            "agent_id": agent_select.value,
        }
        return partitions_selected

    return partition_groups, partitions_display, read_partitions


@app.function
def get_common_names(bulk_names, sim_data):
    bulk_common_names = [
        sim_data.common_names.get_common_name(name) for name in bulk_names
    ]

    duplicates = []

    for item in bulk_common_names:
        if bulk_common_names.count(item) > 1 and item not in duplicates:
            duplicates.append(item)

    for dup in duplicates:
        sp_idxs = [index for index, item in enumerate(bulk_common_names) if item == dup]

        for sp_idx in sp_idxs:
            bulk_rename = str(bulk_common_names[sp_idx]) + f"[{bulk_names[sp_idx]}]"
            bulk_common_names[sp_idx] = bulk_rename

    return bulk_common_names


@app.cell
def _(
    bulk_common_names,
    bulk_names_unique,
    molecule_id_type,
    monomer_ids,
    monomer_label_type,
    monomer_names,
    mrna_cistron_names,
    mrna_gene_ids,
    np,
    os,
    pd,
    rna_label_type,
    rxn_ids,
    val_dataset_select,
    val_options,
):
    pathway_dir = "pathways"

    def get_pathways(pathway_dir):
        pathway_file = os.path.join(pathway_dir, "pathways.txt")
        pathway_df = pd.read_csv(pathway_file, sep="\t")
        pathway_list = pathway_df["name"].values
        pathway_list = list(np.unique(pathway_list))
        return pathway_list

    def get_presets(preset_dir):
        preset_files = os.listdir(preset_dir)
        presets_list = [file.split(".")[0] for file in preset_files]

        return presets_list

    def read_columns(st_column):
        values = []
        for item in st_column:
            items_actual = str(item).split(" // ")
            for item_actual in items_actual:
                values.append(item_actual)
        return values

    def read_presets(pathway_name):
        preset_dict = {}
        if isinstance(pathway_name, str):
            preset_table = pd.read_csv(
                os.path.join(pathway_dir, "pathways.txt"), header=0, sep="\t"
            )
            pathway_df = preset_table[preset_table["name"] == pathway_name]

            preset_dict["reactions"] = read_columns(pathway_df["reactions"])
            preset_dict["genes"] = read_columns(pathway_df["genes"])
            preset_dict["compounds"] = read_columns(pathway_df["compounds"])

        return preset_dict

    def preset_override(preset_name):
        preset_dict = read_presets(preset_name)

        preset_final = {}

        if len(preset_dict) > 0:
            preset_final["reactions"] = np.array(preset_dict["reactions"])[
                np.isin(preset_dict["reactions"], rxn_ids)
            ].tolist()

            preset_final["genes"] = np.array(preset_dict["genes"])[
                np.isin(preset_dict["genes"], mrna_gene_ids)
            ].tolist()

            preset_final["genes"] = np.unique(preset_final["genes"]).tolist()

            if rna_label_type.value == "gene name":
                preset_gene_names = []
                for gene_id in preset_final["genes"]:
                    preset_gene_names.append(
                        mrna_cistron_names[mrna_gene_ids.index(gene_id)]
                    )
                preset_final["genes"] = preset_gene_names

            preset_final["compounds"] = np.array(preset_dict["compounds"])[
                np.isin(preset_dict["compounds"], bulk_names_unique)
            ].tolist()

            preset_final["compounds"] = np.unique(preset_final["compounds"]).tolist()

            preset_final["proteins"] = list(
                np.array(preset_final["compounds"])[
                    np.isin(preset_final["compounds"], monomer_ids)
                ]
            )

            if molecule_id_type.value == "Common name":
                preset_compound_names = []
                for name in preset_final["compounds"]:
                    preset_compound_names.append(
                        bulk_common_names[bulk_names_unique.index(name)]
                    )
                preset_final["compounds"] = preset_compound_names

            if monomer_label_type.value == "common name":
                preset_protein_names = []
                for name in preset_final["proteins"]:
                    preset_protein_names.append(monomer_names[monomer_ids.index(name)])
                preset_final["proteins"] = preset_protein_names

        return preset_final

    def bulk_override(preset_name):
        preset_dict = preset_override(preset_name)
        bulk_list = preset_dict.get("compounds")
        return bulk_list

    def rxn_override(preset_name):
        preset_dict = preset_override(preset_name)
        rxn_list = preset_dict.get("reactions")
        return rxn_list

    def mrna_override(preset_name):
        preset_dict = preset_override(preset_name)
        mrna_list = preset_dict.get("genes")
        return mrna_list

    def protein_override(preset_name):
        preset_dict = preset_override(preset_name)
        protein_list = preset_dict.get("proteins")
        return protein_list

    def protein_val_override(preset_name):
        protein_list = protein_override(preset_name)
        dataset_name = val_dataset_select.value
        protein_ids_val = val_options[dataset_name]["id"]
        protein_val = list(
            np.array(protein_list)[np.isin(protein_list, protein_ids_val)]
        )
        return protein_val

    return (
        bulk_override,
        get_pathways,
        mrna_override,
        pathway_dir,
        protein_override,
        protein_val_override,
        rxn_override,
    )


@app.cell
def _(np, pl):
    def get_plot_df_bulk(
        bulk_select_ui,
        bulk_ids_biocyc,
        bulk_names2biocyc,
        sql_base,
        db_filter,
        datapoints_cap,
        conn,
        molecule_id_ui,
    ):
        if molecule_id_ui.value == "Common name":
            bulk_sp_ids = [bulk_names2biocyc[name] for name in bulk_select_ui.value]
        else:
            bulk_sp_ids = bulk_select_ui.value

        sp_idxs_selected = [
            [
                f"bulk[{index + 1}]"
                for index, item in enumerate(bulk_ids_biocyc)
                if item == sp_i
            ]
            for sp_i in bulk_sp_ids
        ]

        sp_idxs_alias = [
            "+".join(sp_idxs_i) + f" as compound_{count}"
            for count, sp_idxs_i in enumerate(sp_idxs_selected)
        ]

        bulk_sql_opt = (
            f"SELECT {','.join(sp_idxs_alias)},time FROM ({sql_base}) WHERE {db_filter}"
        )

        bulk_sql_opt_sum = (
            "SELECT"
            + ",".join(
                [
                    f" CAST (SUM(compound_{sp_idx}) AS BIGINT) AS compound_{sp_idx}"
                    for sp_idx, _ in enumerate(bulk_sp_ids)
                ]
            )
            + f", time FROM ({bulk_sql_opt}) GROUP BY time"
        )

        bulk_sql_list = (
            "SELECT ("
            + "+".join([f"[compound_{sp_idx}]" for sp_idx, _ in enumerate(bulk_sp_ids)])
            + f") AS bulk_counts, time FROM ({bulk_sql_opt_sum})"
        )

        bulk_sql_ds = sql_downsample(
            bulk_sql_list, bulk_sp_ids, "bulk_counts", datapoints_cap
        )

        df_bulk_read = conn.sql(bulk_sql_ds).pl()

        bulk_counts_mtx = np.stack(df_bulk_read["bulk_counts"])
        bulk_counts_list = [
            bulk_counts_mtx[:, col] for col in range(np.shape(bulk_counts_mtx)[1])
        ]
        bulk_plot_dict = {
            key: val for (key, val) in zip(bulk_select_ui.value, bulk_counts_list)
        }
        bulk_plot_dict["time"] = df_bulk_read["time"].to_list()
        bulk_plot_df = pl.DataFrame(bulk_plot_dict)
        bulk_plot_df_melted = bulk_plot_df.unpivot(
            index="time", variable_name="compound", value_name="counts"
        )

        return bulk_plot_df_melted

    return (get_plot_df_bulk,)


@app.cell
def _(np, pl):
    def get_plot_df(
        default_id_list,
        item_selector_ui,
        listener_name,
        col_name,
        var_name,
        val_name,
        sql_base,
        db_filter,
        datapoints_cap,
        conn,
        default_name_list=None,
        label_ui=None,
        dtype="BIGINT",
    ):
        if label_ui:
            if label_ui.value == "BioCyc ID":
                ids_selected = item_selector_ui.value
            else:
                ids_selected = [
                    default_id_list[default_name_list.index(name)]
                    for name in item_selector_ui.value
                ]
        else:
            ids_selected = item_selector_ui.value

        idxs_selected = [
            f"{col_name}[{default_id_list.index(id) + 1}] AS {col_name}_{idx}"
            for idx, id in enumerate(ids_selected)
        ]
        col_sql_base = f"SELECT {listener_name} as {col_name}, time FROM ({sql_base}) WHERE {db_filter}"
        col_sql_sliced = f"SELECT {','.join(idxs_selected)}, time FROM ({col_sql_base})"
        col_sql_sum = (
            "SELECT"
            + ",".join(
                [
                    f" CAST (SUM({col_name}_{idx}) AS {dtype}) as {col_name}_{idx}"
                    for idx, _ in enumerate(ids_selected)
                ]
            )
            + f",time FROM ({col_sql_sliced}) GROUP BY time"
        )
        col_sql_list = (
            "SELECT ("
            + "+".join([f"[{col_name}_{idx}]" for idx, _ in enumerate(ids_selected)])
            + f") AS {col_name}, time FROM ({col_sql_sum})"
        )
        col_sql_ds = sql_downsample(
            col_sql_list, ids_selected, col_name, datapoints_cap
        )

        col_read_df = conn.sql(col_sql_ds).pl()

        counts_mtx = np.stack(col_read_df[col_name])
        counts_list = [counts_mtx[:, col] for col in range(np.shape(counts_mtx)[1])]
        plot_dict = {
            key: val for (key, val) in zip(item_selector_ui.value, counts_list)
        }
        plot_dict["time"] = col_read_df["time"].to_list()
        plot_df = pl.DataFrame(plot_dict)
        plot_df_melted = plot_df.unpivot(
            index="time", variable_name=var_name, value_name=val_name
        )

        return plot_df_melted

    return (get_plot_df,)


@app.function
def sql_downsample(sql_original, items_list, list_col_name, datapoints_cap=2000):
    ds_sql = f"""
    WITH 
    indexed_data AS (
      SELECT 
        *, 
        ROW_NUMBER() OVER (ORDER BY time) AS rn
    FROM ({sql_original})
    ),
    data_shape AS (
    SELECT
        COUNT(*) AS time_points,
        time_points*{len(items_list)} as data_points,
        data_points/{datapoints_cap} as ds_ratio_frac,
        CEILING(ds_ratio_frac) as ds_ratio

    FROM ({sql_original}))

    SELECT {list_col_name},time
    FROM indexed_data
    CROSS JOIN data_shape
    WHERE rn % data_shape.ds_ratio = 0
    ORDER BY time
    """

    return ds_sql


@app.cell
def _():
    from pathlib import Path

    def tree_to_dict(path):
        path = Path(path)
        # Define the dictionary structure for this level
        d = {"name": path.name}

        if path.is_dir():
            d["type"] = "directory"
            # Recursively call tree_to_dict for every child in the directory
            d["children"] = [tree_to_dict(child) for child in path.iterdir()]
        else:
            d["type"] = "file"

        return d

    def get_dir_tree(path):
        path = Path(path)

        if path.is_file():
            return None

        # Filter: Only include children that don't start with '.'
        return {
            child.name: get_dir_tree(child)
            for child in path.iterdir()
            if not child.name.startswith(".")
        }

    # Usage
    return Path, get_dir_tree


@app.cell
def _(Path, get_dir_tree, wd_root):
    outdir_path = Path(wd_root) / "out"
    outdir_tree = get_dir_tree(outdir_path)
    return (outdir_tree,)


@app.cell
def _(outdir_tree):
    def get_exp(tree):
        exp_list = list(outdir_tree.keys())
        exp_list.sort()
        return exp_list

    def get_variants(tree, exp_id):
        try:
            variant_folders = tree[exp_id]["history"][f"experiment_id={exp_id}"].keys()

            variants = [var.split("variant=")[1] for var in variant_folders]

            variants.sort()

        except KeyError:
            variants = ["N/A"]
        return variants

    def get_seeds(tree, exp_id, var_id):
        try:
            seed_folders = tree[exp_id]["history"][f"experiment_id={exp_id}"][
                f"variant={var_id}"
            ].keys()

            seeds = [seed.split("lineage_seed=")[1] for seed in seed_folders]

            seeds.sort()

        except KeyError:
            seeds = ["N/A"]
        return seeds

    def get_gens(tree, exp_id, var_id, seed_id):
        try:
            gen_folders = tree[exp_id]["history"][f"experiment_id={exp_id}"][
                f"variant={var_id}"
            ][f"lineage_seed={seed_id}"].keys()

            gens = [gen.split("generation=")[1] for gen in gen_folders]

            gens.sort()

        except KeyError:
            gens = ["N/A"]

        return gens

    def get_agents(tree, exp_id, var_id, seed_id, gen_id):
        try:
            agent_folders = tree[exp_id]["history"][f"experiment_id={exp_id}"][
                f"variant={var_id}"
            ][f"lineage_seed={seed_id}"][f"generation={gen_id}"].keys()
            agents = [agent.split("agent_id=")[1] for agent in agent_folders]
            agents.sort()
        except KeyError:
            agents = ["N/A"]
        return agents

    return get_agents, get_exp, get_gens, get_seeds, get_variants


if __name__ == "__main__":
    app.run()
