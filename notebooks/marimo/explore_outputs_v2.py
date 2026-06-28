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
        field_metadata,
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
        field_metadata,
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
    about_intro_md = mo.md(
        """
    Welcome to the vEcoli data explorer notebook. This notebook provides an
    interactive interface to explore, analyze and visualize the outputs of the
    E. coli whole-cell model simulations.

    By default, vEcoli uses the Parquet emitter, which saves simulation output
    in a tabular file format inside a Hive-partitioned directory structure:

    `experiment_id={}/variant={}/lineage_seed={}/generation={}/agent_id={}`

    This allows efficient organization, storage and retrieval of outputs from
    workflows that run many variants, lineage seeds, generations and agent IDs.

    Pick the **analysis type** (`single` / `multidaughter` / `multigeneration` /
    `multiseed`) to set how output is aggregated, then narrow down with the
    partition dropdowns. The selected analysis determines which partitions are
    required: e.g. `single` needs all five, `multiseed` only needs experiment
    and variant.
    """
    )
    return (about_intro_md,)


@app.cell
def _(mo):
    analysis_select = mo.ui.dropdown(
        options=["single", "multidaughter", "multigeneration", "multiseed"],
        value="single",
    )
    return (analysis_select,)


@app.cell
def _(get_exp, mo, outdir_tree):
    _exp_options = get_exp(outdir_tree)

    def _has_history(exp_id):
        entry = outdir_tree.get(exp_id)
        if not isinstance(entry, dict):
            return False
        history = entry.get("history")
        if not isinstance(history, dict):
            return False
        return any(history.values())

    _default_exp = next(
        (e for e in _exp_options if _has_history(e)),
        _exp_options[0] if _exp_options else None,
    )
    exp_select = mo.ui.dropdown(
        options=_exp_options,
        value=_default_exp,
    )
    y_scale = mo.ui.dropdown(options=["linear", "log", "symlog"], value="linear")
    return exp_select, y_scale


@app.cell
def _(analysis_select, partition_groups, partitions_display):
    # Build (label, widget) pairs for the partition keys the chosen analysis
    # type actually needs — consumed by the toolbar in the composition cell.
    _partitions_req = partition_groups[analysis_select.value]
    _all = partitions_display()
    partition_picker_items = [(str(k), _all[k]) for k in _partitions_req]
    return (partition_picker_items,)


@app.cell
def _(exp_select, get_variants, mo, outdir_tree):
    _opts = get_variants(outdir_tree, exp_id=exp_select.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    variant_select = mo.ui.dropdown(options=_opts, value=_default)
    return (variant_select,)


@app.cell
def _(exp_select, get_seeds, mo, outdir_tree, variant_select):
    _opts = get_seeds(outdir_tree, exp_id=exp_select.value, var_id=variant_select.value)
    _default = next((o for o in _opts if o != "N/A"), None)
    seed_select = mo.ui.dropdown(options=_opts, value=_default)
    return (seed_select,)


@app.cell
def _(exp_select, get_gens, mo, outdir_tree, seed_select, variant_select):
    _opts = get_gens(
        outdir_tree,
        exp_id=exp_select.value,
        var_id=variant_select.value,
        seed_id=seed_select.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    gen_select = mo.ui.dropdown(options=_opts, value=_default)
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
    _opts = get_agents(
        outdir_tree,
        exp_id=exp_select.value,
        var_id=variant_select.value,
        seed_id=seed_select.value,
        gen_id=gen_select.value,
    )
    _default = next((o for o in _opts if o != "N/A"), None)
    agent_select = mo.ui.dropdown(options=_opts, value=_default)
    return (agent_select,)


@app.cell
def _(analysis_select, get_db_filter, partitions_dict):
    dbf_dict = partitions_dict(analysis_select.value)
    db_filter = get_db_filter(dbf_dict)

    return (db_filter,)


@app.cell
def _(dataset_sql, exp_select, os, wd_root):
    datapoints_cap = 2000

    history_sql_base, config_sql_base, _ = dataset_sql(
        os.path.join(wd_root, "out"), experiment_ids=[exp_select.value]
    )

    return config_sql_base, datapoints_cap, history_sql_base


@app.cell
def _(get_pathways, mo, pathway_dir):
    select_pathway = mo.ui.dropdown(options=get_pathways(pathway_dir), searchable=True)
    return (select_pathway,)


@app.cell
def _():
    # Pathway picker is rendered by the toolbar in the composition cell.
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
    about_compounds_md = mo.md(
        "The **bulk** store in the vEcoli model tracks individual molecule "
        "counts of modeled compounds (transcription units, RNAs, proteins, "
        "complexes, metabolites, small molecules). Pick compounds by BioCyc "
        "ID or display name, or select a pathway above to auto-populate."
    )
    return (about_compounds_md,)


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
    return (chart_compounds,)


@app.cell
def _(mo):
    about_mrna_md = mo.md(
        "Time course plots of selected mRNA cistron counts. mRNAs may be "
        "specified by gene name or BioCyc ID. A pathway selection above "
        "auto-populates this list."
    )
    return (about_mrna_md,)


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
def _():
    # mRNA controls are rendered by the composition cell below.
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

    return (chart_mrna,)


@app.cell
def _(mo):
    about_proteins_md = mo.md(
        "Time course of protein monomer counts. Monomers can be specified by "
        "common name or BioCyc ID; pathway selection above auto-populates."
    )
    return (about_proteins_md,)


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

    return (chart_monomers,)


@app.cell
def _(mo):
    about_reactions_md = mo.md(
        "Time course of metabolic reaction fluxes. Individual reactions are "
        "selected by BioCyc ID; pathway selection above auto-populates."
    )
    return (about_reactions_md,)


@app.cell
def _(mo, rxn_ids, rxn_override, select_pathway):
    select_rxns = mo.ui.multiselect(
        options=rxn_ids, value=rxn_override(select_pathway.value), max_selections=500
    )
    y_scale_rxns = mo.ui.dropdown(options=["linear", "log", "symlog"], value="symlog")
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

    return (chart_rxns,)


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
    about_validation_md = mo.md(
        "Scatter plot comparing simulated average protein counts to "
        "experimental proteomics datasets (Schmidt 2015, Wisniewski 2014), "
        "restricted to proteins present in both. Pick proteins by common "
        "name or BioCyc ID."
    )
    return (about_validation_md,)


@app.cell
def _():
    # Validation controls are rendered by the composition cell below.
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
    return (chart_val,)


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


@app.cell
def _(mo):
    about_download_md = mo.md(
        """
    Export columns from the parquet history dataset as tab-separated text.
    The current partition filter is applied automatically.

    - **Vector** columns (e.g. `listeners__monomer_counts`) are exported wide:
      one row per timestep, one column per element. Headers come from the
      saved `output_metadata__` IDs when available, otherwise numeric indices.
    - **Scalar** columns can be combined into a single file: pick one or
      more, get a `time + col1 + col2 + …` TSV.
    - **Δt > 0** keeps one row per Δt-second bucket per cell, useful for
      bringing large vector exports back into browser-download size.
    """
    )
    return (about_download_md,)


@app.cell
def _(conn, history_sql_base):
    def get_history_schema(conn, history_sql_base):
        rows = conn.sql(
            f"DESCRIBE SELECT * FROM ({history_sql_base}) LIMIT 0"
        ).fetchall()
        scalar = []
        vec1d = []
        vec2d_plus = []
        partition_cols = {
            "experiment_id",
            "variant",
            "lineage_seed",
            "generation",
            "agent_id",
            "time",
        }
        for r in rows:
            name, dtype = r[0], r[1]
            if name in partition_cols:
                continue
            depth = dtype.count("[]")
            if depth == 0:
                scalar.append(name)
            elif depth == 1:
                vec1d.append(name)
            else:
                vec2d_plus.append(name)
        return scalar, vec1d, vec2d_plus

    scalar_cols, vector_cols_1d, vector_cols_2d = get_history_schema(
        conn, history_sql_base
    )
    return scalar_cols, vector_cols_1d, vector_cols_2d


@app.cell
def _(conn, config_sql_base, field_metadata):
    def get_column_metadata(col_name, fallback_len=None):
        """Return (labels, source) for a vector column header.

        Calls `field_metadata` (which looks up `output_metadata__<col>`).
        `source` is one of:
          - "metadata"          — used the saved ID list as-is
          - "metadata-trimmed"  — saved list longer than data; truncated
          - "metadata-padded"   — saved list shorter than data; padded with idx_
          - "numeric-fallback:<reason>" — field_metadata failed; using idx_
        """
        try:
            md = field_metadata(conn, config_sql_base, col_name)
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            if fallback_len is None:
                return None, f"numeric-fallback:{reason}"
            return [f"idx_{i}" for i in range(fallback_len)], (
                f"numeric-fallback:{reason}"
            )

        labels = [str(x) for x in md]
        if fallback_len is None or len(labels) == fallback_len:
            return labels, "metadata"
        if len(labels) > fallback_len:
            return labels[:fallback_len], "metadata-trimmed"
        # len(labels) < fallback_len
        padded = labels + [f"idx_{i}" for i in range(len(labels), fallback_len)]
        return padded, "metadata-padded"

    def get_vector_length(col_name, history_sql_base, db_filter):
        sql = (
            f'SELECT length("{col_name}") AS n '
            f"FROM ({history_sql_base}) WHERE {db_filter} LIMIT 1"
        )
        row = conn.sql(sql).fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    return get_column_metadata, get_vector_length


@app.cell
def _():
    def _safe_alias(s):
        return (
            str(s)
            .replace('"', '""')
            .replace("\t", " ")
            .replace("\n", " ")
            .replace("\r", " ")
        )

    def _qualify_clause(time_interval):
        """ROW_NUMBER-based first-row-per-bucket filter; '' if no downsampling."""
        if not time_interval or float(time_interval) <= 0:
            return ""
        return (
            f" QUALIFY ROW_NUMBER() OVER ("
            f"PARTITION BY experiment_id, variant, lineage_seed, "
            f"generation, agent_id, FLOOR(time / {float(time_interval)}) "
            f"ORDER BY time) = 1"
        )

    def build_vector_wide_sql(
        history_sql_base, db_filter, col_name, labels, time_interval=0.0
    ):
        # Pre-filter (and optionally time-bucket) the rows we need, only
        # carrying the partition cols, time, and the one vector column.
        base = (
            f"SELECT experiment_id, variant, lineage_seed, generation, "
            f'agent_id, time, "{col_name}" '
            f"FROM ({history_sql_base}) "
            f"WHERE {db_filter}"
            f"{_qualify_clause(time_interval)}"
        )
        select_parts = ["time"] + [
            f'"{col_name}"[{i + 1}] AS "{_safe_alias(label)}"'
            for i, label in enumerate(labels)
        ]
        return (
            f"SELECT {', '.join(select_parts)} "
            f"FROM ({base}) sub "
            f"ORDER BY experiment_id, variant, lineage_seed, "
            f"generation, agent_id, time"
        )

    def build_scalar_sql(history_sql_base, db_filter, cols, time_interval=0.0):
        # time first, then partitions, then user-picked scalars
        select_cols = [
            "time",
            "experiment_id",
            "variant",
            "lineage_seed",
            "generation",
            "agent_id",
        ]
        for c in cols:
            if c not in select_cols:
                select_cols.append(c)
        quoted = [f'"{c}"' for c in select_cols]
        return (
            f"SELECT {', '.join(quoted)} "
            f"FROM ({history_sql_base}) "
            f"WHERE {db_filter}"
            f"{_qualify_clause(time_interval)} "
            f"ORDER BY experiment_id, variant, lineage_seed, "
            f"generation, agent_id, time"
        )

    return build_scalar_sql, build_vector_wide_sql


@app.cell
def _(conn):
    def estimate_export_size(sql, n_data_cols, partition_cols=6):
        """Rough size estimate: row count * (data_cols + partition+time cols) * ~12 bytes.

        Uses 12 bytes/cell as an average across small ints and decimal floats
        rendered as text plus the tab separator. Real sizes vary a lot by dtype.
        """
        n_rows = conn.sql(f"SELECT count(*) FROM ({sql}) AS t").fetchone()[0]
        total_cols = n_data_cols + partition_cols
        est_bytes = int(n_rows) * total_cols * 12
        return int(n_rows), est_bytes

    return (estimate_export_size,)


@app.cell
def _(mo, vector_cols_1d, vector_cols_2d):
    dl_mode_select = mo.ui.radio(
        options=["vector column", "scalar columns"],
        value="vector column",
    )
    _note_2d = (
        f"  \n*Note: {len(vector_cols_2d)} column(s) are 2-D or deeper "
        "and are not offered for wide download.*"
        if vector_cols_2d
        else ""
    )
    dl_schema_note_md = mo.md(
        f"**{len(vector_cols_1d)} 1-D vector / scalar columns detected.**{_note_2d}"
    )
    return dl_mode_select, dl_schema_note_md


@app.cell
def _(mo, vector_cols_1d):
    dl_vector_col_select = mo.ui.dropdown(
        options=sorted(vector_cols_1d), searchable=True
    )
    dl_vector_label_mode = mo.ui.radio(
        options=["metadata IDs (when available)", "numeric indices"],
        value="metadata IDs (when available)",
    )
    return dl_vector_col_select, dl_vector_label_mode


@app.cell
def _(mo, scalar_cols):
    dl_scalar_cols_select = mo.ui.multiselect(
        options=sorted(scalar_cols),
        max_selections=500,
    )
    return (dl_scalar_cols_select,)


@app.cell
def _():
    # Column picker is rendered by the composition cell below.
    return


@app.cell
def _(mo, wd_root):
    dl_filename_input = mo.ui.text(value="export.tsv", placeholder="export.tsv")
    dl_delivery_radio = mo.ui.radio(
        options=[
            "auto (browser if small, disk if large)",
            "browser download",
            "save to disk",
        ],
        value="auto (browser if small, disk if large)",
    )
    dl_disk_dir_input = mo.ui.text(
        value=f"{wd_root}/exports",
        placeholder="/absolute/path/to/output/dir",
    )
    dl_size_threshold_mb = mo.ui.number(value=50, start=1, stop=4096, step=1)
    dl_time_interval = mo.ui.number(value=0.0, start=0.0, stop=1e9, step=1.0)
    return (
        dl_delivery_radio,
        dl_disk_dir_input,
        dl_filename_input,
        dl_size_threshold_mb,
        dl_time_interval,
    )


@app.cell
def _():
    # Delivery / Δt / output-dir layout is rendered by the composition cell.
    return


@app.cell
def _(mo):
    dl_run_button = mo.ui.run_button(label="Generate TSV")
    return (dl_run_button,)


@app.cell
def _(
    build_scalar_sql,
    build_vector_wide_sql,
    conn,
    db_filter,
    dl_delivery_radio,
    dl_disk_dir_input,
    dl_filename_input,
    dl_mode_select,
    dl_run_button,
    dl_scalar_cols_select,
    dl_size_threshold_mb,
    dl_time_interval,
    dl_vector_col_select,
    dl_vector_label_mode,
    estimate_export_size,
    get_column_metadata,
    get_vector_length,
    history_sql_base,
    mo,
    os,
):
    status_md = "_Click **Generate TSV** above to export with the current settings._"
    download_widget = None
    label_source = None
    time_interval = float(dl_time_interval.value or 0.0)

    if dl_run_button.value:
        sql = None
        n_data_cols = 0

        if dl_mode_select.value == "vector column":
            col = dl_vector_col_select.value
            if not col:
                status_md = "**Please select a vector column.**"
            else:
                n_elems = get_vector_length(col, history_sql_base, db_filter)
                if n_elems == 0:
                    status_md = f"**No rows match the current filter for `{col}`.**"
                else:
                    if dl_vector_label_mode.value == "metadata IDs (when available)":
                        labels, label_source = get_column_metadata(
                            col, fallback_len=n_elems
                        )
                    else:
                        labels = [f"idx_{i}" for i in range(n_elems)]
                        label_source = "numeric (user-selected)"
                    if not labels:
                        labels = [f"idx_{i}" for i in range(n_elems)]
                        label_source = label_source or "numeric-fallback:empty-metadata"
                    sql = build_vector_wide_sql(
                        history_sql_base,
                        db_filter,
                        col,
                        labels,
                        time_interval=time_interval,
                    )
                    n_data_cols = len(labels)
        else:
            cols = list(dl_scalar_cols_select.value or [])
            if not cols:
                status_md = "**Please select at least one scalar column.**"
            else:
                sql = build_scalar_sql(
                    history_sql_base,
                    db_filter,
                    cols,
                    time_interval=time_interval,
                )
                n_data_cols = len(cols)

        if sql is not None:
            try:
                n_rows, est_bytes = estimate_export_size(sql, n_data_cols)
            except Exception as e:
                n_rows, est_bytes = 0, 0
                status_md = f"**Size estimate failed:** `{e}`"

            if n_rows == 0 and "failed" not in status_md:
                status_md = "**No rows match the current filter.**"
            elif n_rows > 0:
                threshold_bytes = float(dl_size_threshold_mb.value) * 1e6
                mode = dl_delivery_radio.value
                if mode == "auto (browser if small, disk if large)":
                    mode = (
                        "browser download"
                        if est_bytes <= threshold_bytes
                        else "save to disk"
                    )

                filename = dl_filename_input.value or "export.tsv"

                src_note = (
                    f"  \n*Header labels source:* `{label_source}`"
                    if label_source
                    else ""
                )
                ds_note = (
                    f"  \n*Time downsampling:* one row per "
                    f"`{time_interval}` s bucket per cell."
                    if time_interval > 0
                    else ""
                )
                src_note = src_note + ds_note

                if mode == "browser download":
                    df = conn.sql(sql).pl()
                    buf = df.write_csv(separator="\t").encode("utf-8")
                    download_widget = mo.download(
                        data=buf,
                        filename=filename,
                        label=f"Download {filename}",
                        mimetype="text/tab-separated-values",
                    )
                    status_md = (
                        f"Prepared **{n_rows:,} rows × "
                        f"{n_data_cols + 6} cols** (~{est_bytes / 1e6:.1f} MB "
                        f"estimated) for browser download.{src_note}"
                    )
                else:
                    out_dir = dl_disk_dir_input.value
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, filename)
                    safe_path = out_path.replace("'", "''")
                    copy_sql = (
                        f"COPY ({sql}) TO '{safe_path}' "
                        f"(FORMAT CSV, HEADER, DELIMITER E'\\t')"
                    )
                    conn.sql(copy_sql)
                    size_mb = os.path.getsize(out_path) / 1e6
                    status_md = (
                        f"Wrote **{n_rows:,} rows × "
                        f"{n_data_cols + 6} cols** to "
                        f"`{out_path}` ({size_mb:.1f} MB on disk).{src_note}"
                    )

    download_status_render = mo.vstack(
        [mo.md(status_md), download_widget]
        if download_widget is not None
        else [mo.md(status_md)]
    )
    return (download_status_render,)


@app.cell
def _(mo):
    download_memory_notes_md = mo.md(
        """
    ### Memory notes for large exports

    - **Wide vector exports** can be huge. `listeners__monomer_counts` has
      ~4,500 elements; a single cell with ~5,000 timesteps is roughly
      `5000 × 4500 × ~10 bytes ≈ 225 MB` of TSV text. A multiseed export
      multiplies that by the number of cells included.
    - **Browser download** materializes the full TSV in Python memory
      (`polars.DataFrame.write_csv → bytes`) before handing it to the
      browser. Above ~100 MB the kernel will start to feel it and the
      browser may refuse the blob.
    - **Save-to-disk mode** uses DuckDB's `COPY (...) TO file (FORMAT CSV)`,
      which streams query results to disk without building the full table
      in RAM. This is the safe choice for vectors with more than ~1,000
      elements or for any multi-cell aggregation.
    - **`auto` delivery** switches modes based on the size estimate and the
      threshold (default 50 MB). The estimate uses ~12 bytes/cell as a
      crude average; real sizes vary with dtype (large floats blow it up,
      small ints come in under).
    - **Time downsampler (Δt > 0)** keeps one row per Δt-second bucket per
      cell. Often the easiest way to drop a 200 MB export into the browser
      tier without losing the overall shape of the trace.
    - **Row count estimation** runs a `COUNT(*)` over the filtered
      subquery on every Generate click. On large multi-experiment
      datasets this scan can be the slowest part of the export.
    """
    )
    return (download_memory_notes_md,)


@app.cell
def _(
    about_intro_md,
    analysis_select,
    mo,
    partition_picker_items,
    select_pathway,
):
    # Toolbar lives in its OWN cell so it always renders, even if a downstream
    # chart cell errors. The user can still pick an experiment / variant /
    # pathway from here to recover.
    _partition_row = []
    for _label, _widget in partition_picker_items:
        _partition_row.append(mo.md(f"**{_label}:**"))
        _partition_row.append(_widget)

    mo.vstack(
        [
            mo.md("# vEcoli Output Explorer"),
            mo.hstack([mo.md("**analysis:**"), analysis_select], justify="start"),
            mo.hstack(_partition_row, justify="start"),
            mo.hstack([mo.md("**pathway:**"), select_pathway], justify="start"),
            mo.accordion({"About this notebook": about_intro_md}),
        ]
    )
    return


@app.cell
def _(
    about_compounds_md,
    about_download_md,
    about_mrna_md,
    about_proteins_md,
    about_reactions_md,
    about_validation_md,
    bulk_sp_plot,
    chart_compounds,
    chart_mrna,
    chart_monomers,
    chart_rxns,
    chart_val,
    dl_delivery_radio,
    dl_disk_dir_input,
    dl_filename_input,
    dl_mode_select,
    dl_run_button,
    dl_scalar_cols_select,
    dl_schema_note_md,
    dl_size_threshold_mb,
    dl_time_interval,
    dl_vector_col_select,
    dl_vector_label_mode,
    download_memory_notes_md,
    download_status_render,
    mo,
    molecule_id_type,
    monomer_label_type,
    monomer_select_plot,
    rna_label_type,
    mrna_select_plot,
    select_rxns,
    val_dataset_select,
    val_id_select,
    val_label_type,
    y_scale,
    y_scale_mrna,
    y_scale_monomers,
    y_scale_rxns,
):
    # Tabs live below the toolbar in their own cell. If any chart cell errors
    # (e.g. transient parquet read), this cell shows the error in-place while
    # the toolbar above remains usable.
    def _chart_or_placeholder(chart, msg):
        return chart if chart is not None else mo.md(f"_{msg}_")

    _compounds_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_compounds_md}),
            mo.hstack(
                [
                    mo.md("**label:**"),
                    molecule_id_type,
                    mo.md("**compounds:**"),
                    bulk_sp_plot,
                    mo.md("**scale:**"),
                    y_scale,
                ],
                justify="start",
            ),
            _chart_or_placeholder(
                chart_compounds, "Select compounds to see the chart."
            ),
        ]
    )

    _mrna_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_mrna_md}),
            mo.hstack(
                [
                    mo.md("**label:**"),
                    rna_label_type,
                    mo.md("**genes:**"),
                    mrna_select_plot,
                    mo.md("**scale:**"),
                    y_scale_mrna,
                ],
                justify="start",
            ),
            _chart_or_placeholder(chart_mrna, "Select mRNAs to see the chart."),
        ]
    )

    _proteins_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_proteins_md}),
            mo.hstack(
                [
                    mo.md("**label:**"),
                    monomer_label_type,
                    mo.md("**proteins:**"),
                    monomer_select_plot,
                    mo.md("**scale:**"),
                    y_scale_monomers,
                ],
                justify="start",
            ),
            _chart_or_placeholder(chart_monomers, "Select proteins to see the chart."),
        ]
    )

    _reactions_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_reactions_md}),
            mo.hstack(
                [
                    mo.md("**reaction IDs:**"),
                    select_rxns,
                    mo.md("**scale:**"),
                    y_scale_rxns,
                ],
                justify="start",
            ),
            _chart_or_placeholder(chart_rxns, "Select reactions to see the chart."),
        ]
    )

    _validation_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_validation_md}),
            mo.hstack(
                [
                    mo.md("**dataset:**"),
                    val_dataset_select,
                    mo.md("**label type:**"),
                    val_label_type,
                    mo.md("**proteins:**"),
                    val_id_select,
                ],
                justify="start",
            ),
            _chart_or_placeholder(chart_val, "Select proteins to see the scatter."),
        ]
    )

    # Fixed-width label column so every input lines up vertically.
    def _dl_label(text):
        return mo.md(
            f'<div style="min-width: 240px; display: inline-block;"><b>{text}</b></div>'
        )

    def _dl_row(label, widget):
        return mo.hstack(
            [_dl_label(label), widget],
            justify="start",
            align="center",
            gap=0.5,
        )

    # "What to export" section: mode + column picker(s)
    if dl_mode_select.value == "vector column":
        _picker_rows = [
            _dl_row("Vector column:", dl_vector_col_select),
            _dl_row("Header labels:", dl_vector_label_mode),
        ]
    else:
        _picker_rows = [_dl_row("Scalar columns:", dl_scalar_cols_select)]

    _what_section = mo.vstack(
        [
            mo.md("#### What to export"),
            _dl_row("Export mode:", dl_mode_select),
            *_picker_rows,
        ]
    )

    # "Output options" section
    _output_rows = [
        _dl_row("Filename:", dl_filename_input),
        _dl_row("Delivery:", dl_delivery_radio),
        _dl_row("Auto-switch threshold (MB):", dl_size_threshold_mb),
        _dl_row("Min Δt between samples (s):", dl_time_interval),
    ]
    if dl_delivery_radio.value != "browser download":
        _output_rows.append(_dl_row("Output directory:", dl_disk_dir_input))

    _output_section = mo.vstack([mo.md("#### Output options"), *_output_rows])

    _download_tab = mo.vstack(
        [
            mo.accordion({"About this view": about_download_md}),
            dl_schema_note_md,
            _what_section,
            _output_section,
            mo.md("####  "),
            dl_run_button,
            download_status_render,
            mo.accordion({"Memory notes for large exports": download_memory_notes_md}),
        ]
    )

    mo.ui.tabs(
        {
            "Compounds": _compounds_tab,
            "mRNA": _mrna_tab,
            "Proteins": _proteins_tab,
            "Reactions": _reactions_tab,
            "Validation": _validation_tab,
            "Download": _download_tab,
        }
    )
    return


if __name__ == "__main__":
    app.run()
