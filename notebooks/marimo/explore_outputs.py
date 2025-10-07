import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import numpy as np
    import pandas as pd
    import duckdb
    import sys
    import altair as alt

    return alt, duckdb, mo, np, os, pd, sys


@app.cell
def _(os, sys):
    wd_root = os.getcwd().split("/notebooks")[0]

    sys.path.append(wd_root)

    from ecoli.library.sim_data import LoadSimData
    from ecoli.library.parquet_emitter import dataset_sql

    sim_data_path = os.path.join(
        wd_root, "reconstruction", "sim_data", "kb", "simData.cPickle"
    )

    sim_data = LoadSimData(sim_data_path).sim_data
    return LoadSimData, dataset_sql, sim_data, sim_data_path, wd_root


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
def _(get_bulk_ids, get_rxn_ids, sim_data, sim_data_path):
    bulk_ids = get_bulk_ids(sim_data_path)
    rxn_ids = get_rxn_ids(sim_data_path)
    cistron_data = sim_data.process.transcription.cistron_data
    mrna_cistron_ids = cistron_data["id"][cistron_data["is_mRNA"]].tolist()
    mrna_cistron_names = [
        sim_data.common_names.get_common_name(cistron_id)
        for cistron_id in mrna_cistron_ids
    ]
    return bulk_ids, mrna_cistron_names, rxn_ids


@app.cell
def _(mo):
    analysis_select = mo.ui.dropdown(
        options=["single", "multidaughter", "multigeneration", "multiseed"],
        value="single",
    )
    mo.hstack(["analysis type:", analysis_select], justify="start")
    return (analysis_select,)


@app.cell
def _(bulk_ids, mo, os, wd_root):
    sp_select = mo.ui.multiselect(options=bulk_ids)
    exp_select = mo.ui.dropdown(options=os.listdir(os.path.join(wd_root, "out")))
    y_scale = mo.ui.dropdown(options=["linear", "log", "symlog"], value="linear")

    return exp_select, sp_select, y_scale


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
def _(exp_select, get_variants, mo):
    variant_select = mo.ui.dropdown(options=get_variants(exp_id=exp_select.value))
    return (variant_select,)


@app.cell
def _(exp_select, get_seeds, mo, variant_select):
    seed_select = mo.ui.dropdown(
        options=get_seeds(exp_id=exp_select.value, var_id=variant_select.value)
    )
    return (seed_select,)


@app.cell
def _(exp_select, get_gens, mo, seed_select, variant_select):
    gen_select = mo.ui.dropdown(
        options=get_gens(
            exp_id=exp_select.value,
            var_id=variant_select.value,
            seed_id=seed_select.value,
        )
    )
    return (gen_select,)


@app.cell
def _(exp_select, gen_select, get_agents, mo, seed_select, variant_select):
    agent_select = mo.ui.dropdown(
        options=get_agents(
            exp_id=exp_select.value,
            var_id=variant_select.value,
            seed_id=seed_select.value,
            gen_id=gen_select.value,
        )
    )
    return (agent_select,)


@app.cell
def _(analysis_select, get_db_filter, partitions_dict):
    # dbf_dict = partitions_dict(exp_select.value,variant_select.value,seed_select.value,gen_select.value,agent_select.value)
    dbf_dict = partitions_dict(analysis_select.value)
    db_filter = get_db_filter(dbf_dict)

    return (db_filter,)


@app.cell
def _(dataset_sql, db_filter, exp_select, os, wd_root):
    pq_columns = [
        "bulk",
        "listeners__fba_results__base_reaction_fluxes",
        "listeners__rna_counts__full_mRNA_cistron_counts",
    ]

    history_sql_base, _, _ = dataset_sql(
        os.path.join(wd_root, "out"), experiment_ids=[exp_select.value]
    )
    history_sql_filtered = f"SELECT {','.join(pq_columns)},time FROM ({history_sql_base}) WHERE {db_filter} ORDER BY time"
    return (history_sql_filtered,)


@app.cell
def _(bulk_ids, history_sql_filtered, load_outputs, np, sp_select):
    output_loaded = load_outputs(history_sql_filtered)

    bulk_mtx = np.stack(output_loaded["bulk"].values)

    sp_idxs = [bulk_ids.index(bulk_id) for bulk_id in sp_select.value]

    sp_trajs = [bulk_mtx[:, sp_idx] for sp_idx in sp_idxs]

    return output_loaded, sp_trajs


@app.cell
def _(output_loaded, pd, sp_select, sp_trajs):
    plot_dict = {key: val for (key, val) in zip(sp_select.value, sp_trajs)}

    plot_dict["time"] = output_loaded["time"]

    plot_df = pd.DataFrame(plot_dict)

    return (plot_df,)


@app.cell
def _(downsample, plot_df):
    df_long = plot_df.melt(
        id_vars=["time"],  # Columns to keep as identifier variables
        var_name="bulk_molecules",  # Name for the new column containing original column headers
        value_name="counts",  # Name for the new column containing original column values
    )

    dfds_long = downsample(df_long)
    return (dfds_long,)


@app.cell
def _(mo):
    mo.md("""bulk molecule counts""")
    return


@app.cell
def _(mo, sp_select, y_scale):
    mo.hstack(["bulk id(s):", sp_select, "scale:", y_scale], justify="start")
    return


@app.cell
def _(alt, dfds_long, y_scale):
    alt.Chart(dfds_long).mark_line().encode(
        x=alt.X("time:Q", scale=alt.Scale(type="linear"), axis=alt.Axis(tickCount=4)),
        y=alt.Y("counts:Q", scale=alt.Scale(type=y_scale.value)),
        color="bulk_molecules:N",
    )
    return


@app.cell
def _(mo):
    mo.md("""mRNA counts""")
    return


@app.cell
def _(mo, mrna_cistron_names):
    mrna_select = mo.ui.multiselect(options=mrna_cistron_names)
    y_scale_mrna = mo.ui.dropdown(options=["linear", "log"], value="log")
    mo.hstack(["gene name(s):", mrna_select, "scale:", y_scale_mrna], justify="start")
    return mrna_select, y_scale_mrna


@app.cell
def _(downsample, mrna_cistron_names, mrna_select, np, output_loaded, pd):
    mrna_mtx = np.stack(
        output_loaded["listeners__rna_counts__full_mRNA_cistron_counts"]
    )

    mrna_idxs = [mrna_cistron_names.index(gene_id) for gene_id in mrna_select.value]

    mrna_trajs = [mrna_mtx[:, mrna_idx] for mrna_idx in mrna_idxs]

    mrna_plot_dict = {key: val for (key, val) in zip(mrna_select.value, mrna_trajs)}

    mrna_plot_dict["time"] = output_loaded["time"]

    mrna_plot_df = pd.DataFrame(mrna_plot_dict)

    mrna_df_long = mrna_plot_df.melt(
        id_vars=["time"],  # Columns to keep as identifier variables
        var_name="gene names",  # Name for the new column containing original column headers
        value_name="counts",  # Name for the new column containing original column values
    )

    mrna_dfds_long = downsample(mrna_df_long)
    return (mrna_dfds_long,)


@app.cell
def _(alt, mrna_dfds_long, y_scale_mrna):
    alt.Chart(mrna_dfds_long).mark_line().encode(
        x=alt.X("time:Q", scale=alt.Scale(type="linear"), axis=alt.Axis(tickCount=4)),
        y=alt.Y("counts:Q", scale=alt.Scale(type=y_scale_mrna.value)),
        color="gene names:N",
    )
    return


@app.cell
def _(mo):
    mo.md("""reaction fluxes""")
    return


@app.cell
def _(mo, rxn_ids):
    select_rxns = mo.ui.multiselect(options=rxn_ids)
    y_scale_rxns = mo.ui.dropdown(options=["linear", "log"], value="log")
    mo.hstack(["reaction id(s):", select_rxns, "scale:", y_scale_rxns], justify="start")
    return select_rxns, y_scale_rxns


@app.cell
def _(downsample, np, output_loaded, pd, rxn_ids, select_rxns):
    rxns_mtx = np.stack(
        output_loaded["listeners__fba_results__base_reaction_fluxes"].values
    )

    rxns_idxs = [rxn_ids.index(rxn) for rxn in select_rxns.value]

    rxn_trajs = [rxns_mtx[:, rxn_idx] for rxn_idx in rxns_idxs]

    plot_rxns_dict = {key: val for (key, val) in zip(select_rxns.value, rxn_trajs)}

    plot_rxns_dict["time"] = output_loaded["time"]

    plot_rxns_df = pd.DataFrame(plot_rxns_dict)

    rxns_df_long = plot_rxns_df.melt(
        id_vars=["time"],  # Columns to keep as identifier variables
        var_name="reaction_id",  # Name for the new column containing original column headers
        value_name="flux",  # Name for the new column containing original column values
    )

    rxns_dfds_long = downsample(rxns_df_long)
    return (rxns_dfds_long,)


@app.cell
def _(alt, rxns_dfds_long, y_scale_rxns):
    alt.Chart(rxns_dfds_long).mark_line().encode(
        x=alt.X("time:Q", scale=alt.Scale(type="linear"), axis=alt.Axis(tickCount=4)),
        y=alt.Y("flux:Q", scale=alt.Scale(type=y_scale_rxns.value)),
        color="reaction_id:N",
    )
    return


@app.cell
def _(exp_select, os, wd_root):
    def get_variants(exp_id, outdir=os.path.join(wd_root, "out")):
        try:
            vars_ls = os.listdir(
                os.path.join(
                    outdir,
                    exp_select.value,
                    "history",
                    f"experiment_id={exp_select.value}",
                )
            )

            variant_folders = [
                folder for folder in vars_ls if not folder.startswith(".")
            ]

            variants = [var.split("variant=")[1] for var in variant_folders]

        except (FileNotFoundError, TypeError):
            variants = ["N/A"]

        return variants

    def get_seeds(exp_id, var_id, outdir=os.path.join(wd_root, "out")):
        try:
            seeds_ls = os.listdir(
                os.path.join(
                    outdir,
                    exp_select.value,
                    "history",
                    f"experiment_id={exp_select.value}",
                    f"variant={var_id}",
                )
            )
            seed_folders = [folder for folder in seeds_ls if not folder.startswith(".")]

            seeds = [seed.split("lineage_seed=")[1] for seed in seed_folders]
        except (FileNotFoundError, TypeError):
            seeds = ["N/A"]

        return seeds

    def get_gens(exp_id, var_id, seed_id, outdir=os.path.join(wd_root, "out")):
        try:
            gens_ls = os.listdir(
                os.path.join(
                    outdir,
                    exp_select.value,
                    "history",
                    f"experiment_id={exp_select.value}",
                    f"variant={var_id}",
                    f"lineage_seed={seed_id}",
                )
            )

            gen_folders = [folder for folder in gens_ls if not folder.startswith(".")]

            gens = [gen.split("generation=")[1] for gen in gen_folders]
        except (FileNotFoundError, TypeError):
            gens = ["N/A"]

        return gens

    def get_agents(
        exp_id, var_id, seed_id, gen_id, outdir=os.path.join(wd_root, "out")
    ):
        try:
            agents_ls = os.listdir(
                os.path.join(
                    outdir,
                    exp_select.value,
                    "history",
                    f"experiment_id={exp_select.value}",
                    f"variant={var_id}",
                    f"lineage_seed={seed_id}",
                    f"generation={gen_id}",
                )
            )

            agent_folders = [
                folder for folder in agents_ls if not folder.startswith(".")
            ]
            agents = [agent.split("agent_id=")[1] for agent in agent_folders]
        except (FileNotFoundError, TypeError):
            agents = ["N/A"]

        return agents

    return get_agents, get_gens, get_seeds, get_variants


@app.cell
def _(partition_groups, read_partitions):
    # def partitions_dict(exp_id,var_id,seed_id,gen_id,agent_id):
    #     return {"experiment_id":f"'{exp_id}'", "variant":var_id, "lineage_seed":seed_id, "generation":gen_id, "agent_id":agent_id}

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


@app.cell
def _(duckdb, itertools, np):
    def load_outputs(sql):
        outputs_df = duckdb.sql(sql).df()
        outputs_df = outputs_df.groupby("time", as_index=False).sum()

        return outputs_df

    def downsample(df_long):
        tp_all = np.unique(df_long["time"]).astype(int)
        ds_ratio = int(np.ceil(np.shape(df_long)[0] / 20000))
        tp_ds = list(itertools.islice(tp_all, 0, max(tp_all), ds_ratio))
        df_ds = df_long[np.isin(df_long["time"], tp_ds)]

        return df_ds

    return downsample, load_outputs


@app.cell
def _():
    import itertools

    return (itertools,)


@app.cell
def _(np):
    int(np.ceil(0.78))
    return


if __name__ == "__main__":
    app.run()
