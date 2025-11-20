import os
from typing import Any
from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import pickle
from scipy.stats import pearsonr
from wholecell.utils import units
from ecoli.library.parquet_emitter import field_metadata, read_stacked_columns
from ecoli.library.sim_data import LoadSimData
from ecoli.processes.metabolism import VOLUME_UNITS, MASS_UNITS

plt.rcParams["figure.dpi"] = 300


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
    variant_names: dict[str, str],
):
    exp_id = list(sim_data_paths.keys())[0]
    sim_data_path = list(sim_data_paths[exp_id].values())[0]
    sim_data = LoadSimData(sim_data_path).sim_data

    validation_data_path = validation_data_paths[0]
    with open(validation_data_path, "rb") as f:
        validation_data = pickle.load(f)

    toyaReactions = validation_data.reactionFlux.toya2010fluxes["reactionID"]
    toyaFluxes = validation_data.reactionFlux.toya2010fluxes["reactionFlux"]
    toyaStdev = validation_data.reactionFlux.toya2010fluxes["reactionFluxStdev"]
    toyaFluxesDict = dict(zip(toyaReactions, toyaFluxes))
    toyaStdevDict = dict(zip(toyaReactions, toyaStdev))

    base_reaction_ids = field_metadata(
        conn, config_sql, "listeners__fba_results__base_reaction_fluxes"
    )
    base_reaction_id_to_index = {
        rxn_id: i for (i, rxn_id) in enumerate(base_reaction_ids)
    }

    data_columns = [
        "agent_id",
        "generation",
        "lineage_seed",
        "listeners__fba_results__base_reaction_fluxes",
        "listeners__mass__cell_mass",
        "listeners__mass__dry_mass",
    ]

    result = pl.DataFrame(read_stacked_columns(history_sql, data_columns, conn=conn))

    lineage_seeds = np.unique(result["lineage_seed"]).tolist()
    generations = {}

    for seed in lineage_seeds:
        generations[f"lineage_seed={seed}"] = np.unique(
            result.filter(pl.col("lineage_seed") == seed)["generation"]
        ).tolist()

    agents = {}

    for seed_key in generations.keys():
        seed = int(seed_key.split("=")[1])
        seed_gens = generations[seed_key]
        agents_gen = {}
        for generation in seed_gens:
            results_filtered = result.filter(pl.col("lineage_seed") == seed).filter(
                pl.col("generation") == generation
            )
            agents_gen[f"generation={generation}"] = np.unique(
                results_filtered["agent_id"]
            ).tolist()
        agents[seed_key] = agents_gen

    cellDensity = sim_data.constants.cell_density
    mmol_per_g_per_h = units.mmol / units.g / units.h

    cellMass = result["listeners__mass__cell_mass"]
    dryMass = result["listeners__mass__dry_mass"]
    coefficient = dryMass / cellMass * cellDensity.asNumber(MASS_UNITS / VOLUME_UNITS)

    fluxes_full = np.stack(result["listeners__fba_results__base_reaction_fluxes"])
    fluxes_converted = np.array(
        [
            fluxes_full[row_idx, :] / coefficient[row_idx] * 3600
            for row_idx in range(len(coefficient))
        ]
    )

    result = result.insert_column(
        result.shape[1] - 1, pl.Series("fluxes_converted", fluxes_converted)
    )

    modelFluxes = {}

    for toyaReaction in toyaReactions:
        modelFluxes[toyaReaction] = []

    for seed_key in agents.keys():
        seed = int(seed_key.split("=")[1])
        for gen_key in agents[seed_key].keys():
            gen = int(gen_key.split("=")[1])
            for agent in agents[seed_key][gen_key]:
                result_agent = (
                    result.filter(pl.col("lineage_seed") == seed)
                    .filter(pl.col("generation") == gen)
                    .filter(pl.col("agent_id") == agent)
                )
                agent_fluxes = np.stack(result_agent["fluxes_converted"])
                for toyaReaction in toyaReactions:
                    rxn_index = base_reaction_id_to_index[toyaReaction]
                    fluxTimeCourse = agent_fluxes[:, rxn_index]
                    modelFluxes[toyaReaction].append(np.mean(fluxTimeCourse))

    toyaVsReactionAve = []
    for rxn, toyaFlux in toyaFluxesDict.items():
        if rxn in modelFluxes:
            toyaVsReactionAve.append(
                (
                    np.mean(modelFluxes[rxn]),
                    toyaFlux.asNumber(mmol_per_g_per_h),
                    np.std(modelFluxes[rxn]),
                    toyaStdevDict[rxn].asNumber(mmol_per_g_per_h),
                )
            )

    toyaVsReactionAve = np.array(toyaVsReactionAve)
    rWithAll = pearsonr(toyaVsReactionAve[:, 0], toyaVsReactionAve[:, 1])

    plt.figure(figsize=[8, 8])
    ax = plt.axes()
    plt.title(
        "Central Carbon Metabolism Flux, Pearson R = %.4f, p = %s\n"
        % (rWithAll[0], rWithAll[1])
    )

    plt.xlabel("Toya 2010 Reaction Flux [mmol/g/hr]")
    plt.ylabel("Mean WCM Reaction Flux [mmol/g/hr]")

    plt.errorbar(
        toyaVsReactionAve[:, 1],
        toyaVsReactionAve[:, 0],
        xerr=toyaVsReactionAve[:, 3],
        yerr=toyaVsReactionAve[:, 2],
        fmt=".",
        ecolor="k",
        alpha=0.5,
        linewidth=0.5,
    )
    ylim = plt.ylim()
    plt.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], color="k")

    plt.plot(toyaVsReactionAve[:, 1], toyaVsReactionAve[:, 1], color="black")

    plt.plot(
        toyaVsReactionAve[:, 1],
        toyaVsReactionAve[:, 0],
        "ob",
        markeredgewidth=0.1,
        alpha=0.9,
    )

    ax.set_xlim([-20, 30])
    ax.set_ylim([-20, 70])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_yticks(list(range(int(ylim[0]), int(ylim[1]) + 1, 10)))
    ax.set_xticks(list(range(int(xlim[0]), int(xlim[1]) + 1, 10)))
    plt.savefig(os.path.join(outdir, "centralCarbonMetabolismScatter.png"))
