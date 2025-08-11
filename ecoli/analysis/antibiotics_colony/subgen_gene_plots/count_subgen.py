"""
This script creates a bar graph showing the number of antibiotic response genes
and the number of all genes that are sub-generational vs. generational. We
define generational genes as those that are expressed at least once per agent,
on average.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import os
import pandas as pd
import pickle
from tqdm import tqdm

# antibiotic_response_genes.txt was created by listing out all the genes EcoCyc
# considers related to antibiotic response in E. Coli
# (https://ecocyc.org/ECOLI/NEW-IMAGE?type=ECOCYC-CLASS&object=GO:0046677).
# ompF and ompC (porins) were also included in this txt file even though they
# were not listed on EcoCyc.
RESPONSE_GENES_PATH = (
    "ecoli/analysis/antibiotics_colony/subgen_gene_plots/"
    + "antibiotic_response_genes.txt"
)
# rnas.tsv file is from the release version of wcEcoli
RNAS_TSV_PATH = "reconstruction/ecoli/flat/rnas.tsv"
SIM_DATA_PATH = "out/kb/simData.cPickle"


def convert_dict_to_df(data):
    mRNA_counts = []
    times = []
    agent_ids = []
    time = data[0]
    time_data = data[1]
    for agent_id, agent_data in time_data["agents"].items():
        if "listeners" not in agent_data:
            continue
        agent_ids.append(agent_id)
        times.append(time)
        mRNA_counts.append(
            agent_data["listeners"]["transcript_elongation_listener"][
                "countRnaSynthesized"
            ]
        )
    df = pd.DataFrame(mRNA_counts)
    df["Time"] = times
    df["Agent ID"] = agent_ids
    return df


def count_antibiotic_subgen(data):
    sim_data = pickle.load(open(SIM_DATA_PATH, "rb"))
    # Get indices of antibiotic response genes in mRNA count array
    all_TU_ids = sim_data.process.transcription.rna_data["id"]
    rnas = pd.read_csv(RNAS_TSV_PATH, sep="\t", comment="#")
    response_genes = pd.read_csv(RESPONSE_GENES_PATH, header=None)
    response_genes.rename(columns={0: "common_name"}, inplace=True)
    response_rnas = rnas.merge(response_genes, how="inner", on="common_name")
    response_rnas["id"] = response_rnas["id"].apply(lambda x: f"{x}[c]")
    response_TU_ids = np.where(np.isin(all_TU_ids, response_rnas["id"]))[0]

    data_columns = ~np.isin(data.columns, ["Agent ID", "Time"])
    n_agents = len(data.loc[:, "Agent ID"].unique())
    mRNA_expression = data.loc[:, data_columns].sum(axis=0) / n_agents

    # Genes that are expressed on average 1+ times per agent are generational
    generational = (mRNA_expression >= 1).sum()
    subgenerational = len(mRNA_expression) - generational
    assert subgenerational == (mRNA_expression < 1).sum()
    print(f"Generational genes (total): {generational}")
    print(f"Sub-generational genes (total): {subgenerational}")

    response_expression = mRNA_expression[response_TU_ids]
    response_generational = (response_expression >= 1).sum()
    response_subgenerational = len(response_expression) - response_generational
    assert response_subgenerational == (response_expression < 1).sum()
    print(f"Generational genes (antibiotic response): {response_generational}")
    print(f"Sub-generational genes (antibiotic response): {response_subgenerational}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-r", "--raw_data", help="Saved from ecoli.analysis.db.get_gene_expression_data"
    )
    group.add_argument(
        "-d",
        "--data",
        help="Saved csv file from running this script once with raw data",
    )
    args = parser.parse_args()
    # TODO: Convert to use DuckDB
    raise NotImplementedError("Still need to convert to use DuckDB!")
    if args.raw_data:
        data = pickle.load(open(args.data, "rb"))
        with ProcessPoolExecutor(30) as executor:
            print("Converting data to DataFrame...")
            time_dfs = list(
                tqdm(executor.map(convert_dict_to_df, data.items()), total=len(data))
            )
        data = pd.concat(time_dfs)
        os.makedirs("data/colony_data/", exist_ok=True)
        data.to_csv("data/colony_data/glc_10000_expressome_df.csv")
    else:
        data = pd.read_csv(args.data, dtype={"Agent ID": str, "Seed": str}, index_col=0)
    count_antibiotic_subgen(data)
