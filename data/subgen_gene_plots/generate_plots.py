"""
This script creates a bar graph showing the number of antibiotic response genes and
the number of all genes that are sub-generational vs. generational. We define
generational genes as those that are expressed at least once in all agents.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# Generational genes from Science paper (Fig 4c) minus one
CUTOFF_INDEX = 1546

# antibiotic_response_genes.txt was created by listing out all the genes EcoCyc
# considers related to antibiotic response in E. Coli
# (https://ecocyc.org/ECOLI/NEW-IMAGE?type=ECOCYC-CLASS&object=GO:0046677).
# ompF and ompC (porins) were also included in this txt file even though they
# were not listed on EcoCyc.
RESPONSE_GENES_PATH = 'data/subgen_gene_plots/antibiotic_response_genes.txt'
# rnas.tsv file is from the release version of wcEcoli
RNAS_TSV_PATH = 'reconstruction/ecoli/flat/rnas.tsv'
SIM_DATA_PATH = 'reconstruction/sim_data/kb/simData.cPickle'

def convert_dict_to_df(data):
    mRNA_counts = []
    times = []
    agent_ids = []
    time = data[0]
    time_data = data[1]
    for agent_id, agent_data in time_data['agents'].items():
        if 'listeners' not in agent_data:
            continue
        agent_ids.append(agent_id)
        times.append(time)
        mRNA_counts.append(agent_data['listeners']['mRNA_counts'])
    df = pd.DataFrame(mRNA_counts)
    df['Time'] = times
    df['Agent ID'] = agent_ids
    return df

def make_antibiotic_subgen_plot(data):
    sim_data = pickle.load(open(SIM_DATA_PATH, 'rb'))
    # Get indices of antibiotic response genes in mRNA count array
    all_TU_ids = sim_data.process.transcription.rna_data['id']
    mrna_indices = np.where(sim_data.process.transcription.rna_data[
        'is_mRNA'])[0]
    rnas = pd.read_csv(RNAS_TSV_PATH, sep='\t', comment='#')
    response_genes = pd.read_csv(RESPONSE_GENES_PATH, header=None)
    response_genes.rename(columns={0: 'common_name'}, inplace=True)
    response_rnas = rnas.merge(response_genes, how='inner', on='common_name')
    response_rnas['id'] = response_rnas['id'].apply(lambda x: f'{x}[c]')
    response_TU_ids = np.where(np.isin(all_TU_ids, response_rnas['id']))[0]
    response_mRNA_indices = np.where(np.isin(mrna_indices, response_TU_ids))[0]

    with ProcessPoolExecutor(30) as executor:
        print('Converting data to DataFrame...')
        time_dfs = list(tqdm(executor.map(convert_dict_to_df, data.items()),
            total=len(data)))
    data = pd.concat(time_dfs)
    data.to_pickle(f'data/glc_10000_transcriptome_df.pkl')

    grouped_agents = data.groupby('Agent ID')
    mRNA_expression = np.zeros_like(mrna_indices)
    n_agents = 0
    for _, agent_data in grouped_agents:
        agent_data = agent_data.sort_values('Time')
        agent_data = agent_data.diff()
        mRNA_expression[(agent_data>0).any(axis='rows').to_numpy()] += 1
        n_agents += 1
    
    generational = (mRNA_expression==n_agents).sum()
    subgenerational = len(mRNA_expression) - generational
    assert subgenerational == (mRNA_expression!=n_agents).sum()
    print(f'Generational genes (total): {generational}')
    print(f'Sub-generational genes (total): {subgenerational}')

    response_expression = mRNA_expression[response_mRNA_indices]
    response_generational = (response_expression==n_agents).sum()
    response_subgenerational = len(response_expression) - response_generational
    assert response_subgenerational == (response_expression!=n_agents).sum()
    print('Generational genes (antibiotic response): '
        f'{response_generational}')
    print('Sub-generational genes (antibiotic response): '
        f'{response_subgenerational}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        help="Saved transcriptome pickle from ecoli.analysis.db.get_transcriptome_data"
    )
    args = parser.parse_args()
    data = pickle.load(open(args.data, 'rb'))
    make_antibiotic_subgen_plot(data)