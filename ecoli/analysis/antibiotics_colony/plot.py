import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib
matplotlib.use('agg')
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from vivarium.library.dict_utils import get_value_from_path

from ecoli.analysis.db import access_counts, deserialize_and_remove_units
from ecoli.analysis.antibiotics_colony.timeseries import (
    plot_snapshot_timeseries,
    plot_generational_timeseries,
    plot_concentration_timeseries)
from ecoli.analysis.antibiotics_colony.distributions import (
    plot_final_distributions,
    plot_death_distributions)
from ecoli.analysis.antibiotics_colony.miscellaneous import (
    plot_colony_growth_rates,
    # plot_environment_concentrations,
    plot_final_fold_changes
)
from ecoli.analysis.antibiotics_colony import DE_GENES

FIGURE_MAPPING = {
    '2B': plot_snapshot_timeseries,
    '2C': plot_concentration_timeseries,
    '2D': plot_generational_timeseries,
    '2E': plot_final_distributions,
    '2F': plot_colony_growth_rates,
    # '2G': plot_environment_concentrations,
    '3B': plot_snapshot_timeseries,
    '3C': plot_concentration_timeseries,
    '3D': plot_generational_timeseries,
    '3E': plot_final_distributions,
    '3F': plot_colony_growth_rates,
    '3G': plot_final_fold_changes,
    '4B': plot_snapshot_timeseries,
    '4C': plot_concentration_timeseries,
    '4D': plot_generational_timeseries,
    '4E': plot_final_distributions,
    '4F': plot_colony_growth_rates,
    '4G': plot_death_distributions,
    # Figure 5 exploring longer running sims for antibiotics
    # since MIC is typically measured after 24 hr incubation?
}

# Condition -> Seed -> Experiment ID
EXPERIMENT_ID_MAPPING = {
    'Glucose': {
        0: '2022-10-25_04-55-23_505282+0000',
        # 100: '2022-10-25_04-55-42_566175+0000',
        # 10000: '2022-10-25_04-55-58_237473+0000',
    },
    'Tetracycline (1.5 mg/L)': {
        0: '2022-10-26_04-50-26_679331+0000',
        # 100: '2022-10-26_04-50-44_985109+0000',
        # 10000: '2022-10-26_04-51-09_119248+0000',
    },
    # 'Tetracycline (1 mg/L)': {
    #     0: '2022-10-26_16-14-20_340871+0000',
    # },
    # 'Tetracycline (0.5 mg/L)': {
    #     0: '2022-10-26_16-12-59_155560+0000',
    # },
    # 'Tetracycline (0.1 mg/L)': {
    #     0: '2022-10-26_16-13-30_275056+0000',
    # },
    'Ampicillin (2 mg/L)': {
        0: '2022-10-25_23-34-42_953042+0000',
        # 100: '2022-10-25_23-35-09_774093+0000',
        # 10000: '2022-10-25_23-35-33_067785+0000',
    },
    # 'Ampicillin (4 mg/L)': {
    #     0: '2022-10-26_16-09-20_929686+0000',
    # },
    # 'Ampicillin (6 mg/L)': {
    #     0: '2022-10-26_16-10-03_405620+0000',
    # },
    # 'Ampicillin (8 mg/L)': {
    #     0: '2022-10-26_16-10-38_989889+0000',
    # },
    # 'Ampicillin (10 mg/L)': {
    #     0: '2022-10-26_16-11-10_712722+0000',
    # },
}

# EXPERIMENT_ID_MAPPING = {
#     'Glucose': {
#         0: '2022-10-21_00-01-40_803264+0000',
#     },
#     'Tetracycline (1.5 mg/L)': {
#         0: '2022-10-21_02-22-25_372025+0000',
#     },
#     'Ampicillin (2 mg/L)': {
#         0: '2022-10-21_02-22-25_372025+0000',
#     },
# }

PATHS_TO_LOAD = {
    'Dry mass': ('listeners', 'mass', 'dry_mass'),
    'AcrAB-TolC': ('bulk', 'TRANS-CPLX-201[m]'),
    'Periplasmic tetracycline': ('periplasm', 'concentrations', 'tetracycline'),
    'Cytoplasmic tetracycline': ('cytoplasm', 'concentrations', 'tetracycline'),
    'Active MarR': ('bulk', 'CPLX0-7710[c]'),
    'Inactive MarR': ('bulk', 'marR-tet[c]'),
    'micF-ompF duplex': ('bulk', 'micF-ompF[c]'),
    'Inactive 30S subunit': ('bulk', 'CPLX0-3953-tetracycline[c]'),
    'Active ribosomes': ('listeners', 'aggregated', 'active_ribosome_len'),
    'Outer tet. permeability (cm/s)': ('kinetic_parameters', 'outer_tetracycline_permeability'),
    'Murein tetramer': ('bulk', 'CPD-12261[p]'),
    'PBP1a complex': ('bulk', 'CPLX0-7717[m]'),
    'PBP1a mRNA': ('mrna', 'EG10748_RNA'),
    'PBP1b alpha complex': ('bulk', 'CPLX0-3951[i]'),
    'PBP1b mRNA': ('mrna', 'EG10605_RNA'),
    'PBP1b gamma complex': ('bulk', 'CPLX0-8300[c]'),
    'Death': ('burst',),
    'Wall cracked': ('wall_state', 'cracked'),
    'AmpC monomer': ('monomer', 'EG10040-MONOMER'),
    'ampC mRNA': ('mrna', 'EG10040_RNA'),
    'Boundary': ('boundary',),
    'Volume': ('listeners', 'mass', 'volume')
}

for gene_data in DE_GENES[['Gene name', 'id', 'monomer_ids']].values:
    if gene_data[0] != 'MicF':
        PATHS_TO_LOAD[f'{gene_data[0]} mRNA'] = ('mrna', gene_data[1])
    gene_data[2] = eval(gene_data[2])
    if len(gene_data[2]) > 0:
        monomer_name = gene_data[0][0].upper() + gene_data[0][1:]
        PATHS_TO_LOAD[f'{monomer_name} monomer'] = (
            'monomer', gene_data[2][0])


def agent_data_table(raw_data, paths_dict, condition, seed):
    """Combine data from all agents into DataFrames for each timestep.
    
    Args:
        raw_data: Tuple of (time, dictionary at time for one replicate).
        paths_dict: Dictionary mapping paths within each agent to names
            that will be used the keys in the returned dictionary.
        condition: String identifier for experimental condition
        seed: Initial seed for this replicate
    
    Returns:
        Dataframe where each column is a path and each row is an agent."""
    time = raw_data[0]
    raw_data = raw_data[1]
    collected_data = {'Agent ID': []}
    agents_at_time = raw_data['agents']
    for agent_id, agent_at_time in agents_at_time.items():
        collected_data['Agent ID'].append(agent_id)
        for name, path in paths_dict.items():
            if name not in collected_data:
                collected_data[name] = []
            value_in_agent = get_value_from_path(agent_at_time, path)
            # Replace missing values with 0
            if value_in_agent == None:
                value_in_agent = 0
            collected_data[name].append(value_in_agent)
    collected_data = pd.DataFrame(collected_data)
    collected_data['Time'] = [time] * len(collected_data)
    collected_data['Seed'] = [seed] * len(collected_data)
    collected_data['Condition'] = [condition] * len(collected_data)
    return collected_data


def load_data(cpus=8, sampling_rate=2, host="10.138.0.75", port=27017):
    # Get the required data
    monomers = [path[-1] for path in PATHS_TO_LOAD.values() if path[0]=='monomer']
    mrnas = [path[-1] for path in PATHS_TO_LOAD.values() if path[0]=='mrna']
    inner_paths = [path for path in PATHS_TO_LOAD.values() 
        if path[-1] not in mrnas and path[-1] not in monomers]
    outer_paths = [('data', 'dimensions'), ('data', 'fields')]
    all_data = {}
    for condition, seeds in EXPERIMENT_ID_MAPPING.items():
        all_data.setdefault(condition, {})
        for seed, experiment_id in seeds.items():
            all_data[condition].setdefault(seed, {})
            data = access_counts(
                experiment_id=experiment_id,
                monomer_names=monomers,
                mrna_names=mrnas,
                inner_paths=inner_paths,
                outer_paths=outer_paths,
                host=host,
                port=port,
                sampling_rate=sampling_rate,
                cpus=36)
            with ProcessPoolExecutor(cpus) as executor:
                print('Deserializing data and removing units...')
                deserialized_data = list(tqdm(executor.map(
                    deserialize_and_remove_units, data.values()),
                    total=len(data)))
            data = dict(zip(data.keys(), deserialized_data))
            # Get spatial environment data for snapshot plots
            print('Extracting spatial environment data...')
            all_data[condition][seed]['bounds'] = data[
                min(data)]['dimensions']['bounds']
            all_data[condition][seed]['fields'] = {
                time: data_at_time['fields'] 
                for time, data_at_time in data.items()
            }
            agent_df_paths = partial(agent_data_table,
                paths_dict=PATHS_TO_LOAD, condition=condition, seed=seed)
            with ProcessPoolExecutor(cpus) as executor:
                print('Converting data to DataFrame...')
                data = list(tqdm(executor.map(agent_df_paths, data.items()),
                    total=len(data)))
            data = pd.concat(data, ignore_index=True)
            all_data[condition][seed]['df'] = data
    return all_data


def main():
    sns.set_style('white')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--figure_ids', '-f', type=str, nargs='+', default=[],
        help="""List of Figure IDs to make. See keys of ``FIGURE_MAPPING``
        for valid types."""
    )
    parser.add_argument(
        '--sampling_rate', '-s', type=int, default=2,
        help="""Custom sampling rate for pulling data from MongoDB"""
    )
    parser.add_argument(
        '--cpus', '-c', type=int, default=8,
        help="""Number of CPU cores to use."""
    )
    parser.add_argument(
        '--hostname', '-n', type=str, default="10.138.0.75",
        help="""Hostname for MongoDB."""
    )
    parser.add_argument(
        '--port', '-p', type=int, default=27017,
        help="""Port for MongoDB."""
    )
    args = parser.parse_args()
    data = load_data(
        cpus=args.cpus,
        sampling_rate=args.sampling_rate,
        host=args.hostname,
        port=args.port)
    figure_ids = args.figure_ids
    if len(figure_ids) == 0:
        figure_ids = FIGURE_MAPPING.keys()
    os.makedirs('out/analysis/antibiotics_colony/', exist_ok=True)
    for figure_id in figure_ids:
        print(f'Making Figure {figure_id}...')
        FIGURE_MAPPING[figure_id](data)


if __name__ == '__main__':
    main()
