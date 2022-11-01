import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import pickle

from vivarium.library.dict_utils import get_value_from_path

from ecoli.analysis.db import access_counts, deserialize_and_remove_units
from ecoli.plots.snapshots import plot_tags
from ecoli.analysis.antibiotics_colony.timeseries import (
    plot_snapshot_timeseries,
    plot_generational_timeseries,
    plot_concentration_timeseries)
from ecoli.analysis.antibiotics_colony.distributions import (
    plot_final_distributions,
    plot_death_distributions)
from ecoli.analysis.antibiotics_colony.miscellaneous import (
    plot_colony_growth_rates,
    plot_final_fold_changes
)
from ecoli.analysis.antibiotics_colony import DE_GENES, MAX_TIME

# Condition -> Seed -> Experiment ID
EXPERIMENT_ID_MAPPING = {
    'Glucose': {
        0: '2022-10-25_04-55-23_505282+0000',
        100: '2022-10-25_04-55-42_566175+0000',
        10000: '2022-10-25_04-55-58_237473+0000',
    },
    'Tetracycline (1.5 mg/L)': {
        0: '2022-10-30_08-46-46_378082+0000',
        100: '2022-10-30_08-46-56_187346+0000',
        10000: '2022-10-30_08-47-08_473173+0000',
    },
    'Tetracycline (4 mg/L)': {
        0: '2022-10-30_08-47-15_257703+0000',
    },
    'Tetracycline (2 mg/L)': {
        0: '2022-10-30_08-47-21_656090+0000',
    },
    'Tetracycline (1 mg/L)': {
        0: '2022-10-30_08-47-27_295650+0000',
    },
    'Tetracycline (0.5 mg/L)': {
        0: '2022-10-30_08-47-34_723561+0000',
    },
    'Ampicillin (2 mg/L)': {
        0: '2022-10-28_05-47-52_977686+0000',
        100: '2022-10-28_05-48-14_864394+0000',
        10000: '2022-10-29_04-41-58_544174+0000',
    },
    'Ampicillin (4 mg/L)': {
        0: '2022-10-28_05-51-55_482567+0000',
    },
    'Ampicillin (1.5 mg/L)': {
        0: '2022-10-28_05-52-43_927562+0000',
    },
    'Ampicillin (1 mg/L)': {
        0: '2022-10-28_05-53-09_174585+0000',
    },
    'Ampicillin (0.5 mg/L)': {
        0: '2022-10-28_05-53-53_873981+0000',
    },
}

PATHS_TO_LOAD = {
    'Dry mass': ('listeners', 'mass', 'dry_mass'),
    'AcrAB-TolC': ('bulk', 'TRANS-CPLX-201[m]'),
    'Periplasmic tetracycline': ('periplasm', 'concentrations', 'tetracycline'),
    'Cytoplasmic tetracycline': ('cytoplasm', 'concentrations', 'tetracycline'),
    'Periplasmic ampicillin': ('periplasm', 'concentrations', 'ampicillin'),
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


def make_figure_1a(data, metadata):
    # Generational (ompF) vs sub-generational (marR) expression
    columns_to_plot = {
        'ompF mRNA': (255, 0, 0),
        'OmpF monomer': (255, 0, 0),
        'marR mRNA': (0, 0, 255),
        'MarR monomer': (0, 0, 255)}
    _, axes = plt.subplots(2, 2, sharex='col', figsize=(6, 6))
    # Only plot first seed glucose sim data
    mask = (data.loc['Condition']=='Glucose') & (data.loc['Seed']==0)
    data = data.loc[mask, :]
    agent_ids = data.loc[data.loc[:, 'Time']==MAX_TIME, 'Agent ID']
    # Arbitrarily pick a surviving agent to plot trace of
    highlight_agent = agent_ids[0]
    plot_generational_timeseries(
        data=data, axes=axes, columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, colony_scale=False)
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_1a.svg')
    plt.close()
    print('Done with Figure 1A.')

def make_figure_1c(data, metadata):
    # Single-cell vs colony-scale data
    columns_to_plot = {
        'Active ribosomes': (0, 0, 255)
    }
    _, ax = plt.subplots(1, 1, figsize=(4, 4))
    # Only plot first seed glucose sim data
    mask = (data.loc['Condition']=='Glucose') & (data.loc['Seed']==0)
    data = data.loc[mask, :]
    final_timestep = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    # Arbitrarily pick a surviving agent to plot trace of
    agent_ids = final_timestep.loc[:, 'Agent ID']
    highlight_agent = agent_ids[0]
    plot_generational_timeseries(
        data=data, axes=[ax], columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, colony_scale=False)
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_1c_timeseries.svg')
    plt.close()
    
    # Convert DataFrame data back to dictionary form for tag plot
    snapshot_data = {
        final_timestep.loc[:, 'Time']: {
            agent_id: {
                'boundary': boundary,
                'Active ribosomes': active_ribosomes
            }
            for agent_id, boundary, active_ribosomes in zip(
                final_timestep.loc[:, 'Agent ID'],
                final_timestep.loc[:, 'Boundary'],
                final_timestep.loc[:, 'Active ribosomes']
            )
        }
    }
    plot_tags(
        data=snapshot_data,
        bounds=metadata['Glucose'][0]['bounds'],
        snapshot_times=MAX_TIME,
        n_snapshots=1,
        background_color='white',
        tagged_molecules=[('Dry mass',)],
        tag_colors={('Dry mass',): (0, 0, 255)},
        plot_width=6
    )
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_1c_snapshot.svg')
    plt.close()
    print('Done with Figure 1C.')


def make_figure_2(data, metadata):
    # Overview of glucose data for seed 0 (can put other seeds in supp.)
    mask = (data.loc['Condition']=='Glucose') & (data.loc['Seed']==0)
    data = data.loc[mask, :]
    final_timestep = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    agent_ids = final_timestep.loc[:, 'Agent ID']
    highlight_agent = agent_ids[0]
    # 5 equidistant snapshot plots in a row (Fig. 2a)
    plot_snapshot_timeseries(
        data=data, metadata=metadata, highlight_lineage=highlight_agent)
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_2a_snapshots.svg')
    plt.close()

    # Set up subplot layout for timeseries plots
    gs_kw = {"width_ratios": [1, 1, 1, 1, 1], "height_ratios": [1, 1, 1, 1, 1]}
    fig, axes = plt.subplot_mosaic(
        [
            ["B__", "B__", "B__", "B__"],
            ["C1_", "C2_", "C3_", "C4_"],
            ["D1_", "D2_", "D3_", "D4_"],
        ],
        gridspec_kw=gs_kw,
        figsize=(8, 6),
        layout="constrained",
    )
    # Make sure squares are squares
    for i in range(1, 5):
        axes[f"C{i}_"].set_box_aspect(1)
        axes[f"D{i}_"].set_box_aspect(1)

    columns_to_plot = {
        'Dry mass': (0, 0, 255)
    }
    plot_generational_timeseries(
        data=data, axes=axes, columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, colony_scale=False)
    columns_to_plot = {
        'ompF mRNA': (255, 0, 0),
        'marR mRNA': (0, 0, 255),
        'ampC mRNA': (255, 140, 0),
        'tolC mRNA': (128, 0, 128),
        'OmpF monomer': (255, 0, 0),
        'MarR monomer': (0, 0, 255),
        'AmpC monomer': (255, 140, 0),
        'TolC monomer': (128, 0, 128),
    }
    plot_concentration_timeseries(
        data=data, axes=axes[1:], columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, colony_scale=True)
    
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_2b_d_snapshot.svg')
    plt.close()
    print('Done with Figure 2.')


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


def load_data(experiment_id=None, cpus=8, sampling_rate=2,
    host="10.138.0.75", port=27017
):
    # Get data for the specified experiment_id
    monomers = [path[-1] for path in PATHS_TO_LOAD.values() if path[0]=='monomer']
    mrnas = [path[-1] for path in PATHS_TO_LOAD.values() if path[0]=='mrna']
    inner_paths = [path for path in PATHS_TO_LOAD.values() 
        if path[-1] not in mrnas and path[-1] not in monomers]
    outer_paths = [('data', 'dimensions'), ('data', 'fields')]
    for condition, seeds in EXPERIMENT_ID_MAPPING.items():
        for seed, curr_experiment_id in seeds.items():
            if curr_experiment_id != experiment_id:
                continue
            metadata = {condition: {seed: {}}}
            rep_data = access_counts(
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
                    deserialize_and_remove_units, rep_data.values()),
                    total=len(rep_data)))
            rep_data = dict(zip(rep_data.keys(), deserialized_data))
            # Get spatial environment data for snapshot plots
            print('Extracting spatial environment data...')
            metadata[condition][seed]['bounds'] = rep_data[
                min(rep_data)]['dimensions']['bounds']
            metadata[condition][seed]['fields'] = {
                time: data_at_time['fields'] 
                for time, data_at_time in rep_data.items()
            }
            agent_df_paths = partial(agent_data_table,
                paths_dict=PATHS_TO_LOAD, condition=condition, seed=seed)
            with ProcessPoolExecutor(cpus) as executor:
                print('Converting data to DataFrame...')
                rep_dfs = list(tqdm(executor.map(
                    agent_df_paths, rep_data.items()),
                    total=len(rep_data)))
            # Save data for each experiment as local pickle
            pd.concat(rep_dfs).to_pickle(f'data/{experiment_id}.pkl')
            with open(f'data/{experiment_id}_metadata.pkl', 'wb') as f:
                pickle.dump(metadata, f)


def main():
    sns.set_style('white')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--figure_ids', '-f', type=str, nargs='+', default=[],
        help="""List of Figure IDs to make. See keys of ``FIGURE_MAPPING``
        for valid types."""
    )
    args = parser.parse_args()
    os.makedirs('out/analysis/paper_figures/', exist_ok=True)


if __name__ == '__main__':
    main()
