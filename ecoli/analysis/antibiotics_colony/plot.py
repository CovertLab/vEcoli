from concurrent.futures import ProcessPoolExecutor
from functools import partial
import matplotlib
# matplotlib.use('svg')
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'Arial'
from matplotlib.colors import rgb_to_hsv
import numpy as np
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
from ecoli.analysis.antibiotics_colony import (
    DE_GENES, MAX_TIME, COUNTS_PER_FL_TO_NANOMOLAR)

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
        'ompF mRNA': '0.8',
        'marR mRNA': '0.4',
        'OmpF monomer': '0.8',
        'MarR monomer': '0.4'}
    fig, axes = plt.subplots(2, 2, sharex='col', figsize=(6, 6))
    axes = np.ravel(axes)
    # Only plot first seed glucose sim data
    mask = (data.loc[:, 'Condition']=='Glucose') & (data.loc[:, 'Seed']==0)
    data = data.loc[mask, :]
    agent_ids = data.loc[data.loc[:, 'Time']==MAX_TIME, 'Agent ID']
    # Arbitrarily pick a surviving agent to plot trace of
    highlight_agent = agent_ids[1]
    plot_generational_timeseries(
        data=data, axes=axes, columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, colony_scale=False,
        highlight_color=None)
    # Put gene name on top and remove superfluous axes labels
    gene_1 = axes[0].get_ylabel().split(' ')[0]
    gene_2 = axes[1].get_ylabel().split(' ')[0]
    axes[0].set_ylabel('mRNA (counts)')
    axes[2].set_ylabel('Monomer (counts)')
    axes[0].set_title(f'Exponential: {gene_1}')
    axes[1].set_title(f'Sub-generational: {gene_2}')
    axes[0].yaxis.set_label_coords(-0.3, 0.5)
    axes[2].yaxis.set_label_coords(-0.3, 0.5)
    axes[1].yaxis.label.set_visible(False)
    axes[3].yaxis.label.set_visible(False)
    for ax in axes:
        [item.set_fontsize(8) for item in ax.get_xticklabels()]
        [item.set_fontsize(8) for item in ax.get_yticklabels()]
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.tick_params(axis='both', which='major')
    plt.draw()
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_1a.svg')
    plt.close()
    print('Done with Figure 1A.')

def make_figure_1c(data, metadata):
    # Single-cell vs colony-scale data
    columns_to_plot = {
        'Active ribosomes': (0, 0.4, 1)
    }
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    # Only plot first seed glucose sim data
    mask = (data.loc[:, 'Condition']=='Glucose') & (data.loc[:, 'Seed']==0)
    data = data.loc[mask, :]
    final_timestep = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    # Arbitrarily pick a surviving agent to plot trace of
    agent_ids = final_timestep.loc[:, 'Agent ID']
    highlight_agent = agent_ids[1]
    plot_generational_timeseries(
        data=data, axes=[ax], columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, colony_scale=False,
        highlight_color=None)
    ax.set_title(f'Active ribosomes (counts)')
    ax.yaxis.label.set_visible(False)
    [item.set_fontsize(8) for item in ax.get_xticklabels()]
    [item.set_fontsize(8) for item in ax.get_yticklabels()]
    ax.xaxis.label.set_fontsize(12)
    ax.tick_params(axis='both', which='major')
    plt.draw()
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_1c_timeseries.svg')
    plt.close()
    
    # Convert DataFrame data back to dictionary form for tag plot
    snapshot_data = {
        MAX_TIME: {
            'agents': {
                agent_id: {
                    'boundary': boundary,
                    # Convert from counts to nM
                    'Active ribosomes': (
                        active_ribosomes / boundary['volume'] *
                        COUNTS_PER_FL_TO_NANOMOLAR)
                }
                for agent_id, boundary, active_ribosomes in zip(
                    final_timestep.loc[:, 'Agent ID'],
                    final_timestep.loc[:, 'Boundary'],
                    final_timestep.loc[:, 'Active ribosomes']
                )
            },
            'fields': metadata['Glucose'][0]['fields'][MAX_TIME]
        }
    }
    fig = plot_tags(
        data=snapshot_data,
        bounds=metadata['Glucose'][0]['bounds'],
        snapshot_times=[MAX_TIME],
        n_snapshots=1,
        background_color='white',
        tagged_molecules=[('Active ribosomes',)],
        tag_colors={
            ('Active ribosomes',): rgb_to_hsv((0, 0.4, 1))
        },
        default_font_size=30,
        tag_label_size=30,
        membrane_color=(0, 0, 0),
        min_color='white',
        scale_bar_length=0,
        colorbar_decimals=0,
        convert_to_concs=False
    )
    # Snapshot figure axes
    # 0: time axis
    # 1: colony snapshot
    # 2: subplot for colorbar
    # 3: colorbar
    fig.set_size_inches(12, 10)
    fig.axes[2].set_position([0.8, 0.1, 0.8, 0.8])
    ylimits = np.round(fig.axes[3].get_ylim())
    fig.axes[3].set_yticks([])
    fig.axes[3].set_title(int(ylimits[1]), fontsize=20)
    fig.axes[3].text(0.5, -0.01, int(ylimits[0]),
        horizontalalignment='center', verticalalignment='top',
        transform=fig.axes[3].transAxes, fontsize=20)
    fig.axes[1].set_position([0.1, 0.05, 0.65, 0.9])
    fig.axes[0].set_position([0.1, 0.1, 0.65, 0.9])
    fig.axes[0].tick_params(labelsize=20)
    fig.axes[0].set_xlabel('Time (s)', labelpad=0, fontsize=24)
    fig.axes[1].set_title('Active ribosomes (nM)', y=1.05, fontsize=24)
    fig.axes[1].set_ylabel(None)
    fig.savefig('out/analysis/paper_figures/fig_1c_snapshot.svg')
    plt.close()
    print('Done with Figure 1C.')


def make_figure_2(data, metadata):
    # Overview of glucose data for seed 0 (can put other seeds in supp.)
    final_timestep = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    agent_ids = final_timestep.loc[:, 'Agent ID']
    highlight_agent = agent_ids[1]
    # 5 equidistant snapshot plots in a row (Fig. 2a)
    plot_snapshot_timeseries(
        data=data, metadata=metadata, highlight_lineage=highlight_agent,
        highlight_color=(0, 0.4, 1))

    # Set up subplot layout for timeseries plots
    fig = plt.figure()
    gs = fig.add_gridspec(3, 4)
    axes = [fig.add_subplot(gs[0, :])]
    for i in range(4):
        axes.append(fig.add_subplot(gs[2, i]))
    for i in range(4):
        axes.append(fig.add_subplot(gs[1, i], sharex=axes[i+1]))

    columns_to_plot = {
        'Dry mass': (0, 0.4, 1),
    }
    plot_generational_timeseries(
        data=data, axes=axes, columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, highlight_color=None,
        colony_scale=False, align=False)
    columns_to_plot = {
        'OmpF monomer': (0, 0.4, 1),
        'MarR monomer': (0, 0.4, 1),
        'AmpC monomer': (0, 0.4, 1),
        'TolC monomer': (0, 0.4, 1),
        'ompF mRNA': (0, 0.4, 1),
        'marR mRNA': (0, 0.4, 1),
        'ampC mRNA': (0, 0.4, 1),
        'tolC mRNA': (0, 0.4, 1),
    }
    plot_concentration_timeseries(
        data=data, axes=axes[1:], columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, highlight_color=(0, 0, 1),
        colony_scale=True)
    # Add more regularly spaced tick marks to top row
    time_ticks = axes[0].get_xticks()
    time_ticks = time_ticks.tolist() + np.arange(
        time_ticks[0], time_ticks[1], 50).tolist()
    time_ticks = [int(time) for time in time_ticks]
    axes[0].set_xticks(time_ticks)
    # Put gene name on top and remove superfluous axes labels
    gene = axes[1].get_ylabel().split(' ')[0]
    axes[5].set_title(gene, fontsize=16)
    axes[5].set_ylabel('mRNA (nM)')
    axes[1].set_ylabel('Monomer (nM)')
    axes[1].yaxis.set_label_coords(-0.2, 0.5)
    for i in range(2, 5):
        gene = axes[i].get_ylabel().split(' ')[0]
        axes[i].yaxis.label.set_visible(False)
        axes[4+i].set_title(gene)
        axes[4+i].yaxis.label.set_visible(False)
    for ax in axes[5:]:
        ax.xaxis.label.set_visible(False)
    for ax in axes:
        [item.set_fontsize(12) for item in ax.get_xticklabels()]
        [item.set_fontsize(12) for item in ax.get_yticklabels()]
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis='both', which='major')
    fig.set_size_inches(16, 8)
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_2b_d_snapshot.svg')
    plt.close()
    print('Done with Figure 2.')


def make_figure_3(data, metadata):
    # Overview of tetracycline data for seed 0 (can put other seeds in supp.)
    final_timestep = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    agent_ids = final_timestep.loc[:, 'Agent ID']
    highlight_agent = agent_ids[1]
    # 5 equidistant snapshot plots in a row (Fig. 2a)
    plot_snapshot_timeseries(
        data=data, metadata=metadata, highlight_lineage=highlight_agent,
        highlight_color=(0, 0, 1))

    # Set up subplot layout for timeseries plots
    fig = plt.figure()
    gs = fig.add_gridspec(4, 4)
    axes = [fig.add_subplot(gs[1, :])]
    axes.append(fig.add_subplot(gs[0, :], sharex=axes[0]))
    for i in range(4):
        axes.append(fig.add_subplot(gs[3, i]))
    for i in range(4):
        axes.append(fig.add_subplot(gs[2, i], sharex=axes[i+2]))

    # Convert to micromolar
    data.loc[:, 'Cytoplasmic tetracycline'] *= 1000
    data = data.rename(columns={'Cytoplasmic tetracycline': 'Tetracycline (uM)'})
    columns_to_plot = {
        'Tetracycline (uM)': (0, 0, 1),
        'Dry mass': (0, 0, 1),
    }
    plot_generational_timeseries(
        data=data, axes=axes, columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, highlight_color=None,
        colony_scale=False, align=False)
    columns_to_plot = {
        'OmpF monomer': (0, 0, 1),
        'MarA monomer': (0, 0, 1),
        'Active ribosomes': (0, 0, 1),
        'AcrAB-TolC': (0, 0, 1),
        'ompF mRNA': (0, 0, 1),
        'marA mRNA': (0, 0, 1),
        'Inactive 30S subunit': (0, 0, 1),
        'micF-ompF duplex': (0, 0, 1),        
    }
    plot_concentration_timeseries(
        data=data, axes=axes[2:], columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, highlight_color=(0, 0, 1),
        colony_scale=True)
    
    # Add more regularly spaced tick marks to top row
    time_ticks = axes[0].get_xticks()
    rounded_time_ticks = np.ceil(time_ticks/50) * 50
    time_ticks = time_ticks.tolist() + np.arange(
        rounded_time_ticks[0], rounded_time_ticks[1], 50).tolist()
    time_ticks = [int(time) for time in time_ticks]
    axes[0].set_xticks(time_ticks)
    axes[1].set_xticks(time_ticks)
    for ax in axes:
        [item.set_fontsize(12) for item in ax.get_xticklabels()]
        [item.set_fontsize(12) for item in ax.get_yticklabels()]
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis='both', which='major')
    for ax in axes[-4:]:
        ax.xaxis.label.set_visible(False)
    for ax in axes[:2]:
        ax.xaxis.label.set_visible(False)
    fig.set_size_inches(16, 10)
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_3b_d_snapshot.svg')
    plt.close()
    print('Done with Figure 3.')


def make_figure_4(data, metadata):
    # Overview of ampicillin data for seed 0 (can put other seeds in supp.)
    final_timestep = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    agent_ids = final_timestep.loc[:, 'Agent ID']
    highlight_agent = agent_ids[1]
    # 5 equidistant snapshot plots in a row (Fig. 2a)
    plot_snapshot_timeseries(
        data=data, metadata=metadata, highlight_lineage=highlight_agent,
        highlight_color=(0, 0, 1))

    # Set up subplot layout for timeseries plots
    fig = plt.figure()
    gs = fig.add_gridspec(4, 4)
    axes = [fig.add_subplot(gs[1, :])]
    axes.append(fig.add_subplot(gs[0, :], sharex=axes[0]))
    for i in range(4):
        axes.append(fig.add_subplot(gs[3, i]))
    for i in range(4):
        axes.append(fig.add_subplot(gs[2, i], sharex=axes[i+2]))

    # Convert to micromolar
    data.loc[:, 'Periplasmic ampicillin'] *= 1000
    data = data.rename(columns={
        'Periplasmic ampicillin': 'Ampicillin (uM)',
        'PBP1b gamma complex': 'PBP1b complex'
    })
    columns_to_plot = {
        'Ampicillin (uM)': (0, 0, 1),
        'Dry mass': (0, 0, 1),
    }
    plot_generational_timeseries(
        data=data, axes=axes, columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, highlight_color=None,
        colony_scale=False, align=False)
    columns_to_plot = {
        'PBP1a complex': (0, 0, 1),
        'PBP1b complex': (0, 0, 1),
        'AmpC monomer': (0, 0, 1),
        'Wall cracked': (0, 0, 1),
        'PBP1a mRNA': (0, 0, 1),
        'PBP1b mRNA': (0, 0, 1),
        'ampC mRNA': (0, 0, 1),
        'Murein tetramer': (0, 0, 1),        
    }
    plot_concentration_timeseries(
        data=data, axes=axes[2:], columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, highlight_color=(0, 0, 1),
        colony_scale=True)
    
    # Add more regularly spaced tick marks to top row
    time_ticks = axes[0].get_xticks()
    rounded_time_ticks = np.ceil(time_ticks/50) * 50
    time_ticks = time_ticks.tolist() + np.arange(
        rounded_time_ticks[0], rounded_time_ticks[1], 50).tolist()
    time_ticks = [int(time) for time in time_ticks]
    axes[0].set_xticks(time_ticks)
    axes[1].set_xticks(time_ticks)
    for ax in axes:
        [item.set_fontsize(12) for item in ax.get_xticklabels()]
        [item.set_fontsize(12) for item in ax.get_yticklabels()]
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(axis='both', which='major')
    for ax in axes[-4:]:
        ax.xaxis.label.set_visible(False)
    for ax in axes[:2]:
        ax.xaxis.label.set_visible(False)
    fig.set_size_inches(16, 10)
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_4b_d_snapshot.svg')
    plt.close()
    print('Done with Figure 4.')


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
    os.makedirs('out/analysis/paper_figures/', exist_ok=True)
    # with open('data/sim_data_dfs/2022-10-25_04-55-23_505282+0000.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # with open('data/sim_data_dfs/2022-10-25_04-55-23_505282+0000_metadata.pkl', 'rb') as f:
    #     metadata = pickle.load(f)
    # make_figure_1a(data, metadata)
    # make_figure_1c(data, metadata)
    # make_figure_2(data, metadata)
    # with open('data/sim_data_dfs/2022-10-30_08-46-46_378082+0000.pkl', 'rb') as f:
    #     tet_data = pickle.load(f)
    # with open('data/sim_data_dfs/2022-10-30_08-46-46_378082+0000_metadata.pkl', 'rb') as f:
    #     tet_metadata = pickle.load(f)
    # make_figure_3(tet_data, tet_metadata)
    with open('data/sim_data_dfs/2022-10-28_05-47-52_977686+0000.pkl', 'rb') as f:
        amp_data = pickle.load(f)
    with open('data/sim_data_dfs/2022-10-28_05-47-52_977686+0000_metadata.pkl', 'rb') as f:
        amp_metadata = pickle.load(f)
    make_figure_4(amp_data, amp_metadata)


if __name__ == '__main__':
    main()
