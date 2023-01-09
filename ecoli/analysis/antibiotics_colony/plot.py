
import argparse
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vivarium.library.dict_utils import deep_merge

from ecoli.analysis.antibiotics_colony import (COUNTS_PER_FL_TO_NANOMOLAR,
                                               DE_GENES, EXPERIMENT_ID_MAPPING,
                                               MAX_TIME, SPLIT_TIME,
                                               restrict_data)
from ecoli.analysis.antibiotics_colony.exploration import (
    plot_exp_growth_rate, plot_ampc_phylo)
from ecoli.analysis.antibiotics_colony.timeseries import (plot_field_snapshots,
                                                          plot_tag_snapshots,
                                                          plot_timeseries)
from ecoli.analysis.antibiotics_colony.validation import (
    plot_colony_growth, plot_mrna_fc, plot_protein_synth_inhib,
    plot_synth_prob_fc)


def make_figure_1a(data, metadata):
    # Retrieve only glucose data from seed 10000
    data = data.loc[(data.loc[:, 'Condition']=='Glucose') &
        (data.loc[:, 'Seed']==10000), :]

    # Generational (ompF) vs sub-generational (marR) expression (Fig 1a)
    columns_to_plot = {
        'ompF mRNA': '0.4',
        'marR mRNA': (0, 0.4, 1),
        'OmpF monomer': '0.4',
        'MarR monomer': (0, 0.4, 1)}
    fig, axes = plt.subplots(2, 2, sharex='col', figsize=(6, 6))
    axes = np.ravel(axes)
    # Arbitrarily pick a surviving agent to plot trace of
    highlight_agent = '011001001'
    print(f'Highlighted agent: {highlight_agent}')
    plot_timeseries(
        data=data, axes=axes, columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent, background_lineages=False)
    axes[0].set_xlabel(None)
    axes[1].set_xlabel(None)
    # Put gene name on top and remove superfluous axes labels
    gene_1 = axes[0].get_ylabel().split(' ')[0]
    gene_2 = axes[1].get_ylabel().split(' ')[0]
    axes[0].set_ylabel('mRNA\n(counts)')
    axes[2].set_ylabel('Monomer\n(counts)')
    axes[0].set_title(f'Exponential: {gene_1}', fontsize=10)
    axes[1].set_title(f'Sub-generational: {gene_2}', fontsize=10)
    axes[0].yaxis.set_label_coords(-0.3, 0.5)
    axes[2].yaxis.set_label_coords(-0.3, 0.5)
    axes[1].yaxis.label.set_visible(False)
    axes[3].yaxis.label.set_visible(False)
    axes[0].xaxis.set_visible(False)
    axes[0].spines.bottom.set_visible(False)
    axes[1].xaxis.set_visible(False)
    axes[1].spines.bottom.set_visible(False)
    for ax in axes:
        [item.set_fontsize(8) for item in ax.get_xticklabels()]
        [item.set_fontsize(8) for item in ax.get_yticklabels()]
        ax.xaxis.label.set_fontsize(9)
        ax.yaxis.label.set_fontsize(9)
        ax.xaxis.set_label_coords(0.5, -0.2)
    ax.tick_params(axis='both', which='major')
    fig.set_size_inches(4, 3)
    plt.draw()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    os.makedirs('out/analysis/paper_figures/1a/', exist_ok=True)
    plt.savefig(f'out/analysis/paper_figures/1a/fig_1a_{highlight_agent}.svg',
        bbox_inches='tight')
    plt.close()
    print('Done with Figure 1A.')


def make_figure_2b(data, metadata):
    # Overview of glucose data for seed 0 (can put other seeds in supp.)
    final_timestep = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    agent_ids = final_timestep.loc[:, 'Agent ID']
    highlight_agent = agent_ids[100]
    print(f'Highlighted agent: {highlight_agent}')
    # 5 equidistant snapshot plots in a row (Fig. 2b)
    plot_field_snapshots(
        data=data, metadata=metadata, highlight_lineage=highlight_agent,
        highlight_color=(0, 0.4, 1), min_pct=0.8, colorbar_decimals=2)
    print('Done with Figure 2B.')


def make_figure_2c(data, metadata):
    # Use same highlighted agent as in Figure 2B
    final_timestep = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    agent_ids = final_timestep.loc[:, 'Agent ID']
    highlight_agent = agent_ids[100]
    print(f'Highlighted agent: {highlight_agent}')

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
    plot_timeseries(
        data=data, axes=axes, columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent)
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
    # Convert to concentrations using periplasmic or cytoplasmic volume
    periplasmic = ['OmpF monomer', 'AmpC monomer', 'TolC monomer']
    for column in columns_to_plot:
        if column in periplasmic:
            data.loc[:, column] /= (data.loc[:, 'Volume'] * 0.2)
        else:
            data.loc[:, column] /= (data.loc[:, 'Volume'] * 0.8)
        data.loc[:, column] *= COUNTS_PER_FL_TO_NANOMOLAR
    plot_timeseries(
        data=data, axes=axes[1:], columns_to_plot=columns_to_plot,
        highlight_lineage=highlight_agent)
    # Add more regularly spaced tick marks to top row
    time_ticks = axes[0].get_xticks()
    new_ticks = np.arange(1, np.ceil(time_ticks[1]), 1).astype(int)
    # No need for tick at 7 since final tick is 7.2
    new_ticks = new_ticks[new_ticks != 7].tolist()
    time_ticks = [0] + new_ticks + [time_ticks[1]]
    axes[0].set_xticks(ticks=time_ticks, labels=time_ticks)
    # Put gene name on top and remove superfluous axes labels
    gene = axes[1].get_ylabel().split(' ')[0]
    axes[0].set_ylabel('Dry mass (fg)')
    axes[5].set_title(gene, fontsize=12, fontweight='bold')
    axes[5].set_ylabel('mRNA (nM)')
    axes[1].set_ylabel('Protein (nM)')
    for i in range(2, 5):
        gene = axes[i].get_ylabel().split(' ')[0]
        axes[i].yaxis.label.set_visible(False)
        axes[4+i].set_title(gene, fontsize=12, fontweight='bold')
        axes[4+i].yaxis.label.set_visible(False)
    for ax in axes[5:]:
        ax.xaxis.set_visible(False)
        ax.spines.bottom.set_visible(False)
    for ax in axes:
        [item.set_fontsize(8) for item in ax.get_xticklabels()]
        [item.set_fontsize(8) for item in ax.get_yticklabels()]
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
        ax.tick_params(axis='both', which='major')
    fig.set_size_inches(7, 4)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.45)
    for ax in axes[1:]:
        ax.xaxis.set_label_coords(0.5, -0.3)
        left, bottom, width, height = ax.get_position().bounds
        ax.set_position((left, bottom-0.15, width, height))
    left, bottom, width, height = axes[0].get_position().bounds
    axes[0].set_position((left, bottom+0.03, width, height))
    axes[0].xaxis.set_label_coords(0.5, -0.3)
    axes[0].yaxis.set_label_coords(-0.09, 0.5)
    axes[5].yaxis.set_label_coords(-0.5, 0.5)

    # Prettify axes (moving axis titles in to save space)
    for ax in axes[1:5]:
        xmin, xmax = ax.get_xlim()
        ax.set_xticks([(xmin + xmax) / 2], labels=[ax.get_xlabel()], minor=True)
        ax.set_xlabel(None)
        ax.tick_params(
            which='minor',
            width=0,
            length=ax.xaxis.get_major_ticks()[0].get_tick_padding(),
            labelsize=10
        )
    plt.savefig('out/analysis/paper_figures/fig_2c_timeseries.svg',
        bbox_inches='tight')
    plt.close()
    print('Done with Figure 2C.')


def make_figure_3a(data, metadata):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    plot_colony_growth(data, ax)
    plt.tight_layout()
    fig.savefig('out/analysis/paper_figures/tet_colony_growth.svg')
    plt.close()
    print('Done with Figure 3A.')


def make_figure_3b(data, metadata):
    data = data.loc[data.loc[:, 'Time']<=MAX_TIME, :]
    data = data.sort_values(['Condition', 'Agent ID', 'Time'])
    plot_exp_growth_rate(data, metadata)
    plt.close()
    print('Done with Figure 3B.')


def make_figure_3c(data, metadata):
    # Short-term changes to tet. exposure
    # Filter data to only include 150 seconds before and after
    glucose_mask = ((data.loc[:, 'Time'] >= 11400) &
        (data.loc[:, 'Time'] <= SPLIT_TIME) &
        (data.loc[:, 'Condition'] == 'Glucose'))
    tet_mask = ((data.loc[:, 'Time'] >= SPLIT_TIME) &
        (data.loc[:, 'Time'] <= 11700) &
        (data.loc[:, 'Condition'] == 'Tetracycline (1.5 mg/L)'))
    transition_data = data.loc[glucose_mask | tet_mask, :]
    # Convert tetracycline concentrations to uM
    transition_data.loc[:, 'Periplasmic tetracycline'] *= 1000
    transition_data.loc[:, 'Cytoplasmic tetracycline'] *= 1000
    fig, axes = plt.subplots(1, 3, figsize=(7, 2))
    short_term_columns = {
        'Periplasmic tetracycline': 0,
        'Cytoplasmic tetracycline': 1,
        'Active ribosomes': 2,
    }
    # Convert to concentration using cytoplasmic volume
    transition_data.loc[:, 'Active ribosomes'] /= (
        transition_data.loc[:, 'Volume'] * 0.8)
    transition_data.loc[:, 'Active ribosomes'] *= COUNTS_PER_FL_TO_NANOMOLAR
    for column, ax_idx in short_term_columns.items():
        plot_timeseries(
            data=transition_data,
            axes=[axes.flat[ax_idx]],
            columns_to_plot={column: (0, 0.4, 1)},
            highlight_lineage='mean',
            filter_time=False,
            background_alpha=0.5,
            background_linewidth=0.3)
    for ax in axes.flat:
        ylim = ax.get_ylim()
        ax.set_yticks(np.round(ylim, 0).astype(int))
        ax.set_xlabel(None)
        # Mark minutes since tetracycline addition
        ax.set_xticks(ticks=[11430/3600, 11490/3600, 11550/3600,
            11610/3600, 11670/3600], labels=[-2, -1, 0, 1, 2])
        ax.spines.bottom.set(bounds=(11400/3600, 11700/3600), linewidth=1,
            visible=True, color=(0, 0, 0), alpha=1)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(None)
        ax.set_title(ylabel)
    axes.flat[0].set_ylabel('Conc. (uM)')
    fig.supxlabel('Minutes Since Tetracycline Addition')
    plt.tight_layout()
    fig.savefig('out/analysis/paper_figures/tet_short_term.svg')
    plt.close()
    print('Done with Figure 3C.')


def make_figure_3d(data, metadata):
    # Long-term changes to tet. exposure
    # Filter data to include glucose for first 11550 seconds and
    # tetracycline data for remainder of simulation
    long_transition_data = restrict_data(data)
    long_term_columns = {
        'micF-ompF duplex': 0,
        'ompF mRNA': 1,
        'OmpF monomer': 2,
        'acrA mRNA': 3,
        'AcrA monomer': 4,
        'AcrAB-TolC': 5,
    }
    # Convert to concentrations using periplasmic or cytoplasmic volume
    periplasmic = ['OmpF monomer', 'AcrAB-TolC', 'AcrA monomer']
    for column in long_term_columns:
        if column in periplasmic:
            long_transition_data.loc[:, column] /= (
                long_transition_data.loc[:, 'Volume'] * 0.2)
        else:
            long_transition_data.loc[:, column] /= (
                long_transition_data.loc[:, 'Volume'] * 0.8)
        long_transition_data.loc[:, column] *= COUNTS_PER_FL_TO_NANOMOLAR
    fig, axes = plt.subplots(2, 3, figsize=(7, 4))
    for column, ax_idx in long_term_columns.items():
        plot_timeseries(
            data=long_transition_data,
            axes=[axes.flat[ax_idx]],
            columns_to_plot={column: (0, 0.4, 1)},
            highlight_lineage='mean',
            filter_time=False,
            background_alpha=0.5,
            background_linewidth=0.3)
    for ax in axes.flat:
        ylim = ax.get_ylim()
        ax.set_yticks(np.round(ylim, 0).astype(int))
        # Mark hours since tetracycline addition
        xlim = np.array(ax.get_xlim())
        xticks = np.append(xlim, 11550/3600)
        xtick_labels = np.trunc(xticks-11550/3600).astype(int)
        ax.set_xticks(ticks=xticks, labels=xtick_labels)
        ax.set_xlabel(None)
        ax.spines.bottom.set(bounds=(0, MAX_TIME/3600), linewidth=1,
            visible=True, color=(0, 0, 0), alpha=1)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(None)
        ax.set_title(ylabel)
    fig.supylabel('Concentration (uM)')
    fig.supxlabel('Hours Since Tetracycline Addition')
    plt.tight_layout()
    fig.savefig('out/analysis/paper_figures/tet_long_term.svg')
    plt.close()
    print('Done with Figure 3D.')


def make_figure_3e(data, metadata):
    genes_to_plot = DE_GENES.loc[:, 'Gene name']
    fig, axs = plt.subplots(1, 2, figsize=(7, 3))
    plot_synth_prob_fc(data, axs[0], genes_to_plot, 0)
    plot_mrna_fc(data, axs[1], genes_to_plot, 0)
    axs[0].set_yticks([0, 0.5, 1, 1.5])
    axs[1].spines['left'].set_bounds((-1, 1.5))
    axs[1].set_yticks([-1, -0.5, 0, 0.5, 1, 1.5])
    fig.savefig('out/analysis/paper_figures/tet_synth_prob_unfiltered.svg', bbox_inches='tight')
    plt.close()
    print('Done with Figure 3E.')


def make_figure_3f(data, metadata):
    jenner = pd.read_csv('data/sim_dfs/jenner_2013.csv', header=None).rename(
        columns={0: 'Tetracycline', 1: 'Percent inhibition'})
    jenner['Source'] = ['Jenner et al. 2013'] * len(jenner)
    jenner.loc[:, 'Percent inhibition'] = 1 - (jenner.loc[:, 'Percent inhibition'] / 100)
    olson = pd.read_csv('data/sim_dfs/olson_2006.csv', header=None).rename(
        columns={0: 'Tetracycline', 1: 'Percent inhibition'})
    olson.loc[:, 'Percent inhibition'] /= 100
    olson['Source'] = ['Olson et al. 2006'] * len(olson) 
    literature = pd.concat([jenner, olson])
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    plot_protein_synth_inhib(data, ax, literature)
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/protein_synth_inhib.svg')
    plt.close()
    print('Done with Figure 3F.')


def make_figure_4a(data, metadata):
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    plot_colony_growth(data, ax, antibiotic_col='Initial external amp.',
        mic=5.724, antibiotic='Ampicillin')
    plt.tight_layout()
    fig.savefig('out/analysis/paper_figures/amp_colony_growth.svg')
    plt.close()
    print('Done with Figure 4A.')


def make_figure_4b(data, metadata):
    # Only include seed 0
    data = data.loc[data.loc[:, 'Seed']==0, :]

    # Get fold change over average glucose porosity
    data['Relative porosity'] = data.loc[:, 'Porosity'] * data.loc[:, 'Extension factor']
    mean_glc_porosity = data.loc[data.loc[:, 'Condition'] == 'Glucose', 
        'Relative porosity'].mean()
    fc_col = 'Porosity\n($\mathregular{log_2}$ fold change)'
    data.loc[:, fc_col] = np.log2(data.loc[:, 'Relative porosity'] / mean_glc_porosity)
    data.loc[data.loc[:, fc_col]==-np.inf, fc_col] = 0

    # Set up custom divergent colormap
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        'divergent', [(0, 0.4, 1), (1, 1, 1), (0.678, 0, 0.125)])
    magnitude = data.loc[:, fc_col].abs().max()
    norm = matplotlib.colors.Normalize(vmin=-magnitude, vmax=magnitude)
    plot_tag_snapshots(
        data=data, metadata=metadata, tag_colors={fc_col: {'cmp': cmp, 'norm': norm}},
        snapshot_times=np.array([1.9, 3.2, 4.5, 5.8, 7.1]) * 3600)
    print('Done with Figure 4B.')


def make_figure_4c(data, metadata):
    plot_ampc_phylo(data)
    print('Done with Figure 4C.')


def load_pickles(experiment_ids):
    data = []
    metadata = {}
    for exp_id in experiment_ids:
        with open(f'data/sim_dfs/{exp_id}.pkl', 'rb') as f:
            exp_data = pickle.load(f)
            if exp_data.loc[:, 'Dry mass'].iloc[-1]==0:
                exp_data = exp_data.iloc[:-1, :]
            data.append(exp_data)
        with open(f'data/sim_dfs/{exp_id}_metadata.pkl', 'rb') as f:
            metadata = deep_merge(metadata, pickle.load(f))
    data = pd.concat(data)
    initial_external_tet = []
    initial_external_amp = []
    for condition in data['Condition'].unique():
        cond_data = data.loc[data.loc[:, 'Condition'] == condition, :]
        if condition == 'Glucose':
            initial_external_tet += [0] * len(cond_data)
            initial_external_amp += [0] * len(cond_data)
            continue
        curr_len = len(initial_external_tet)
        for boundary_data in cond_data.loc[:, 'Boundary']:
            # Assumes only one antibiotic is used at a time
            tet_conc = boundary_data['external']['tetracycline']
            if tet_conc != 0:
                initial_external_tet += [tet_conc] * len(cond_data)
                initial_external_amp += [0] * len(cond_data)
                break
            amp_conc = boundary_data['external']['ampicillin[p]']
            if amp_conc != 0:
                initial_external_amp += [amp_conc] * len(cond_data)
                initial_external_tet += [0] * len(cond_data)
                break
        if len(initial_external_tet)==curr_len:
            initial_external_tet += [0] * len(cond_data)
            initial_external_amp += [0] * len(cond_data)
    data['Initial external tet.'] = initial_external_tet
    data['Initial external amp.'] = initial_external_amp
    return data, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fig_ids',
        '-f',
        help='List of lowercase figure IDs to create (e.g. 1a 2b 3c). \
            Default is all.',
        nargs='+',
    )
    args = parser.parse_args()

    tet_local = [
        '2022-12-08_01-13-41_036971+0000',
        '2022-12-08_01-37-02_043920+0000',
        '2022-12-08_01-37-17_383563+0000',
        '2022-12-08_01-37-25_382616+0000',
        '2022-12-08_01-37-31_999399+0000',
        '2022-12-08_01-37-38_566402+0000',
        '2022-12-08_01-37-44_216110+0000',
        '2022-12-08_01-37-52_725211+0000',
        '2022-12-08_01-37-57_809101+0000',
        '2022-12-08_01-38-03_635076+0000',
        '2022-12-08_01-38-09_020029+0000'
    ]
    conditions = {
        '1a': ['Glucose'],
        '2b': ['Glucose'],
        '2c': ['Glucose'],
        '3a': ['Glucose', 'Tetracycline (0.5 mg/L)', 'Tetracycline (1 mg/L)',
            'Tetracycline (1.5 mg/L)', 'Tetracycline (2 mg/L)',
            'Tetracycline (4 mg/L)'],
        '3b': ['Glucose', 'Tetracycline (0.5 mg/L)', 'Tetracycline (1 mg/L)',
            'Tetracycline (1.5 mg/L)', 'Tetracycline (2 mg/L)',
            'Tetracycline (4 mg/L)'],
        '3c': ['Glucose', 'Tetracycline (1.5 mg/L)'],
        '3d': ['Glucose', 'Tetracycline (1.5 mg/L)'],
        '3e': ['Glucose', 'Tetracycline (1.5 mg/L)'],
        '3f': [str(i) for i in range(11)],
        '4a': ['Glucose', 'Ampicillin (0.5 mg/L)', 'Ampicillin (1 mg/L)',
            'Ampicillin (1.5 mg/L)', 'Ampicillin (2 mg/L)',
            'Ampicillin (4 mg/L)'],
        '4b': ['Glucose', 'Ampicillin (2 mg/L)'],
        '4c': ['Glucose', 'Ampicillin (2 mg/L)'],
    }
    seeds = {
        '1a': [10000],
        '2b': [10000],
        '2c': [10000],
        '3a': [0],
        '3b': [0],
        '3c': [0],
        '3d': [0],
        '3e': [0, 10000],
        '3f': [0],
        '4a': [0],
        '4b': [0],
        '4c': [0],
    }
    if args.fig_ids is None:
        args.fig_ids = conditions.keys() - {'3f'}

    ids_to_load = []
    for fig_id in args.fig_ids:
        if fig_id == '3f':
            ids_to_load.extend(tet_local)
            continue
        for condition in conditions[fig_id]:
            for seed in seeds[fig_id]:
                ids_to_load.append(EXPERIMENT_ID_MAPPING[condition][seed])
    # De-duplicate IDs while preserving order
    ids_to_load = list(dict.fromkeys(ids_to_load))
    
    data, metadata = load_pickles(ids_to_load)

    for fig_id in args.fig_ids:
        filter = (np.isin(data.loc[:, 'Condition'], conditions[fig_id]) &
            np.isin(data.loc[:, 'Seed'], seeds[fig_id]))
        fig_data = data.loc[filter, :].copy()
        globals()[f'make_figure_{fig_id}'](fig_data, metadata)


if __name__ == '__main__':
    main()
