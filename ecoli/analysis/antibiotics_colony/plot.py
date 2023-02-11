
import argparse
import os
import pickle
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import anchored_artists
import numpy as np
import pandas as pd
from scipy.constants import N_A
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
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
    plot_synth_prob_fc, plot_death_timescale_analysis)
from ecoli.plots.snapshots import plot_snapshots


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


def make_figure_1b(data, metadata):
    doubling_times = []
    grouped_data = data.groupby(['Seed'])
    for condition, condition_data in grouped_data:
        final_agents = condition_data.loc[condition_data.loc[:, 'Time']==26000, 'Agent ID'].tolist()
        grouped_agents = condition_data.groupby(['Agent ID'])
        for agent, agent_data in grouped_agents:
            # Total of 4 cells die across all 3 seeds, exclude them
            if (agent not in final_agents) and (agent_data.loc[:, 'Wall cracked'].sum()==0):
                doubling_time = (agent_data.loc[:, 'Time'].max()
                    - agent_data.loc[:, 'Time'].min())
                if doubling_time < 40*60:
                    print(condition, agent)
                doubling_times.append(doubling_time)
    fig, ax = plt.subplots(figsize=(2,2))
    ax.vlines(44, 0, 350, colors=['tab:orange'])
    ax.text(45, 350, 'experiment\n\u03C4 = 44 min.', verticalalignment='top',
        horizontalalignment='left', c='tab:orange',fontsize=8)
    doubling_times = np.array(doubling_times) / 60
    ax.hist(doubling_times)
    ax.set_xlabel('Doubling time (min)')
    ax.set_ylabel('# of simulated cells')
    ax.set_ylim()
    sim_avg = doubling_times.mean()
    ax.vlines(sim_avg, ax.get_ylim()[0], 275, linestyles=['dashed'], colors=['k'])
    ax.text(sim_avg+2, 275, f'simulation\n\u03C4 = {np.round(sim_avg, 1)} min.',
        verticalalignment='top', horizontalalignment='left', c='tab:blue', fontsize=8)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    plt.tight_layout()
    plt.savefig(f'out/analysis/paper_figures/fig_1b_doubling_time.svg',
        bbox_inches='tight')
    plt.close()


def make_figure_1c(data, metadata):
    submass_fc = {
        'Protein': [],
        'Small molecule': [],
        'Water': [],
        'rRNA': [],
        'tRNA': [],
        'mRNA': [],
        'DNA': [],
    }
    grouped_data = data.groupby(['Seed'])
    for condition, condition_data in grouped_data:
        final_agents = condition_data.loc[condition_data.loc[:, 'Time']==26000, 'Agent ID'].tolist()
        grouped_agents = condition_data.groupby(['Agent ID'])
        for agent, agent_data in grouped_agents:
            # Total of 4 cells die across all 3 seeds, exclude them
            if (agent not in final_agents) and (agent_data.loc[:, 'Wall cracked'].sum()==0):
                birth_time = agent_data.loc[:, 'Time'].min()
                birth_data = agent_data.loc[agent_data.loc[:, 'Time']==birth_time, :]
                death_time = agent_data.loc[:, 'Time'].max()
                death_data = agent_data.loc[agent_data.loc[:, 'Time']==death_time, :]
                for submass in submass_fc: 
                    birth_submass = birth_data.loc[:, f'{submass} mass'].to_numpy()[0]
                    submass_fc[submass].append(
                        death_data.loc[:, f'{submass} mass'].to_numpy()[0] / 
                        birth_submass
                    )
    submass_fc = pd.DataFrame(submass_fc)
    submass_fc = submass_fc.rename(columns={'Small molecule': 'Small\nmolecule'})
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.violinplot(data=submass_fc, color=(0, 0.4, 1)) 
    ax.set_ylabel(r'$\frac{\mathrm{Mass~at~division}}{\mathrm{Initial~mass}}$', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'out/analysis/paper_figures/fig_1c_submass_fc.svg',
        bbox_inches='tight')
    plt.close()


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
        'TolC monomer': (0, 0.4, 1),
        'AmpC monomer': (0, 0.4, 1),
        'MarR monomer': (0, 0.4, 1),
        'ompF mRNA': (0, 0.4, 1),
        'tolC mRNA': (0, 0.4, 1),
        'ampC mRNA': (0, 0.4, 1),
        'marR mRNA': (0, 0.4, 1),
    }
    # Convert to concentrations using periplasmic or cytoplasmic volume
    periplasmic = ['OmpF monomer', 'AmpC monomer', 'TolC monomer']
    for column in columns_to_plot:
        if column in periplasmic:
            data.loc[:, column] /= (data.loc[:, 'Volume'] * 0.2)
        else:
            data.loc[:, column] /= (data.loc[:, 'Volume'] * 0.8)
        data.loc[:, column] *= COUNTS_PER_FL_TO_NANOMOLAR
    monomer_cols = list(columns_to_plot)[:4]
    # Convert monomer concentrations to mM
    data.loc[:, monomer_cols] /= 1000
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
    axes[1].set_ylabel('Protein (\u03BCM)')
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
    axes[1].yaxis.set_label_coords(-0.5, 0.5)

    # Get fractional upper limit for MarR monomer expression
    marR_max = np.round(data.loc[:, 'MarR monomer'].max(), 1)
    axes[4].set_ylim([0, marR_max])
    axes[4].set_yticks([0, marR_max], [0, marR_max], fontsize=8)
    axes[4].spines.left.set_bounds([0, marR_max])

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


def make_figure_2d(data, metadata):
    # Make plots describing glucose depletion from environment
    # Convert glucose uptake to units of mmol / g DCW / hr
    exchange_data = data.loc[:, ['Dry mass', 'Exchanges']]
    glc_flux = exchange_data.apply(lambda x: x['Exchanges']['GLC[p]'] / N_A * 
        1000 / (x['Dry mass'] * 1e-15) / 2 * 3600, axis=1)
    data['Glucose intake'] = -glc_flux
    grouped_data = data.groupby('Agent ID').mean()
    print(f'Mean glucose intake (mmol/g DCW/hr): {grouped_data.loc[:, "Glucose intake"].mean()}')
    print(f'Std. dev. glucose intake (mmol/g DCW/hr): {grouped_data.loc[:, "Glucose intake"].std()}')
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(3, 3))
    sns.histplot(grouped_data.loc[:, "Glucose intake"], color=(0, 0.4, 1),
        ax=ax, linewidth=1)
    ax.set_xlabel('Glucose intake\n(mmol/g DCW/hr)', fontsize=10)
    ax.set_ylabel('Simulated cells', fontsize=10)
    ax.set_xticks(ax.get_xticks(), ax.get_xticks().astype(int), fontsize=10)
    ax.set_yticks(ax.get_yticks(), ax.get_yticks().astype(int), fontsize=10)
    plt.tight_layout()
    plt.savefig('out/analysis/paper_figures/fig_2d_glc_intake.svg',
        bbox_inches='tight')
    plt.close()

    field_data = metadata['Glucose'][10000]['fields']
    xticks = np.arange(0, 50, 5)
    xcoords = xticks + 2.5
    sample_times = [10400, 15600, 20800, 26000]
    cmap = matplotlib.colormaps['Greys']
    norm = matplotlib.colors.Normalize(
        vmin=0, vmax=26000)
    fig, ax = plt.subplots(figsize=(3,3))
    for sample_time in sample_times:
        cross_sec = np.array((field_data[sample_time]['GLC[p]'])).T[4]
        color = cmap(norm(sample_time))
        ax.plot(xcoords, cross_sec, c=color)
        ax.scatter(xcoords, cross_sec, c=color)
        ax.text(51, cross_sec.mean(), f'{np.round(sample_time/3600, 1)} hr',
            horizontalalignment='left', verticalalignment='center',
            c=color)
    ax.set_xlabel('Distance from left edge\nof environment (\u03BCm)')
    ax.set_ylabel('Cross-sectional glucose (mM)')
    plt.savefig('out/analysis/paper_figures/fig_2d_env_cross.svg',
        bbox_inches='tight')
    plt.close()


def make_figure_2e(data, metadata):
    # Create colormap where agents that are more related are closer on viridis colormap
    locations = np.array([agent['location'] for agent in data.loc[:, 'Boundary']])
    data['x'] = locations[:, 0]
    data['y'] = locations[:, 1]
    grouped_data = data.loc[:, ['Agent ID', 'x', 'y']].groupby('Agent ID')
    cmap = matplotlib.colormaps['viridis']
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2**7)
    agent_colors = {}
    for agent_id, agent_data in grouped_data:
        if len(agent_id) > 1:
            parent_id = agent_id[:-1]
            exp = 8 - len(parent_id)
        else:
            parent_id = '0'
            exp = 8
        binary_agent = int(parent_id, 2) * 2**exp
        color = cmap(norm(binary_agent))
        agent_colors[agent_id] = matplotlib.colors.rgb_to_hsv(color[:-1])
    plt.close()

    final_agents = data.loc[data.loc[:, 'Time']==26000, ['Agent ID', 'Boundary']]
    final_agents.rename(columns={'Boundary': 'boundary'}, inplace=True)
    agent_data = final_agents.set_index('Agent ID').T.to_dict()
    seed = data.loc[:, 'Seed'].unique()[0]
    snapshot_data = {
        'bounds': metadata['Glucose'][seed]['bounds'],
        'agents': {
            26000: agent_data
        },
        'fields': {26000: metadata['Glucose'][seed]['fields'][26000]},
        'snapshot_times': [26000],
        'agent_colors': agent_colors,
        'scale_bar_length': 5,
        'membrane_width': 0,
        'colorbar_decimals': 1,
        'default_font_size': 12,
        'figsize': (2, 2),
        'include_fields': [],
        'field_label_size': 0,
        'xlim': [10, 40],
        'ylim': [10, 40]
    }
    snapshots_fig = plot_snapshots(**snapshot_data)
    # New scale bar with reduced space between bar and label
    snapshots_fig.axes[1].artists[0].remove()
    scale_bar = anchored_artists.AnchoredSizeBar(
        snapshots_fig.axes[1].transData,
        5,
        "5 μm",
        "lower left",
        frameon=False,
        size_vertical=0.5,
        fontproperties={'size': 9}
    )
    snapshots_fig.axes[1].add_artist(scale_bar)
    # Remove time axis
    snapshots_fig.axes[0].remove()
    # Clean up tick marks
    snapshots_fig.axes[0].set_yticks([])
    snapshots_fig.axes[0].set_xticks([])
    # Resize subplot layout to fill space
    gs = matplotlib.gridspec.GridSpec(1, 1)
    snapshots_fig.axes[0].set_subplotspec(gs[0])
    title = snapshots_fig.axes[0].get_title()
    snapshots_fig.axes[0].set_title(title, pad=10, fontsize=12)
    plt.tight_layout()
    plt.savefig(f'out/analysis/paper_figures/fig_2e_snapshot_relatedness_{seed}.svg',
        bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(0.25, 2))
    fig.subplots_adjust(right=0.4)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax,
        orientation='vertical', label='Relatedness')
    ax.set_yticks([])
    ax.set_ylabel(ax.get_ylabel(), size=8)
    fig.savefig('out/analysis/paper_figures/fig_2e_cbar.svg')


def make_figure_2f(data, metadata):
    final_agents = data.loc[data.loc[:,'Time']==26000, 'Agent ID'].unique()
    # Convert glucose uptake to units of mmol / g DCW / hr
    exchange_data = data.loc[:, ['Dry mass', 'Exchanges']]
    glc_flux = exchange_data.apply(lambda x: x['Exchanges']['GLC[p]'] / N_A * 
        1000 / (x['Dry mass'] * 1e-15) / 2 * 3600, axis=1)
    data['Glucose intake'] = -glc_flux
    agent_data = data.groupby('Agent ID')
    df = {
        'doubling_times': [],
        'glucose_intake': [],
        'ribosome_conc': [],
        'initial_mass': [],
        'delta_mass': []
    }
    fig, ax = plt.subplots(figsize=(10, 2))
    for agent_id, agent in agent_data:
        if agent_id in final_agents:
            continue
        plt_data = agent.loc[agent.loc[:, 'Time']>agent.loc[:,'Time'].min(), :]
        ax.plot(plt_data.loc[:, 'Time'], plt_data.loc[:, 'Glucose intake'])
        df['doubling_times'].append(agent.loc[:, 'Time'].max() -
            agent.loc[:, 'Time'].min())
        df['glucose_intake'].append(agent.loc[:, 'Glucose intake'].mean())
        ribo_conc = (agent.loc[:, 'Active ribosomes']/agent.loc[:, 'Volume']
            * COUNTS_PER_FL_TO_NANOMOLAR)
        df['ribosome_conc'].append(ribo_conc.mean())
        df['initial_mass'].append(agent.loc[agent.loc[:, 'Time']==agent.loc[
            :, 'Time'].min(), 'Cell mass'].iloc[0])
        df['delta_mass'].append(agent.loc[agent.loc[:, 'Time']==agent.loc[
            :, 'Time'].max(), 'Dry mass'].iloc[0] - agent.loc[agent.loc[
                :, 'Time']==agent.loc[:, 'Time'].min(), 'Dry mass'].iloc[0])

    ax.set_ylabel('Glucose intake\n(mmol/g DCW/hr)')
    ax.set_xlabel('Time (sec.)')
    plt.savefig('out/glucose_intake_over_time.svg', bbox_inches='tight')
    plt.close()

    df = pd.DataFrame(df)
    res = smf.ols(formula='doubling_times ~ glucose_intake + ribosome_conc'
        '+ initial_mass + delta_mass', data=df).fit()
    print(res.summary())
    # Standardize regression coefficients
    std_beta = res.params[1:] * df.std()[1:] / df.std()[0]
    for var in std_beta.index:
        pearson_r = np.corrcoef(df['doubling_times'], df[var])[0, 1]
        print(f'Pratt index for {var} (std. coef = {std_beta[var]}, p = '
            f'{res.pvalues[var]}): {std_beta[var] * pearson_r}')

    # Confirm good model fit
    plt.scatter(df['doubling_times'], res.predict(df.iloc[:, 1:]))

    lit_data = pd.read_csv('/home/covertlab/Downloads/wpd_datasets.csv')
    plt.plot(lit_data.iloc[1:22, 0].astype(float), lit_data.iloc[1:22, 1].astype(float))
    plt.plot(lit_data.iloc[1:, 2].astype(float), lit_data.iloc[1:, 3].astype(float))
    ax = plt.gca()
    ax.set_xlim(min(lit_data.iloc[1:22, 0].astype(float)), 6.5)
    ax2 = ax.twiny()
    ax2.set_xlim(min(df['Initial lengths']), max(df['Final lengths']))
    import seaborn as sns
    sns.kdeplot(data=df[['Initial lengths', 'Final lengths']])


def make_figure_2g(data, metadata):
    # Perform linear regression on protein expression at final time point for
    # all possible pairs of the four antibiotic resistance genes
    data = restrict_data(data)
    monomers = ['OmpF monomer', 'AmpC monomer', 'TolC monomer',
        'MarR monomer']
    data.loc[:, monomers] = (data.loc[:, monomers].divide(data.loc[:, 'Volume'],
        axis=0) * COUNTS_PER_FL_TO_NANOMOLAR)
    avg_concs = data.loc[:, monomers + ['Agent ID']].groupby('Agent ID').mean()
    import seaborn as sns
    sns.pairplot(avg_concs, kind='reg')
    
    combos = list(combinations(monomers, 2))
    for monomer_1, monomer_2 in combos:
        r, p = pearsonr(avg_concs[monomer_1], avg_concs[monomer_2])
        adj_p = p * len(combos)
        print(f'{monomer_1} vs. {monomer_2}: r = {r},'
            f' Bonferroni corrected p = {adj_p}')

    print()


def make_figure_3a(data, metadata):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2))
    plot_colony_growth(data, axs)
    offset_time = SPLIT_TIME/3600
    min_time = np.round(-offset_time, 1)
    max_time = np.round(MAX_TIME/3600-offset_time, 1)
    ticks = np.array([int(min_time), 0, int(max_time)])
    ax = axs[0]
    ax.set_xticks(ticks+offset_time, ticks, size=8)
    ax.spines['bottom'].set_bounds(0, MAX_TIME/3600)
    ax.spines['left'].set_bounds(ax.get_ylim())
    ax.set_xlabel('Hours after tet. addition', size=8)
    ax.set_ylabel('Colony mass', size=8)
    yticklabels = [f'$10^{int(exp)}$' for exp in np.log10(ax.get_yticks())]
    ax.set_yticks(ax.get_yticks(), yticklabels, size=9)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.2, left=0.31, right=1)
    fig.savefig('out/analysis/paper_figures/fig_3a_tet_colony_mass.svg',
        bbox_inches='tight')
    plt.close()
    print('Done with Figure 3A.')


def make_figure_3b(data, metadata):
    data = data.loc[data.loc[:, 'Time']<=MAX_TIME, :]
    data = data.sort_values(['Condition', 'Agent ID', 'Time'])
    # Draw blue border around highlighted agent lineage
    highlight_agent_id = '0011111'
    plot_exp_growth_rate(data, metadata, highlight_agent_id)
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
    # Convert to concentration using cytoplasmic volume
    transition_data.loc[:, 'Active ribosomes'] /= (
        transition_data.loc[:, 'Volume'] * 0.8)
    transition_data.loc[:, 'Active ribosomes'] *= COUNTS_PER_FL_TO_NANOMOLAR / 1000
    transition_data.rename(columns={
        'Periplasmic tetracycline': 'Tetracycline\n(periplasm)',
        'Cytoplasmic tetracycline': 'Tetracycline\n(cytoplasm)',
        'Active ribosomes': 'Active\nribosomes',
    }, inplace=True)
    fig, axes = plt.subplots(1, 3, figsize=(5, 1.5))
    short_term_columns = {
        'Tetracycline\n(periplasm)': 0,
        'Tetracycline\n(cytoplasm)': 1,
        'Active\nribosomes': 2,
    }
    for column, ax_idx in short_term_columns.items():
        plot_timeseries(
            data=transition_data,
            axes=[axes.flat[ax_idx]],
            columns_to_plot={column: (0, 0.4, 1)},
            highlight_lineage='0011111',
            filter_time=False,
            background_alpha=0.5,
            background_linewidth=0.3)
    for ax in axes.flat:
        ylim = ax.get_ylim()
        yticks = np.round(ylim, 0).astype(int)
        ax.set_yticks(yticks, yticks, size=9)
        ax.set_xlabel(None)
        # Mark minutes since tetracycline addition
        ax.set_xticks(ticks=[11430/3600, 11490/3600, 11550/3600,
            11610/3600, 11670/3600], labels=[-2, -1, 0, 1, 2], size=9)
        ax.spines.bottom.set(bounds=(11400/3600, 11700/3600), linewidth=1,
            visible=True, color=(0, 0, 0), alpha=1)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(None)
        ax.set_title(ylabel, size=9)
    axes.flat[0].set_ylabel('\u03BCM', size=9, labelpad=0)
    axes.flat[1].set_xlabel('Minutes after tetracycline addition', size=9)
    # Use scientific notation for high active ribosome concentrations
    new_ylim = np.round(axes.flat[-1].get_ylim(), 0).astype(int)
    axes.flat[-1].set_yticks(new_ylim, new_ylim, size=9)
    axes.flat[-1].spines.left.set(bounds=new_ylim, linewidth=1,
        visible=True, color=(0, 0, 0), alpha=1)
    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.3, wspace=0.35)
    fig.savefig('out/analysis/paper_figures/fig_3c_tet_short_term.svg')
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
    }
    # Convert to concentrations using periplasmic or cytoplasmic volume
    periplasmic = ['OmpF monomer']
    for column in long_term_columns:
        if column in periplasmic:
            long_transition_data.loc[:, column] /= (
                long_transition_data.loc[:, 'Volume'] * 0.2)
        else:
            long_transition_data.loc[:, column] /= (
                long_transition_data.loc[:, 'Volume'] * 0.8)
        long_transition_data.loc[:, column] *= COUNTS_PER_FL_TO_NANOMOLAR
    fig, axes = plt.subplots(1, 3, figsize=(5, 1.5))
    for column, ax_idx in long_term_columns.items():
        plot_timeseries(
            data=long_transition_data,
            axes=[axes.flat[ax_idx]],
            columns_to_plot={column: (0, 0.4, 1)},
            highlight_lineage='0011111',
            filter_time=False,
            background_alpha=0.5,
            background_linewidth=0.3)
    split_hours = SPLIT_TIME/3600
    rounded_split_hours = np.round(split_hours, 1)
    for ax in axes.flat:
        ylim = ax.get_ylim()
        yticks = np.round(ylim, 0).astype(int)
        ax.set_yticks(yticks, yticks, size=9)
        # Mark hours since tetracycline addition
        xlim = np.array(ax.get_xlim())
        xticks = np.append(xlim, split_hours)
        xtick_labels = np.trunc(xticks-split_hours).astype(int).tolist()
        xtick_labels = [label if label!=int(-split_hours) else -rounded_split_hours 
            for label in xtick_labels]
        ax.set_xticks(ticks=xticks, labels=xtick_labels, size=9)
        ax.set_xlabel(None)
        ax.spines.bottom.set(bounds=(0, MAX_TIME/3600), linewidth=1,
            visible=True, color=(0, 0, 0), alpha=1)
        ylabel = ax.get_ylabel()
        ax.set_ylabel(None)
        ax.set_title(ylabel, size=9, pad=12)
    axes.flat[0].set_ylabel('nM', size=9, labelpad=-6)
    fig.supxlabel('Hours after tetracycline addition', size=9)
    # Use scientific notation for high OmpF monomer concentrations
    new_ylim = np.round(np.array(axes.flat[-1].get_ylim())/10000, 0) * 10000
    axes.flat[-1].set_yticks(new_ylim, (new_ylim/10000).astype(int), size=9)
    axes.flat[-1].spines.left.set(bounds=new_ylim, linewidth=1,
        visible=True, color=(0, 0, 0), alpha=1)
    axes.flat[-1].text(0, 1, r'x$10^4$', transform=axes.flat[-1].transAxes, size=8)
    plt.tight_layout()
    fig.subplots_adjust(left=0.08, right=0.99, top=0.8, bottom=0.3, wspace=0.35)
    fig.savefig('out/analysis/paper_figures/fig_3d_tet_long_term.svg')
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
    fig.savefig('out/analysis/paper_figures/fig_3f_tet_gene_exp.svg', bbox_inches='tight')
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
    fig, ax = plt.subplots(1, 1, figsize=(1.75,2))
    plot_protein_synth_inhib(data, ax, literature)
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0.25, left=0.2, right=1)
    plt.savefig('out/analysis/paper_figures/protein_synth_inhib.svg', bbox_inches='tight')
    plt.close()
    print('Done with Figure 3F.')


def make_figure_3g(data, metadata):
    final_data = data.loc[data.loc[:, 'Time']==MAX_TIME, :]
    final_data['External Tetracycline (\u03BCM)'] = np.round(
        final_data.loc[:, 'Initial external tet.'] * 1000, 3)
    
    mM_fL_to_mol = 1e-18
    peri_tet = (final_data.loc[:, 'Periplasmic tetracycline'] * final_data.loc[
        :, 'Volume'] * 0.2 * mM_fL_to_mol)
    cyto_tet = (final_data.loc[:, 'Cytoplasmic tetracycline'] * final_data.loc[
        :, 'Volume'] * 0.8 * mM_fL_to_mol)
    tet_30s = final_data.loc[:, 'Inactive 30S subunit'] / N_A
    total_tet = peri_tet + cyto_tet + tet_30s
    mol_per_fL_to_uM = 1e21
    final_data['Internal Tetracycline (\u03BCM)'] = (total_tet /
        final_data.loc[:, 'Volume'] * mol_per_fL_to_uM)

    fig, ax = plt.subplots()
    grouped_data = final_data.groupby('Condition')
    avg_data = grouped_data.mean()
    std_dev = grouped_data.std()
    ax.scatter(avg_data['External Tetracycline (\u03BCM)'],
        avg_data['Internal Tetracycline (\u03BCM)'], c='k')
    plt.errorbar(avg_data['External Tetracycline (\u03BCM)'],
        avg_data['Internal Tetracycline (\u03BCM)'], yerr=
        std_dev['Internal Tetracycline (\u03BCM)'], fmt='o', c='k')
    best_fit = np.polynomial.Polynomial.fit(
        avg_data['External Tetracycline (\u03BCM)'],
        avg_data['Internal Tetracycline (\u03BCM)'], 1, window=(
        min(avg_data['External Tetracycline (\u03BCM)']),
        max(avg_data['External Tetracycline (\u03BCM)'])))
    xx, yy = best_fit.linspace()
    ax.plot(xx, yy, 'k--')
    # Tetracycline accumulation measured by Thanassi et al. 1995
    ax.scatter([5], [75])
    ax.text(5.5, 75, 'Thanassi et al. 1995\n5.7x predicted', c='tab:blue')
    # Calculate concentration from Argast and Beck 1985
    avg_cell_volume = data.loc[data.loc[:, 'Condition']=='Glucose', 'Volume'].mean()
    argast = 7.6e-20 / avg_cell_volume * mol_per_fL_to_uM
    ax.scatter([5], [argast])
    ax.text(5.5, argast, 'Argast and Beck 1985\n4.1x predicted', c='tab:orange')
    ax.set_xlabel('External Tetracycline (\u03BCM)')
    ax.set_ylabel('Internal Tetracycline (\u03BCM)')
    import seaborn as sns
    sns.despine(ax=ax, offset=3, trim=True)
    plt.tight_layout()
    plt.subplots_adjust(top=1, bottom=0.25, left=0.2, right=1)
    plt.savefig('out/analysis/paper_figures/supp_tet_ompf_acrab.svg', bbox_inches='tight')
    plt.close()
    print('Done with Figure 3G.')


def make_figure_4a(data, metadata):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2))
    plot_colony_growth(data, axs, antibiotic_col='Initial external amp.',
        mic=5.724, antibiotic='Amp.')
    offset_time = SPLIT_TIME/3600
    min_time = np.round(-offset_time, 1)
    max_time = np.round(MAX_TIME/3600-offset_time, 1)
    ticks = np.array([int(min_time), 0, int(max_time)])
    ax = axs[0]
    ax.set_xticks(ticks+offset_time, ticks, size=8)
    ax.spines['bottom'].set_bounds(0, MAX_TIME/3600)
    ax.spines['left'].set_bounds(ax.get_ylim())
    ax.set_xlabel('Hours after amp. addition', size=8)
    ax.set_ylabel('Colony mass', size=8)
    yticklabels = [f'$10^{int(exp)}$' for exp in np.log10(ax.get_yticks())]
    ax.set_yticks(ax.get_yticks(), yticklabels, size=9)
    plt.tight_layout()
    fig.subplots_adjust(top=0.95, bottom=0.2, left=0.31, right=1)
    fig.savefig('out/analysis/paper_figures/fig_4a_amp_colony_mass.svg')
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
        'divergent', [(0, 0, 0), (1, 1, 1), (0.678, 0, 0.125)])
    magnitude = data.loc[:, fc_col].abs().max()
    norm = matplotlib.colors.Normalize(vmin=-magnitude, vmax=magnitude)
    snapshot_times = np.array([1.9, 3.2, 4.5, 5.8, 7.1]) * 3600
    snapshot_times = np.array([3.2, 4.5, 5.8, 7.1]) * 3600

    # Draw blue border around highlighted agent lineage
    highlight_agent_id = '001111111'
    highlight_agent_ids = [highlight_agent_id[:i+1] for i  in range(len(highlight_agent_id))]
    highlight_agent = {agent_id: {
        'membrane_width': 0.5, 'membrane_color': (0, 0.4, 1)}
        for agent_id in highlight_agent_ids}

    fig = plot_tag_snapshots(
        data=data, metadata=metadata, tag_colors={fc_col: {'cmp': cmp, 'norm': norm}},
        snapshot_times=snapshot_times, return_fig=True, figsize=(6, 1.5),
        highlight_agent=highlight_agent, show_membrane=True)
    fig.axes[0].set_xticklabels(
        np.abs(np.round(fig.axes[0].get_xticks()/3600 - SPLIT_TIME/3600, 1)))
    fig.axes[0].set_xlabel('Hours after ampicillin addition')
    fig.savefig('out/analysis/paper_figures/fig_4b_amp_snapshots.svg',
        bbox_inches='tight')
    plt.close()
    print('Done with Figure 4B.')


def make_figure_4c(data, metadata):
    plot_ampc_phylo(data)
    print('Done with Figure 4C.')


def make_figure_4d(data, metadata):
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    axs = [ax]
    plot_death_timescale_analysis(data, axs)
    plt.savefig('out/analysis/paper_figures/fig_4d_misc.svg')
    plt.close()
    print('Done with Figure 4D.')


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
        '1b': ['Glucose'],
        '2b': ['Glucose'],
        '2c': ['Glucose'],
        '2d': ['Glucose'],
        '2e': ['Glucose'],
        '2f': ['Glucose'],
        '2g': ['Glucose'],
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
        '3g': ['Glucose', 'Tetracycline (0.5 mg/L)', 'Tetracycline (1 mg/L)',
            'Tetracycline (1.5 mg/L)', 'Tetracycline (2 mg/L)',
            'Tetracycline (4 mg/L)'],
        '4a': ['Glucose', 'Ampicillin (0.5 mg/L)', 'Ampicillin (1 mg/L)',
            'Ampicillin (1.5 mg/L)', 'Ampicillin (2 mg/L)',
            'Ampicillin (4 mg/L)'],
        '4b': ['Glucose', 'Ampicillin (2 mg/L)'],
        '4c': ['Glucose', 'Ampicillin (2 mg/L)'],
        '4d': ['Ampicillin (0.5 mg/L)', 'Ampicillin (1 mg/L)',
            'Ampicillin (1.5 mg/L)', 'Ampicillin (2 mg/L)',
            'Ampicillin (4 mg/L)'],
    }
    seeds = {
        '1a': [10000],
        '1b': [0, 100, 10000],
        '2b': [10000],
        '2c': [10000],
        '2d': [10000],
        '2e': [10000],
        '2f': [10000],
        '2g': [10000],
        '3a': [0],
        '3b': [0],
        '3c': [0],
        '3d': [0],
        '3e': [0, 100, 10000],
        '3f': [0],
        '3g': [0],
        '4a': [0],
        '4b': [0],
        '4c': [0],
        '4d': [0],
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

# TODO: Figure to show that environmental ampicillin concentration does not change much over time

if __name__ == '__main__':
    main()

"""
Antibiotic simulation data stats
{
  db: 'simulations',
  collections: 3,
  views: 0,
  objects: 6883245,
  avgObjSize: 1590217.76253642,
  dataSize: 10945858462890,
  storageSize: 3095633932288,
  freeStorageSize: 439554437120,
  indexes: 7,
  indexSize: 435052544,
  indexFreeStorageSize: 81178624,
  totalSize: 3096068984832,
  totalFreeStorageSize: 439635615744,
  scaleFactor: 1,
  fsUsedSize: 3102315601920,
  fsTotalSize: 3112330854400,
  ok: 1
}
"""
