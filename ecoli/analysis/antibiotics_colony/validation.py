import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit, root_scalar

from ecoli.analysis.antibiotics_colony import (DE_GENES, MAX_TIME, SPLIT_TIME,
                                               restrict_data)


def plot_colony_growth(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    antibiotic: str = 'Tetracycline',
    antibiotic_col: str = 'Initial external tet.',
    mic: float = 3.375
) -> None:
    '''Plot traces of total colony mass and total colony growth rate.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', and
            'Dry Mass'. The first experimental condition in the 'Condition'
            column is treated as a control (default gray). Include at most
            2 conditions and 1 seed per condition.
        axes: Instance of Matplotlib Axes to plot on
        antibiotic: Name of antibiotic to put as legend title
        antibiotic_col: Name of column with initial external antibiotic concs.
        mic: Minimum inhibitory concentration (uM, rounded to 3 decimal places)
    '''
    data['antibiotic_conc'] = np.round(data.loc[:, antibiotic_col] * 1000, 3)
    columns_to_include = ['Dry mass', 'Time', 'antibiotic_conc', 'Condition']
    data = data.loc[:, columns_to_include]
    mask = (data.loc[:, 'antibiotic_conc']==0) & (
        data.loc[:, 'Time'] > SPLIT_TIME) & (
        data.loc[:, 'Time'] <= MAX_TIME)
    remaining_glucose_data = data.loc[mask, :]
    data = restrict_data(data)
    data = pd.concat([data, remaining_glucose_data])
    data = data.groupby(['antibiotic_conc', 'Time']).sum().reset_index()
    
    # Convert time to hours
    data.loc[:, 'Time'] = data.loc[:, 'Time'] / 3600

    cmap = matplotlib.colormaps['Greys']
    tet_min = data.loc[:, 'antibiotic_conc'].min()
    tet_max = data.loc[:, 'antibiotic_conc'].max()
    norm = matplotlib.colors.Normalize(
        vmin=1.5*tet_min-0.5*tet_max, vmax=tet_max)
    antibiotic_concs = data.loc[:, 'antibiotic_conc'].unique()
    palette = {antibiotic_conc: cmap(norm(antibiotic_conc))
        for antibiotic_conc in antibiotic_concs}
    palette[mic] = (0, 0.4, 1)

    sns.lineplot(
        data=data, x='Time', y='Dry mass',
        hue='antibiotic_conc', ax=ax, palette=palette, errorbar=None)
    # Set y-limits so that major ticks surround data
    curr_ylimits = np.log10([
        data.loc[:, 'Dry mass'].min(),
        data.loc[:, 'Dry mass'].max()])
    ax.set_ylim(10**np.floor(curr_ylimits[0]), 10**np.ceil(curr_ylimits[1]))
    max_x = np.round(data.loc[:, 'Time'].max(), 1)
    ticks = list(np.arange(0, max_x, 2, dtype=int)) + [max_x]
    labels = [str(tick) for tick in ticks]
    ax.set_xticks(ticks=ticks, labels=labels)
    # Log scale so linear means exponential growth
    ax.set(yscale='log')
    ax.legend(frameon=False, title=f'{antibiotic} (uM)', fontsize=8, title_fontsize=8)
    sns.despine(ax=ax, offset=3, trim=True)


def plot_synth_prob_fc(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    genes_to_plot: List[str] = None,
    filter: float = 0.0
) -> None:
    '''Plot scatter plot of simulated and experimental log2 fold change for
    synthesis probabilities of key genes regulated during tetracycline exposure.
    Uses log-scale colormap to show relative mRNA counts.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', 'Seed',
            and 'Dry Mass'. The first experimental condition in the 'Condition'
            column is treated as a control.
        ax: Single instance of Matplotlib Axes to plot on.
        genes_to_plot: List of gene names to include in plot.
        filter: Minimum average instantaneous mRNA count that will still be plotted.
    '''
    data = data.loc[data.loc[:, 'Time'] <= MAX_TIME, :]
    data.loc[:, 'ompF mRNA'] += data.loc[:, 'micF-ompF duplex']
    synth_probs = [f'{gene} synth prob' for gene in genes_to_plot
        if gene not in ['MicF']]
    mrna_cols = [f'{gene} mRNA' for gene in genes_to_plot
        if gene not in ['MicF']]
    mrna_data = data.loc[:, synth_probs + mrna_cols + ['Condition']]
    conditions = mrna_data.loc[:, 'Condition'].unique()
    # Get average synthesis probabilities and mRNA counts per condition
    mrna_data = mrna_data.groupby(['Condition']).mean()
    # Convert to relative synth probs assuming that first condition is control
    relative_data = np.log2(mrna_data.loc[conditions[1], synth_probs] /
        mrna_data.loc[conditions[0], synth_probs])
    relative_data.index = [mrna_name.split(' synth prob')[0] for mrna_name in relative_data.index]
    relative_data = pd.DataFrame(relative_data).rename(columns={
        0: 'Sim. RNA synth. prob. $\mathregular{log_2 FC}$'})
    relative_data['Avg. sim. mRNA'] = mrna_data.loc[conditions[1], mrna_cols].to_numpy()
    # Compare against literature relative amounts
    tet_degenes = DE_GENES.loc[:, ['Gene name', 'Fold change']].set_index(
        'Gene name').drop(['MicF'], axis=0).rename(columns={
            'Fold change': 'Literature RNA $\mathregular{log_2 FC}$'})
    tet_degenes.loc[:, 'Literature RNA $\mathregular{log_2 FC}$'] = np.log2(
        tet_degenes.loc[:, 'Literature RNA $\mathregular{log_2 FC}$'])
    relative_data = relative_data.join(tet_degenes, how='inner').reset_index()
    relative_data = relative_data.rename(columns={'index': 'Gene'})

    # Filter extremely low counts
    relative_data = relative_data.loc[relative_data.loc[:, 'Avg. sim. mRNA'] >= filter]
    norm = matplotlib.colors.LogNorm(vmin=relative_data.loc[:, 'Avg. sim. mRNA'].min(),
        vmax=relative_data.loc[:, 'Avg. sim. mRNA'].max())
    ompf_filter = (relative_data.loc[:, 'Gene'] == 'ompF')
    sns.scatterplot(data=relative_data[ompf_filter],
        x='Literature RNA $\mathregular{log_2 FC}$',
        y='Sim. RNA synth. prob. $\mathregular{log_2 FC}$', ax=ax, 
        c=(0, 0.4, 1), legend=False, edgecolor='k')
    sns.scatterplot(data=relative_data[~ompf_filter],
        x='Literature RNA $\mathregular{log_2 FC}$',
        y='Sim. RNA synth. prob. $\mathregular{log_2 FC}$', ax=ax, hue='Avg. sim. mRNA',
        hue_norm=norm, palette='binary', legend=False, edgecolor='k')
    min_fc = relative_data.loc[:, 'Literature RNA $\mathregular{log_2 FC}$'].min()
    max_fc = relative_data.loc[:, 'Literature RNA $\mathregular{log_2 FC}$'].max()
    validation_line = np.linspace(min_fc, max_fc, 2)
    ax.plot(validation_line, validation_line, '--', c='0.4')
    sns.despine(ax=ax, offset=3, trim=True)


def plot_mrna_fc(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    genes_to_plot: List[str] = None,
    filter: float = 0.0
) -> None:
    '''Plot scatter plot of simulated and experimental log2 fold change for
    final mRNA concentrations of key genes regulated during tetracycline exposure.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', and 'Dry Mass'.
            The first experimental condition in the 'Condition' column is treated as
            a control.
        genes_to_plot: List of gene names to include in plot.
        filter: Minimum average instantaneous mRNA count that will still be plotted.
    '''
    data = data.loc[data.loc[:, 'Time'] <= MAX_TIME, :]
    data.loc[:, 'ompF mRNA'] += data.loc[:, 'micF-ompF duplex']
    mrna_cols = [f'{gene} mRNA' for gene in genes_to_plot
        if gene not in ['MicF']] + ['GAPDH mRNA']
    mrna_data = data.loc[:, mrna_cols + ['Condition', 'Volume']]
    conditions = mrna_data.loc[:, 'Condition'].unique()
    # Get mRNA concentrations aggregated over entire final colonies
    mrna_data = mrna_data.set_index(['Condition'])
    avg_inst_rna = mrna_data.loc[conditions[1], mrna_cols].mean().to_numpy()
    mrna_data = mrna_data.divide(mrna_data.loc[:, 'Volume'], axis=0).drop(
        ['Volume'], axis=1).reset_index()
    mrna_data = mrna_data.groupby(['Condition']).mean()
    # Normalize by housekeeping gene expression
    mrna_data = mrna_data.divide(mrna_data.loc[:, 'GAPDH mRNA'], axis=0)
    # Convert to relative amounts assuming that first condition is control
    relative_data = pd.DataFrame(np.log2(mrna_data.loc[conditions[1]] /
            mrna_data.loc[conditions[0]]))
    relative_data.index = [mrna_name.split(' mRNA')[0] for mrna_name in relative_data.index]
    relative_data = relative_data.rename(columns={0: 'Sim. RNA $\mathregular{log_2 FC}$'})
    relative_data['Avg. sim. mRNA'] = avg_inst_rna
    # Compare against literature relative amounts
    tet_degenes = DE_GENES.loc[:, ['Gene name', 'Fold change']].set_index(
        'Gene name').drop(['MicF'], axis=0).rename(columns={
            'Fold change': 'Literature RNA $\mathregular{log_2 FC}$'})
    tet_degenes.loc[:, 'Literature RNA $\mathregular{log_2 FC}$'] = np.log2(
        tet_degenes.loc[:, 'Literature RNA $\mathregular{log_2 FC}$'])
    relative_data = relative_data.join(tet_degenes, how='inner').reset_index()
    relative_data = relative_data.rename(columns={'index': 'Gene'})

    # Filter extremely low counts
    relative_data = relative_data.loc[relative_data.loc[:, 'Avg. sim. mRNA'] >= filter]
    norm = matplotlib.colors.LogNorm(vmin=relative_data.loc[:, 'Avg. sim. mRNA'].min(),
        vmax=relative_data.loc[:, 'Avg. sim. mRNA'].max())
    # Highlight ompF in blue
    ompf_filter = (relative_data.loc[:, 'Gene'] == 'ompF')
    sns.scatterplot(data=relative_data[ompf_filter],
        x='Literature RNA $\mathregular{log_2 FC}$',
        y='Sim. RNA $\mathregular{log_2 FC}$', ax=ax, 
        c=(0, 0.4, 1), legend=False, edgecolor='k')
    sns.scatterplot(data=relative_data[~ompf_filter],
        x='Literature RNA $\mathregular{log_2 FC}$',
        y='Sim. RNA $\mathregular{log_2 FC}$', ax=ax, hue='Avg. sim. mRNA',
        hue_norm=norm, palette='binary', legend=False, edgecolor='k')
    sm = plt.cm.ScalarMappable(cmap='binary', norm=norm)
    sm.set_array([])
    ax.figure.colorbar(sm, ax=ax.figure.axes, pad=0.01,
        label='Avg. instantaneous mRNAs/cell')
    min_fc = relative_data.loc[:, 'Literature RNA $\mathregular{log_2 FC}$'].min()
    max_fc = relative_data.loc[:, 'Literature RNA $\mathregular{log_2 FC}$'].max()
    validation_line = np.linspace(min_fc, max_fc, 2)
    ax.plot(validation_line, validation_line, '--', c='0.4')
    sns.despine(ax=ax, offset=3, trim=True)


def plot_protein_synth_inhib(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    literature: pd.DataFrame = None,
):
    '''Plot scatter plot of normalized % protein synthesis inhibition across a 
    variety of tetracycline concentrations.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', 'Seed',
            and 'Dry Mass'.
        ax: Single instance of Matplotlib Axes to plot on.
        literature: DataFrames with 3 columns: 'Tetracycline', 'Percent inhibition',
            and 'Source'
    '''
    # Assume tetracycline is added at t = 0 (reached equilibrium by 400 s)
    sampled_time = data.loc[(data.loc[:, 'Time'] > 400) & (data.loc[:, 'Time'] < 500), :]
    sampled_time['Total ribosomes'] = (sampled_time.loc[:, 'Active ribosomes'] +
        sampled_time.loc[:, 'Inactive 30S subunit'])
    sampled_time.loc[:, 'Cytoplasmic tetracycline'] *= 1000
    sampled_time = sampled_time.sort_values('Time')

    grouped_agents = sampled_time.groupby(['Condition', 'Seed', 'Agent ID'])
    normed_data = {
        'Condition': [],
        'Normed delta': [],
        'Tetracycline': []
    }
    for (condition, seed, agent_id), agent_data in grouped_agents:
        protein_deltas = np.diff(agent_data.loc[:, 'Protein mass'])
        # Ignore final timestep (protein mass deltas require two timepoints)
        agent_data = agent_data.iloc[:-1, :]
        # Normalize by ribosome and RNA counts to match cell-free system
        normed_deltas = protein_deltas / agent_data.loc[:,
            'Total ribosomes'] / agent_data.loc[:, 'Total mRNA']
        normed_data['Condition'].append(condition)
        normed_data['Normed delta'].append(normed_deltas.mean())
        normed_data['Tetracycline'].append(agent_data.loc[
            :, 'Cytoplasmic tetracycline'].mean())
    normed_data = pd.DataFrame(normed_data)
    normed_data = normed_data.groupby(['Condition']).mean()
    control = normed_data.index[normed_data['Tetracycline']==0]
    normed_data['Percent inhibition'] = 1 - (normed_data.loc[
        :, 'Normed delta'] / normed_data.loc[control, 'Normed delta'].to_numpy())
    normed_data['Source'] = ['Simulation'] * len(normed_data)
    normed_data = normed_data.loc[:, ['Percent inhibition',
        'Tetracycline', 'Source']]
    normed_data = normed_data.loc[normed_data.loc[:, 'Tetracycline']>0, :]
    palette = {'Simulation': (0, 0.4, 1)}
    if literature is not None:
        gray = 0.3
        for source in literature.loc[:, 'Source'].unique():
            palette[source] = str(gray)
            gray += 0.3
        normed_data = pd.concat([normed_data, literature])
    fig, ax = plt.subplots(1, 1)
    def func(x, a, b, c):
        return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b))

    
    ic_50 = []
    normed_data = normed_data.sort_values('Source')
    for i, source in enumerate(normed_data.loc[:, 'Source'].unique()):
        source_data = normed_data.loc[normed_data.loc[:, 'Source']==source, :]
        fittedParameters, pcov = curve_fit(func, source_data['Tetracycline'], source_data['Percent inhibition'])
        x = np.linspace(source_data['Tetracycline'].min(), source_data['Tetracycline'].max(), 10000)
        pred = func(x, *(fittedParameters))
        ax.plot(x, pred, c=palette[source])
        sol = root_scalar(lambda x: func(x, *fittedParameters) - 0.5, x0=10, x1=5)
        ic_50 += [' (' + r'$\mathregular{IC_{50}} = $' + str(np.round(sol.root, 1)) + ' uM)'] * len(source_data)
    normed_data.loc[:, 'Source'] = normed_data.loc[:, 'Source'].str.cat(ic_50)

    new_palette = {}
    sources_new = normed_data.loc[:, 'Source'].unique()
    for source in sources_new:
        for orig_source in palette:
            if orig_source in source:
                new_palette[source] = palette[orig_source]
    palette = new_palette

    sns.scatterplot(data=normed_data, x='Tetracycline',
        y='Percent inhibition', hue='Source', ax=ax, palette=palette)
    ax.set_xscale('log')
    sns.despine(offset=3, trim=True)
    plt.legend(frameon=False)
    ax.set_xticks(ax.get_xticks(minor=True)[16:40], minor=True)
    ax.set_ylabel(r'% inhibtion protein synthesis')
    ax.set_xlabel('Tetracycline (cytoplasm, uM)')

    plt.savefig('out/analysis/paper_figures/synth_inhib.svg')
    plt.close()


def plot_mass_fraction(
    data: pd.DataFrame,
):
    '''Plot line plots of submass fold changes over generation time.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest.
    '''
    submasses = ['Cell', 'Dry', 'Water', 'Protein', 'rRNA', 
        'mRNA', 'tRNA', 'DNA', 'Small molecule']
    grouped_agents = data.groupby(['Condition', 'Seed', 'Agent ID'])
    for (condition, seed, agent_id), agent_data in grouped_agents:
        # Start all agents at t = 0 min
        times = agent_data.loc[:, 'Time'] - agent_data.loc[:, 'Time'].iloc[0]
        times = times / 60
        for submass in submasses:
            agent_data[f'{submass} fold change'] = agent_data.loc[
                :, f'{submass} mass'] / agent_data.loc[
                    :, f'{submass} mass'].iloc[0]
            plt.plot(times, agent_data.loc[:, f'{submass} fold change'],
                label=f'{submass}')
        plt.legend()
        plt.xlabel('Time (min)')
        plt.ylabel('Mass fold change (normalized to t = 0 min)')
        out_dir = f'out/analysis/paper_figures/{condition}/{seed}'
        os.makedirs(out_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(f'{out_dir}/{agent_id}.svg')
        plt.close()

