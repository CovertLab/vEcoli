import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit, root_scalar

from ecoli.analysis.antibiotics_colony import (DE_GENES, MAX_TIME, SPLIT_TIME,
                                               restrict_data, AVG_GEN_TIME)


def plot_colony_growth(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    antibiotic: str = 'Tet.',
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
    antibiotic_min = data.loc[:, 'antibiotic_conc'].min()
    antibiotic_max = data.loc[:, 'antibiotic_conc'].max()
    norm = matplotlib.colors.Normalize(
        vmin=1.5*antibiotic_min-0.5*antibiotic_max, vmax=antibiotic_max)
    antibiotic_concs = data.loc[:, 'antibiotic_conc'].unique()
    palette = {antibiotic_conc: cmap(norm(antibiotic_conc))
        for antibiotic_conc in antibiotic_concs}
    palette[mic] = (0, 0.4, 1)

    sns.lineplot(
        data=data, x='Time', y='Dry mass',
        hue='antibiotic_conc', ax=ax, palette=palette, errorbar=None)
    # Set y-limits so that major ticks surround data
    log_ylim = 10**np.round(np.log10([
        data.loc[:, 'Dry mass'].min(),
        data.loc[:, 'Dry mass'].max()]), 0)
    new_ymin = min(log_ylim[0], data.loc[:, 'Dry mass'].min())
    new_ymax = max(log_ylim[1], data.loc[:, 'Dry mass'].max())
    ax.set_ylim(new_ymin, new_ymax)
    max_x = np.round(data.loc[:, 'Time'].max(), 1)
    ticks = list(np.arange(0, max_x, 2, dtype=int)) + [max_x]
    labels = [str(tick) for tick in ticks]
    ax.set_xticks(ticks=ticks, labels=labels)
    # Log scale so linear means exponential growth
    ax.set(yscale='log')
    ax.text(0, 1, f'{antibiotic} (\u03BCM)', transform=ax.transAxes, size=8)
    ypos = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    concs = list(palette.keys())
    colors = list(palette.values())
    for i, y in enumerate(ypos):
        ax.text(0, y, concs[i], c=colors[i],
            transform=ax.transAxes, size=8)
    ax.legend().remove()
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
    mrna_counts = relative_data.loc[:, 'Avg. sim. mRNA'].sort_values().to_numpy()
    min_nonzero = mrna_counts[mrna_counts!=0][0]
    norm = matplotlib.colors.LogNorm(vmin=min_nonzero, vmax=mrna_counts[-1])
    sns.scatterplot(data=relative_data,
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
    mrna_counts = relative_data.loc[:, 'Avg. sim. mRNA'].sort_values().to_numpy()
    min_nonzero = mrna_counts[mrna_counts!=0][0]
    norm = matplotlib.colors.LogNorm(vmin=min_nonzero, vmax=mrna_counts[-1])
    sns.scatterplot(data=relative_data,
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
    normed_data['Source'] = ['This model'] * len(normed_data)
    normed_data = normed_data.loc[:, ['Percent inhibition',
        'Tetracycline', 'Source']]
    normed_data = normed_data.loc[normed_data.loc[:, 'Tetracycline']>0, :]
    palette = {'This model': (0, 0.4, 1)}
    if literature is not None:
        gray = 0.3
        for source in literature.loc[:, 'Source'].unique():
            palette[source] = str(gray)
            gray += 0.3
        normed_data = pd.concat([normed_data, literature])
    # 3-parameter sigmoid function used to fit inhibition curves
    def func(x, a, b, c):
        return  a * np.power(x, b) / (np.power(c, b) + np.power(x, b))
    ic_50 = {}
    normed_data = normed_data.sort_values('Source')
    for i, source in enumerate(normed_data.loc[:, 'Source'].unique()):
        source_data = normed_data.loc[normed_data.loc[:, 'Source']==source, :]
        fittedParameters, pcov = curve_fit(func, source_data['Tetracycline'], source_data['Percent inhibition'])
        x = np.linspace(source_data['Tetracycline'].min(), source_data['Tetracycline'].max(), 10000)
        pred = func(x, *(fittedParameters))
        ax.plot(x, pred, c=palette[source], linewidth=1)
        sol = root_scalar(lambda x: func(x, *fittedParameters) - 0.5, x0=10, x1=5)
        ic_50[source] = str(np.round(sol.root, 1)) + ' \u03BCM'

    new_palette = {}
    sources_new = normed_data.loc[:, 'Source'].unique()
    for source in sources_new:
        for orig_source in palette:
            if orig_source in source:
                new_palette[source] = palette[orig_source]
    palette = new_palette
    markers = {source: 'o' for source in sources_new}
    markers['This model'] = '^'

    sns.scatterplot(data=normed_data, x='Tetracycline',
        y='Percent inhibition', hue='Source', ax=ax, palette=palette,
        style='Source', markers=markers, legend=False, s=16)
    ax.set_xscale('log')
    ax.set_xticks([0.1, 100])
    sns.despine(offset=3, trim=True)
    xticks = np.array([0.1, 1, 10, 100])
    xticklabels = [f'$10^{{{int(exp)}}}$' for exp in np.log10(xticks)]
    ax.set_xticks(xticks, xticklabels, size=8)
    ax.set_yticks([0, 0.5, 1], [0, 0.5, 1], size=8)
    ax.text(0, 0.9, 'This model', c=palette['This model'],
        size=8, transform=ax.transAxes, weight='bold')
    for i, source in enumerate(palette):
        if source != 'This model':
            ax.text(0, 0.8-0.1*i, source, c=palette[source],
                size=8, transform=ax.transAxes)
    ax.legend().remove()
    ax.set_ylabel('% protein synth. inhib.', size=9)
    ax.set_xlabel('Tetracycline (\u03BCM)', size=9)


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


def plot_death_timescale_analysis(
    data: pd.DataFrame,
    axs: List[plt.Axes],
    antibiotic_col: str = 'Initial external amp.',
    mic: float = 5.724
):
    '''Create histogram of cell age (time since birth) until death and labelled
    scatterplot of average time exposed to ampicillin until death against
    initial external ampicillin concentration.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest.
        axes: List of 2 matplotlib Axes instances on which to draw plots
        antibiotic_col: Name of column with initial external antibiotic concs.
        mic: Minimum inhibitory concentration (uM, rounded to 3 decimal places)
    '''
    data['antibiotic_conc'] = np.round(data.loc[:, antibiotic_col] * 1000, 3)
    # Remove data for truncated first generation of cells post-antibiotic addition
    discard_ids = data.loc[data.loc[:, 'Time'] == 11550, 'Agent ID'].unique()
    grouped_data = data.loc[~np.isin(data.loc[:, 'Agent ID'], discard_ids), :].groupby(
        'Condition')
    
    # Compile condition-specific lists of age of cell and total time that cell's
    # lineage was exposed to ampicillin at time of cracking and death
    death_times = {}
    cracking_times = {}
    for condition, condition_data in grouped_data:
        all_agents = condition_data.loc[:, 'Agent ID'].unique()
        death_times[condition] = {'age': [], 'exposure': [], 'amp': 
            condition_data.loc[:, 'antibiotic_conc'].iloc[0]}
        cracking_times[condition] = {'age': [], 'exposure': [], 'amp': 
            condition_data.loc[:, 'antibiotic_conc'].iloc[0]}
        agents_data = condition_data.groupby('Agent ID')
        for agent_id, agent_data in agents_data:
            crack_time = agent_data.loc[agent_data.loc[:, 'Wall cracked'],
                'Time'].to_numpy()
            if len(crack_time)!= 0:
                birth_time = agent_data.loc[:, 'Time'].min()
                cracking_times[condition]['age'].append(crack_time[0] - birth_time)
                cracking_times[condition]['exposure'].append(
                    crack_time[0] - SPLIT_TIME)
                # Cell has no descendants
                if agent_id + '0' not in all_agents:
                    death_time = agent_data.loc[:, 'Time'].max()
                    # Cell did not survive to end of simulation
                    if death_time != MAX_TIME:
                        death_times[condition]['age'].append(death_time - birth_time)
                        death_times[condition]['exposure'].append(
                            death_time - SPLIT_TIME)
    
    # Plot avg # of generations to death for each condition
    avg_times = []
    amp_concs = []
    for condition in death_times:
        avg_times.append(np.mean(death_times[condition]['exposure']))
        amp_concs.append(float(condition.split('Ampicillin (')[
            1].split(' mg/L)')[0]))
    avg_gens = np.array(avg_times) / AVG_GEN_TIME
    axs[0].scatter(amp_concs, avg_gens, c=(0, 0, 0))
    best_fit = np.polynomial.Polynomial.fit(amp_concs, avg_gens, 1, window=(
        min(amp_concs), max(amp_concs)))
    xx, yy = best_fit.linspace()
    axs[0].plot(xx, yy, c=(0, 0, 0))
    # Get amp. conc. for lysis in 1 generation
    liog = np.polynomial.polynomial.polyroots((best_fit-1).coef)[0]
    # LIOG from Boman and Ericksson 1963
    lit_liog = 6
    axs[0].hlines(1, 0, 8, linestyles='dashed', colors=[
        (0, 0, 0, 0.5)], linewidths=1)
    ylim = axs[0].get_ylim()
    axs[0].vlines([liog, lit_liog], 0, ylim[1], linestyles='dashed', colors=[
        (0, 0.4, 1, 1), (0, 0, 0, 1)], linewidths=[1, 1])
    axs[0].text(liog-0.4, ylim[1], 'This model', color=(0, 0.4, 1, 1), size=8,
        rotation=90, verticalalignment='top', horizontalalignment='center')
    axs[0].text(liog+0.6, ylim[1], 'MIC: 2 mg/L', color=(0, 0.4, 1, 1), size=8,
        rotation=90, verticalalignment='top', horizontalalignment='center')
    axs[0].text(lit_liog-0.4, ylim[1], 'Boman 1963', color=(0, 0, 0, 1), size=8,
        rotation=90, verticalalignment='top', horizontalalignment='center')
    axs[0].text(lit_liog+0.6, ylim[1], 'MIC: 4 mg/L', color=(0, 0, 0, 1), size=8,
        rotation=90, verticalalignment='top', horizontalalignment='center')
    axs[0].set_xlabel('Ampicillin (mg/L)', fontsize=9)
    axs[0].set_ylabel('Avg. generations to lysis', fontsize=8)
    axs[0].set_yticks([0, 1, 2, 3], [0, 1, 2, 3], fontsize=9)
    axs[0].set_xticks([0, 2, 4, 6, 8], [0, 2, 4, 6, 8], fontsize=9)
    sns.despine(ax=axs[0], offset=1, trim=True)

    # Plot histogram of cell age at time of death
    cmap = matplotlib.colormaps['Greys']
    antibiotic_min = data.loc[:, 'antibiotic_conc'].min()
    antibiotic_max = data.loc[:, 'antibiotic_conc'].max()
    norm = matplotlib.colors.Normalize(
        vmin=1.5*antibiotic_min-0.5*antibiotic_max, vmax=antibiotic_max)
    antibiotic_concs = data.loc[:, 'antibiotic_conc'].unique()
    palette = {antibiotic_conc: cmap(norm(antibiotic_conc))
        for antibiotic_conc in antibiotic_concs}
    palette[mic] = (0, 0.4, 1)
    death_df = {'External ampicillin (\u03BCM)': [], 'Age at death (min)': []}
    for condition, condition_data in death_times.items():
        death_df['External ampicillin (\u03BCM)'].extend(
            len(condition_data['age']) * [condition_data['amp']])
        death_df['Age at death (min)'].extend(np.array(condition_data[
            'age']) / 60)
    death_df = pd.DataFrame(death_df)
    sns.histplot(data=death_df, x='Age at death (min)',
        hue='External ampicillin (\u03BCM)', common_norm=False,
        stat='density', ax=axs[1], legend=False, palette=palette,
        multiple='stack', binwidth=5)
    axs[1].text(1, 1, f'Ampicillin (\u03BCM)',
        transform=axs[1].transAxes, size=8,
        horizontalalignment='right', verticalalignment='top')
    ypos = [0.9, 0.8, 0.7, 0.6, 0.5]
    concs = list(palette.keys())
    colors = list(palette.values())
    for i, y in enumerate(ypos):
        axs[1].text(1, y, concs[i], c=colors[i],
            transform=axs[1].transAxes, size=8,
            horizontalalignment='right', verticalalignment='top')
    axs[1].legend().remove()
    axs[1].set_xlabel('Age at death (min)', fontsize=9)
    axs[1].set_ylabel('Density', fontsize=9)
    axs[1].set_yticks([0, 0.1, 0.2, 0.3, 0.4],
        [0, 0.1, 0.2, 0.3, 0.4], fontsize=9)
    xticks = np.arange(0, 70, 20, dtype=int)
    axs[1].set_xticks(xticks, xticks, fontsize=9)
    sns.despine(ax=axs[1], offset=1, trim=True)
    plt.tight_layout()
    