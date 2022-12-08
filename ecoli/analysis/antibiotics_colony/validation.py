from typing import List, Dict
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ecoli.analysis.antibiotics_colony import (
    MAX_TIME, SPLIT_TIME, DE_GENES, restrict_data)

def plot_colony_growth_rates(
    data: pd.DataFrame,
    ax: plt.Axes = None,
) -> None:
    '''Plot traces of total colony mass and total colony growth rate.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', and
            'Dry Mass'. The first experimental condition in the 'Condition'
            column is treated as a control (default gray). Include at most
            2 conditions and 1 seed per condition.
        axes: Instance of Matplotlib Axes to plot on
    '''
    columns_to_include = ['Dry mass', 'Condition', 'Time']
    data = data.loc[:, columns_to_include]
    mask = (data.loc[:, 'Condition']=='Glucose') & (
        data.loc[:, 'Time'] > SPLIT_TIME) & (
        data.loc[:, 'Time'] <= MAX_TIME)
    remaining_glucose_data = data.loc[mask, :]
    data = restrict_data(data)
    data = pd.concat([data, remaining_glucose_data])
    data = data.groupby(['Condition', 'Time']).sum().reset_index()
    
    # Convert time to hours
    data.loc[:, 'Time'] = data.loc[:, 'Time'] / 3600

    sns.lineplot(
        data=data, x='Time', y='Dry mass',
        hue='Condition', ax=ax, palette='viridis', errorbar=None)
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
    sns.despine(ax=ax, offset=3, trim=True)


def plot_synth_prob_fc(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    genes_to_plot: List[str] = None,
) -> None:
    '''Plot scatter plot of simulated and experimental log2 fold change for
    synthesis probabilities of key genes regulated during tetracycline exposure.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', 'Seed',
            and 'Dry Mass'. The first experimental condition in the 'Condition'
            column is treated as a control.
        ax: Single instance of Matplotlib Axes to plot on.
        genes_to_plot: List of gene names to include in plot.
    '''
    data = data.loc[data.loc[:, 'Time'] <= MAX_TIME, :]
    mrna_ids = [f'{gene} synth prob' for gene in genes_to_plot
        if gene not in ['MicF', 'ompF']] + ['GAPDH synth prob']
    mrna_data = data.loc[:, mrna_ids + ['Condition']]
    conditions = mrna_data.loc[:, 'Condition'].unique()
    # Get average synthesis probability per condition
    mrna_data = mrna_data.groupby(['Condition']).mean()
    # Convert to relative amounts assuming that first condition is control
    relative_data = (mrna_data.loc[conditions[1]] /
        mrna_data.loc[conditions[0]])
    relative_data.index = [mrna_name.split(' synth prob')[0] for mrna_name in relative_data.index]
    relative_data = pd.DataFrame(relative_data).rename(columns={0: 'Simulated'})
    # Compare against literature relative amounts
    tet_degenes = DE_GENES.loc[:, ['Gene name', 'Fold change']].set_index(
        'Gene name').drop(['MicF', 'ompF'], axis=0).rename(columns={
            'Fold change': 'Viveiros et al. 2007'})
    # tet_degenes.loc[:, 'Viveiros et al. 2007'] = np.log2(
    #     tet_degenes.loc[:, 'Viveiros et al. 2007'])
    relative_data = relative_data.join(tet_degenes, how='inner').reset_index()
    relative_data = relative_data.rename(columns={'index': 'Gene'})

    sns.scatterplot(data=relative_data, x='Viveiros et al. 2007',
        y='Simulated', ax=ax)
    min_fc = relative_data.loc[:, 'Viveiros et al. 2007'].min()
    max_fc = relative_data.loc[:, 'Viveiros et al. 2007'].max()
    validation_line = np.linspace(min_fc, max_fc, 2)
    ax.plot(validation_line, validation_line, '--')
    sns.despine(ax=ax, offset=3, trim=True)


def plot_mrna_fc(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    genes_to_plot: List[str] = None,
) -> None:
    '''Plot scatter plot of simulated and experimental log2 fold change for
    final mRNA concentrations of key genes regulated during tetracycline exposure.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', and 'Dry Mass'.
            The first experimental condition in the 'Condition' column is treated as
            a control.
        genes_to_plot: List of gene names to include in plot.
    '''
    data = data.loc[data.loc[:, 'Time'] <= MAX_TIME, :]
    mrna_ids = [f'{gene} mRNA' for gene in genes_to_plot
        if gene not in ['MicF', 'ompF']] + ['GAPDH mRNA']
    mrna_data = data.loc[:, mrna_ids + ['Condition', 'Volume']]
    conditions = mrna_data.loc[:, 'Condition'].unique()
    # Get mRNA concentrations aggregated over entire final colonies
    mrna_data = mrna_data.set_index(['Condition'])
    mrna_data = mrna_data.divide(mrna_data.loc[:, 'Volume'], axis=0).drop(
        ['Volume'], axis=1).reset_index()
    mrna_data = mrna_data.groupby(['Condition']).sum()
    # Normalize by housekeeping gene expression
    mrna_data = mrna_data.divide(mrna_data.loc[:, 'GAPDH mRNA'], axis=0)
    # Convert to relative amounts assuming that first condition is control
    relative_data = pd.DataFrame(np.log2(mrna_data.loc[conditions[1]] /
            mrna_data.loc[conditions[0]]))
    relative_data.index = [mrna_name.split(' mRNA')[0] for mrna_name in relative_data.index]
    relative_data = relative_data.rename(columns={0: 'Simulated'})
    # Compare against literature relative amounts
    tet_degenes = DE_GENES.loc[:, ['Gene name', 'Fold change']].set_index(
        'Gene name').drop(['MicF', 'ompF'], axis=0).rename(columns={
            'Fold change': 'Viveiros et al. 2007'})
    tet_degenes.loc[:, 'Viveiros et al. 2007'] = np.log2(
        tet_degenes.loc[:, 'Viveiros et al. 2007'])
    relative_data = relative_data.join(tet_degenes, how='inner').reset_index()
    relative_data = relative_data.rename(columns={'index': 'Gene'})

    sns.scatterplot(data=relative_data, x='Viveiros et al. 2007',
        y='Simulated', ax=ax)
    min_fc = relative_data.loc[:, 'Viveiros et al. 2007'].min()
    max_fc = relative_data.loc[:, 'Viveiros et al. 2007'].max()
    validation_line = np.linspace(min_fc, max_fc, 2)
    ax.plot(validation_line, validation_line, '--')
    sns.despine(ax=ax, offset=3, trim=True)


def plot_vs_distance_from_center(
    data: pd.DataFrame,
    bounds: tuple = None,
    ax: plt.Axes = None,
    column_to_plot: str = None,
):
    '''Plot scatter plot of a given column vs distance from environment center.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Supply at most 1 condition. Multiple seeds are OK.
        bounds: Tuple representing width and height of environment.
        axes: Single instance of Matplotlib Axes to plot on.
        column_to_plot: Column to plot against distance from center.
    '''
    data = data.loc[data.loc[:, column_to_plot] > 0, :]
    if len(data) == 0:
        return
    center = np.array(bounds) / 2
    locations = np.array(data.loc[:, 'Boundary'].apply(
        lambda x: x['location']).tolist())
    data['Distance'] = np.linalg.norm(locations-center, axis=1)

    sns.regplot(data=data, x='Distance', y=column_to_plot, ax=ax)


def plot_protein_synth_inhib(
    data: pd.DataFrame,
    ax: plt.Axes = None,
):
    '''Plot scatter plot of normalized % protein synthesis inhibition across a 
    variety of tetracycline concentrations.

    Args:
        data: DataFrame where each row is an agent and each column is a variable
            of interest. Must have these columns: 'Time', 'Condition', 'Seed',
            and 'Dry Mass'. The first experimental condition in the 'Condition'
            column is treated as a control.
        ax: Single instance of Matplotlib Axes to plot on.
    '''
    # Normalize by ribosome count and total mRNA count so synthesis rates are
    # comparable with those from in vitro transcription/translation systems,
    # where these counts should be nearly constant over the course of the assay
    data = data.sort_values(by=['Condition', 'Seed', 'Agent ID', 'Time'])
    data['Total ribosomes'] = (data.loc[:, 'Active ribosomes'] +
        data.loc[:, 'Inactive 30S subunit'])
    data.loc[:, 'Cytoplasmic tetracycline'] *= 1000
    data['Protein delta'] = np.append([0], np.diff(data.loc[:, 'Protein mass']))
    data['Normed delta'] = data.loc[:, 'Protein delta'] / data.loc[
        :, 'Total ribosomes'] / data.loc[:, 'Total mRNA']
    columns_to_include = ['Normed delta', 'Cytoplasmic tetracycline',
        'Condition', 'Seed', 'Agent ID']
    normed_deltas = data.loc[:, columns_to_include].groupby([
        'Condition', 'Seed', 'Agent ID']).agg(
        lambda x: np.mean(x[1:])).reset_index()
    normed_deltas = normed_deltas.groupby(['Condition']).mean()
    normed_deltas['Percent inhibition'] = 1 - (normed_deltas.loc[
        :, 'Normed delta'] / normed_deltas.loc['Glucose', 'Normed delta'])
    sns.scatterplot(data=normed_deltas, x='Cytoplasmic tetracycline',
        y='Percent inhibition', ax=ax)
    ax.set_xscale('log')
    sns.despine(ax=ax, offset=3, trim=True)


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

