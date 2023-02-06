import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import logit
from ete3 import NodeStyle, TreeNode, TreeStyle

from ecoli.analysis.antibiotics_colony import (COUNTS_PER_FL_TO_NANOMOLAR,
    restrict_data)
from ecoli.analysis.antibiotics_colony.timeseries import plot_tag_snapshots
from ecoli.library.cell_wall.hole_detection import detect_holes_skimage


def plot_exp_growth_rate(data, metadata, highlight_agent_id):
    grouped_agents = data.groupby(['Condition', 'Agent ID'])
    new_data = []
    for _, agent_data in grouped_agents:
        delta_t = np.diff(agent_data.loc[:, 'Time'], append=0)
        # Ignore cells for which less than 10 timepoints of data exist
        # to avoid outliers from instability in first few timesteps
        if len(delta_t) < 10:
            continue
        delta_t[-1] = delta_t[-2]
        dry_mass = agent_data.loc[:, 'Dry mass']
        mass_ratio = dry_mass[1:].to_numpy() / dry_mass[:-1].to_numpy()
        mass_ratio = np.append(mass_ratio, mass_ratio[-1])
        agent_data['Doubling rate'] = np.log2(mass_ratio) / delta_t * 3600
        agent_data['active_ribo_concs'] = (agent_data.loc[
            :, 'Active ribosomes'] / agent_data.loc[:, 'Volume'] *
            COUNTS_PER_FL_TO_NANOMOLAR / 1000)
        agent_data['active_rnap_concs'] = (agent_data.loc[
            :, 'Active RNAP'] / agent_data.loc[:, 'Volume'] *
            COUNTS_PER_FL_TO_NANOMOLAR / 1000)
        agent_data['tet_concs'] = np.round(
            agent_data.loc[:, 'Initial external tet.'] * 1000, 3)
        new_data.append(agent_data)

    data = pd.concat(new_data)
    cmap = matplotlib.colormaps['Greys']
    tet_min = data.loc[:, 'tet_concs'].min()
    tet_max = data.loc[:, 'tet_concs'].max()
    norm = matplotlib.colors.Normalize(
        vmin=1.5*tet_min-0.5*tet_max, vmax=tet_max)
    tet_concs = data.loc[:, 'tet_concs'].unique()
    palette = {tet_conc: cmap(norm(tet_conc)) for tet_conc in tet_concs}
    palette[3.375] = (0, 0.4, 1)
    
    time_boundaries = [(11550, 11550+3600), (26000-3600, 26002)]
    cols_to_plot = ['active_ribo_concs', 'active_rnap_concs', 
        'Doubling rate', 'tet_concs', 'Agent ID', 'Condition']
    ylim = (0, 1.45)
    xlim = (4, 26)
    for i, boundaries in enumerate(time_boundaries):
        time_filter = ((data.loc[:, 'Time'] >= boundaries[0])
            & (data.loc[:, 'Time'] < boundaries[1]))
        filtered_data = data.loc[time_filter, cols_to_plot]
        mean_data = filtered_data.groupby(['Condition', 'Agent ID']).mean()
        joint = sns.jointplot(data=mean_data, x='active_ribo_concs',
            y='Doubling rate', hue='tet_concs', palette=palette, marginal_kws={
                'common_norm': False}, joint_kws={'edgecolors': 'face'})
        ax = joint.ax_joint
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        sns.despine(offset=0.1, trim=True, ax=ax)
        sns.despine(trim=True, ax=joint.ax_marg_x, left=True)
        sns.despine(trim=True, ax=joint.ax_marg_y, bottom=True)
        ax.set_xlabel('Active ribosomes (mM)', size=8)
        xticks = [5, 10, 15, 20, 25]
        ax.set_xticks(xticks, xticks, size=8)
        ax.legend().remove()
        if i == 0:
            ax.text(0.1, 0.9, 'Tet. (\u03BCM)', size=8, transform=ax.transAxes)
            for conc_idx, (conc, color) in enumerate(palette.items()):
                ax.text(0.1, 0.8-0.1*conc_idx, conc, size=8,
                    transform=ax.transAxes, c=color)
            ax.set_ylabel('Doubling rate (1/hr)', size=8)
            yticks = np.round(ax.get_yticks(), 1)
            ax.set_yticks(yticks, yticks, size=8)
            joint.ax_marg_x.set_title(r'$1^{\mathrm{st}}$ hr. post-tet.',
                size=8, pad=2, weight='bold')
        else:
            sns.despine(ax=ax, left=True)
            ax.yaxis.set_visible(False)
            joint.ax_marg_x.set_title(r'$4^{\mathrm{th}}$ hr. post-tet.',
                size=8, pad=2, weight='bold')
        joint.figure.set_size_inches(2.35, 2)
        plt.savefig(f'out/analysis/paper_figures/growth_rate_var_ribo_{i}.svg')
        plt.close()

    # Get log 2 fold change over mean glucose growth rate
    glucose_data = data.loc[data.loc[:, 'Condition'] == 'Glucose', :]
    mean_growth_rate = glucose_data.loc[:, 'Doubling rate'].mean()
    fc_col = 'Growth rate\n($\mathregular{log_2}$ fold change)'
    data.loc[:, fc_col] = np.log2(data.loc[:, 'Doubling rate'] / mean_growth_rate)

    # Only include data from glucose and tetracycline MIC
    data = data.loc[np.isin(data.loc[:, 'Condition'],
        ['Glucose', 'Tetracycline (1.5 mg/L)']), :]

    # Set up custom divergent colormap
    highlight_agent_ids = [highlight_agent_id[:i+1] for i  in range(len(highlight_agent_id))]
    highlight_agent = {agent_id: {
        'membrane_width': 0.5, 'membrane_color': (0, 0.4, 1)}
        for agent_id in highlight_agent_ids}
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        'divergent', [(0.678, 0, 0.125), (1, 1, 1), (0, 0, 0)])
    norm = matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5)
    fig = plot_tag_snapshots(
        data=data, metadata=metadata, tag_colors={fc_col: {
            'cmp': cmp, 'norm': norm}}, snapshot_times=np.array([
            3.2, 4.5, 5.8, 7.1]) * 3600, show_membrane=True,
        return_fig=True, figsize=(6, 1.5), highlight_agent=highlight_agent)
    fig.axes[0].set_xticklabels(
        np.abs(np.round(fig.axes[0].get_xticks()/3600 - 11550/3600, 1)))
    fig.axes[0].set_xlabel('Hours after tetracycline addition')
    fig.savefig('out/analysis/paper_figures/fig_3b_tet_snapshots.svg',
        bbox_inches='tight')
    plt.close()


def plot_lattice_timeseries():
    with open('data/lattice_df.pkl', 'rb') as f:
        lattice_data = pickle.load(f)
    snapshot_times = [0, 400, 600, 1200]
    fig, axs = plt.subplots(1, 4)
    for i, ax in enumerate(axs.flat):
        lattice = np.array(lattice_data[snapshot_times[i]]['agents']['0'][
            'wall_state']['lattice'])
        hole_sizes, hole_view = detect_holes_skimage(lattice)
        biggest_hole = np.argmax(hole_sizes) + 1
        lattice[hole_view==biggest_hole] = 2
        ax.imshow(lattice, interpolation='nearest', aspect='auto')
        ax.set_title(hole_sizes[biggest_hole-1])


def make_ete_trees(agent_ids):
    stem = os.path.commonprefix(list(agent_ids))
    id_node_map = dict()
    sorted_agents = sorted(agent_ids)
    roots = []
    for agent_id in sorted_agents:
        phylogeny_id = agent_id[len(stem):]
        parent_phylo_id = phylogeny_id[:-1]
        if parent_phylo_id in id_node_map:
            parent = id_node_map[parent_phylo_id]
            child = parent.add_child(name=agent_id)
        else:
            child = TreeNode(name=agent_id)
            roots.append(child)
        id_node_map[phylogeny_id] = child
    return roots


def plot_ampc_phylo(data):
    data = restrict_data(data)
    agent_ids = data.loc[:, 'Agent ID'].unique().tolist()
    final_agents = data.loc[data.loc[:, 'Time'] == 26000, 'Agent ID'].unique()
    dead_agents = [agent_id for agent_id in agent_ids
        if (agent_id + '0' not in agent_ids) and (agent_id not in final_agents)]
    trees = make_ete_trees(agent_ids)
    assert len(trees) == 1
    tree = trees[0]

    # Set style for overall figure
    tstyle = TreeStyle()
    tstyle.show_scale = False
    tstyle.show_leaf_name = False
    tstyle.scale = None
    tstyle.optimal_scale_level = 'full'
    tstyle.mode = 'c'

    # Color nodes by AmpC concentration
    data['AmpC conc'] = data.loc[:, 'AmpC monomer'] / (
        data.loc[:, 'Volume'] * 0.2) * COUNTS_PER_FL_TO_NANOMOLAR
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        'blue', [(0, 0, 0), (0, 0.4, 1), (1, 1, 1)])
    ampc_concs = data[['AmpC conc', 'Agent ID']].groupby(
        'Agent ID').mean().to_numpy()
    min_conc = ampc_concs.min()
    max_conc = ampc_concs.max()
    # norm = matplotlib.colors.LogNorm(vmin=min_conc, vmax=max_conc)
    norm = matplotlib.colors.Normalize(vmin=min_conc, vmax=max_conc)
    agent_data = data.groupby('Agent ID').mean()
    # Set styles for each node
    for node in tree.traverse():
        nstyle=NodeStyle()
        nstyle['size'] = 20
        nstyle['vt_line_width'] = 3
        nstyle['hz_line_width'] = 3
        nstyle['fgcolor'] = matplotlib.colors.to_hex(
            cmp(norm(agent_data.loc[node.name, 'AmpC conc'])))
        if node.name in dead_agents:
            nstyle['bgcolor'] = 'Gainsboro'
        node.set_style(nstyle)
    tstyle.scale = 10
    tree.render('out/analysis/paper_figures/ampc_phylo.svg', tree_style=tstyle,
        units='in', h=1.5, w=1.5)
    fig, ax = plt.subplots(figsize=(2, 0.25))
    fig.subplots_adjust(bottom=0.6)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmp), cax=ax,
        orientation='horizontal', label='AmpC (periplasm, nM)')
    xticks = [int(np.round(min_conc, 1)), int(np.round(max_conc, 1))]
    ax.set_xticks(xticks, xticks, size=8)
    ax.set_xlabel(ax.get_xlabel(), size=8, labelpad=-7)
    fig.savefig('out/analysis/paper_figures/ampc_cbar.svg')

    # Export Newick file for phylogenetic signal analysis
    tree.write(outfile='out/analysis/paper_figures/amp_tree.nw')
    agent_data.loc[tree.get_leaf_names(), :].to_csv(
        'out/analysis/paper_figures/agent_data.csv')
