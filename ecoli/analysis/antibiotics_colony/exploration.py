import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ete3 import CircleFace, NodeStyle, TextFace, TreeNode, TreeStyle, faces

from ecoli.analysis.antibiotics_colony import (COUNTS_PER_FL_TO_NANOMOLAR,
    restrict_data)
from ecoli.analysis.antibiotics_colony.timeseries import plot_tag_snapshots
from ecoli.library.cell_wall.hole_detection import detect_holes_skimage
from ecoli.library.sim_data import LoadSimData


def plot_exp_growth_rate(data, metadata):
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
    
    time_boundaries = np.linspace(11550, 26000, 4)
    time_boundaries[-1] = 26002
    cols_to_plot = ['active_ribo_concs', 'active_rnap_concs', 
        'Doubling rate', 'tet_concs', 'Agent ID', 'Condition']
    ylim = (0, 1.41)
    xlim = (4, 26)
    for i in range(3):
        time_filter = ((data.loc[:, 'Time'] >= time_boundaries[i]) &
            (data.loc[:, 'Time'] < time_boundaries[i+1]))
        filtered_data = data.loc[time_filter, cols_to_plot]
        mean_data = filtered_data.groupby(['Condition', 'Agent ID']).mean()
        joint = sns.jointplot(data=mean_data, x='active_ribo_concs',
            y='Doubling rate', hue='tet_concs', palette=palette, marginal_kws={
                'common_norm': False}, joint_kws={'edgecolors': 'face'}, height=4)
        joint.ax_joint.set_ylim(ylim)
        joint.ax_joint.set_xlim(xlim)
        if i == 0:
            sns.despine(offset=0.1, trim=True, ax=joint.ax_joint)
            sns.despine(trim=True, ax=joint.ax_marg_x, left=True)
            sns.despine(trim=True, ax=joint.ax_marg_y, bottom=True)
            legend = joint.ax_joint.legend(frameon=False, loc='lower left')
            legend.set_title('Tetracycline (uM)')
            joint.ax_joint.set_xlabel('Active ribosomes (mM)')
            joint.ax_joint.set_ylabel('Doubling rate (1/hr)')
        else:
            sns.despine(offset=0.1, left=True, trim=True, ax=joint.ax_joint)
            sns.despine(trim=True, ax=joint.ax_marg_x, left=True)
            sns.despine(trim=True, ax=joint.ax_marg_y, bottom=True)
            joint.ax_joint.set_xlabel('Active ribosomes (mM)')
            joint.ax_joint.yaxis.set_visible(False)
            joint.ax_joint.legend().remove()
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
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        'divergent', [(0, 0.4, 1), (1, 1, 1), (0.678, 0, 0.125)])
    norm = matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5)
    plot_tag_snapshots(
        data=data, metadata=metadata, tag_colors={fc_col: {
            'cmp': cmp, 'norm': norm}}, snapshot_times=np.array([
            1.9, 3.2, 4.5, 5.8, 7.1]) * 3600, show_membrane=True)


def plot_mara_micf_reg():
    return


def plot_mara_effect(data):
    return

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
    # n_generations = max([len(agent_id) for agent_id in agent_ids])
    final_agents = [agent_id for agent_id in agent_ids
        if agent_id + '0' not in agent_ids]
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
        data.loc[:, 'Volume'] * 0.2)
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        'blue', [(0, 0, 0), (0, 0.4, 1)])
    ampc_concs = data[['AmpC conc', 'Agent ID']].groupby(
        'Agent ID').mean().to_numpy()
    min_conc = ampc_concs.min()
    max_conc = ampc_concs.max()
    norm = matplotlib.colors.LogNorm(vmin=min_conc, vmax=max_conc)
    agent_data = data.groupby('Agent ID').mean()
    # Set styles for each node
    for node in tree.traverse():
        nstyle=NodeStyle()
        nstyle['size'] = 10
        nstyle['vt_line_width'] = 1
        nstyle['hz_line_width'] = 1
        nstyle['fgcolor'] = matplotlib.colors.to_hex(
            cmp(norm(agent_data.loc[node.name, 'AmpC conc'])))
        node.set_style(nstyle)
    tree.render('out/analysis/paper_figures/ampc_phylo.svg', tree_style=tstyle, w=400)
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(bottom=0.3)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmp), cax=ax,
        orientation='horizontal', label='AmpC (periplasm, uM)')
    fig.text(0, 0.4, int(np.round(min_conc, 1)))
    fig.text(0.92, 0.4, int(np.round(max_conc, 1)))
    plt.tight_layout()
    fig.savefig('out/analysis/paper_figures/ampc_cbar.svg')

    # Color nodes by cell wall relative porosity
    # Get fold change over average glucose porosity
    data['Relative porosity'] = data.loc[:, 'Porosity'] * data.loc[:, 'Extension factor']
    mean_glc_porosity = data.loc[data.loc[:, 'Condition'] == 'Glucose', 
        'Relative porosity'].mean()
    fc_col = 'Porosity\n($\mathregular{log_2}$ fold change)'
    data.loc[:, fc_col] = np.log2(data.loc[:, 'Relative porosity'] / mean_glc_porosity)
    data.loc[data.loc[:, fc_col]==-np.inf, fc_col] = 0
    # Set up custom divergent colormap
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        'divergent', [(0, 0.4, 1), (0, 0, 0), (0.678, 0, 0.125)])
    magnitude = data.loc[:, fc_col].abs().max()
    norm = matplotlib.colors.Normalize(vmin=-magnitude, vmax=magnitude)
    agent_data = data.groupby('Agent ID').mean()
    # Set styles for each node
    for node in tree.traverse():
        nstyle=NodeStyle()
        nstyle['size'] = 10
        nstyle['vt_line_width'] = 1
        nstyle['hz_line_width'] = 1
        nstyle['fgcolor'] = matplotlib.colors.to_hex(
            cmp(norm(agent_data.loc[node.name, fc_col])))
        node.set_style(nstyle)
    tstyle._scale = None
    tree.render('out/analysis/paper_figures/porosity_phylo.svg', tree_style=tstyle, w=400)
    fig, ax = plt.subplots(figsize=(4, 1))
    fig.subplots_adjust(bottom=0.3)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmp), cax=ax,
        orientation='horizontal', label='Relative Porosity ($\mathregular{log_2 FC}$)')
    plt.tight_layout()
    fig.savefig('out/analysis/paper_figures/porosity_cbar.svg')
