import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Any

from ecoli.analysis.antibiotics_colony import COUNTS_PER_FL_TO_NANOMOLAR, restrict_data
from ecoli.analysis.antibiotics_colony.timeseries import plot_tag_snapshots


def plot_exp_growth_rate(data, metadata, highlight_agent_id):
    # Create two scatterplots of average doubling rate
    # against active ribosome concentration for cells in the first
    # and fourth hours of tetracycline exposure (3M)
    grouped_agents = data.groupby(["Condition", "Agent ID"])
    new_data = []
    for _, agent_data in grouped_agents:
        delta_t = np.diff(agent_data.loc[:, "Time"], append=0)
        # Ignore cells for which less than 10 timepoints of data exist
        # to avoid outliers from instability in first few timesteps
        if len(delta_t) < 10:
            continue
        delta_t[-1] = delta_t[-2]
        dry_mass = agent_data.loc[:, "Dry mass"]
        mass_ratio = dry_mass[1:].to_numpy() / dry_mass[:-1].to_numpy()
        mass_ratio = np.append(mass_ratio, mass_ratio[-1])
        agent_data["Doubling rate"] = np.log2(mass_ratio) / delta_t * 3600
        agent_data["active_ribo_concs"] = (
            agent_data.loc[:, "Active ribosomes"]
            / agent_data.loc[:, "Volume"]
            * COUNTS_PER_FL_TO_NANOMOLAR
            / 1000
        )
        agent_data["active_rnap_concs"] = (
            agent_data.loc[:, "Active RNAP"]
            / agent_data.loc[:, "Volume"]
            * COUNTS_PER_FL_TO_NANOMOLAR
            / 1000
        )
        agent_data["tet_concs"] = np.round(
            agent_data.loc[:, "Initial external tet."] * 1000, 3
        )
        new_data.append(agent_data)

    data = pd.concat(new_data)
    cmap = matplotlib.colormaps["Greys"]
    tet_min = data.loc[:, "tet_concs"].min()
    tet_max = data.loc[:, "tet_concs"].max()
    norm = matplotlib.colors.Normalize(vmin=1.5 * tet_min - 0.5 * tet_max, vmax=tet_max)
    tet_concs = data.loc[:, "tet_concs"].unique()
    palette = {tet_conc: cmap(norm(tet_conc)) for tet_conc in tet_concs}
    palette[3.375] = (0, 0.4, 1)

    time_boundaries = [(11550, 11550 + 3600), (26000 - 3600, 26002)]
    cols_to_plot = [
        "active_ribo_concs",
        "active_rnap_concs",
        "Doubling rate",
        "tet_concs",
        "Agent ID",
        "Condition",
    ]
    ylim = (0, 1.45)
    xlim = (4, 26)
    for i, boundaries in enumerate(time_boundaries):
        time_filter = (data.loc[:, "Time"] >= boundaries[0]) & (
            data.loc[:, "Time"] < boundaries[1]
        )
        filtered_data = data.loc[time_filter, cols_to_plot]
        mean_data = filtered_data.groupby(["Condition", "Agent ID"]).mean()
        joint = sns.jointplot(
            data=mean_data,
            x="active_ribo_concs",
            y="Doubling rate",
            hue="tet_concs",
            palette=palette,
            marginal_kws={"common_norm": False},
            joint_kws={"edgecolors": "face"},
        )
        ax = joint.ax_joint
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        sns.despine(offset=0.1, trim=True, ax=ax)
        sns.despine(trim=True, ax=joint.ax_marg_x, left=True)
        sns.despine(trim=True, ax=joint.ax_marg_y, bottom=True)
        ax.set_xlabel("Active ribosomes (mM)", size=9)
        xticks = [5, 10, 15, 20, 25]
        ax.set_xticks(xticks, xticks, size=8)
        ax.legend().remove()
        if i == 0:
            ax.text(0.1, 0.9, "Tet. (\u03bcM)", size=8, transform=ax.transAxes)
            for conc_idx, (conc, color) in enumerate(palette.items()):
                ax.text(
                    0.1,
                    0.8 - 0.1 * conc_idx,
                    conc,
                    size=8,
                    transform=ax.transAxes,
                    c=color,
                )
            ax.set_ylabel("Doubling rate (1/hr)", size=9)
            yticks = np.round(ax.get_yticks(), 1)
            ax.set_yticks(yticks, yticks, size=8)
            joint.ax_marg_x.set_title(
                r"$1^{\mathrm{st}}$ hr. post-tet.", size=9, pad=2, weight="bold"
            )
        else:
            sns.despine(ax=ax, left=True)
            ax.yaxis.set_visible(False)
            joint.ax_marg_x.set_title(
                r"$4^{\mathrm{th}}$ hr. post-tet.", size=9, pad=2, weight="bold"
            )
        joint.figure.set_size_inches(2.5, 2)
        plt.savefig(
            "out/analysis/paper_figures/" + f"fig_3m_growth_rate_var_ribo_{i}.svg"
        )
        plt.close()
    print("Done with Figure 3M.")

    # Plot snapshots with intervals of 1.3 hours of decrease in
    # instantaneous growth rate after tetracycline addition (3D)
    # Get log 2 fold change over mean glucose growth rate
    glucose_data = data.loc[data.loc[:, "Condition"] == "Glucose", :]
    mean_growth_rate = glucose_data.loc[:, "Doubling rate"].mean()
    fc_col = "Growth rate\n($\\mathregular{log_2}$ fold change)"
    data.loc[:, fc_col] = np.log2(data.loc[:, "Doubling rate"] / mean_growth_rate)

    # Only include data from glucose and tetracycline MIC
    data = data.loc[
        np.isin(data.loc[:, "Condition"], ["Glucose", "Tetracycline (1.5 mg/L)"]), :
    ]

    # Set up custom divergent colormap
    highlight_agent_ids = [
        highlight_agent_id[: i + 1] for i in range(len(highlight_agent_id))
    ]
    highlight_agent = {
        agent_id: {"membrane_width": 0.5, "membrane_color": (0, 0.4, 1)}
        for agent_id in highlight_agent_ids
    }
    cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
        "divergent", [(0.678, 0, 0.125), (1, 1, 1), (0, 0, 0)]
    )
    norm = matplotlib.colors.Normalize(vmin=-2.5, vmax=2.5)
    fig = plot_tag_snapshots(
        data=data,
        metadata=metadata,
        tag_colors={fc_col: {"cmp": cmp, "norm": norm}},
        snapshot_times=np.array([3.2, 4.5, 5.8, 7.1]) * 3600,
        show_membrane=True,
        return_fig=True,
        figsize=(6, 1.5),
        highlight_agent=highlight_agent,
    )
    fig.axes[0].set_xticklabels(
        np.abs(np.round(fig.axes[0].get_xticks() / 3600 - 11550 / 3600, 1))
    )
    fig.axes[0].set_xlabel("Hours after tetracycline addition")
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig(
        "out/analysis/paper_figures/fig_3b_tet_snapshots.svg", bbox_inches="tight"
    )
    plt.close()

    # Print ratio of final instantaneous doubling rate in tet. sim. vs glc. sim.
    data = restrict_data(data)
    final_ratios = data.loc[data.Time == 26000, "Doubling rate"] / mean_growth_rate
    print(
        f"Mean doubling rate at t=26000 of tet. vs glc. sim.: {final_ratios.mean() * 100}%"
    )
    print(
        f"Std. dev. doubling rate at t=26000 of tet. vs glc. sim.: {final_ratios.std() * 100}%"
    )


def build_tree_structure(agent_ids: list[str]) -> tuple[dict, dict]:
    """
    Build a tree structure from agent IDs.

    Args:
        agent_ids: List of agent IDs

    Returns:
        2-element tuple containing

        - **tree**: Tree structure as nested dictionary
        - **node_depths**: Mapping of phylogeny IDs to their depth and full agent ID
    """
    stem = os.path.commonprefix(list(agent_ids))
    tree: dict[str, Any] = {}
    node_depths = {}

    # Build tree structure
    for agent_id in sorted(agent_ids):
        phylogeny_id = agent_id[len(stem) :]

        # Add to tree structure
        current = tree
        depth = 0
        for i in range(len(phylogeny_id)):
            prefix = phylogeny_id[: i + 1]
            if prefix not in current:
                current[prefix] = {}
            current = current[prefix]
            depth = i + 1

        # Store the full agent ID with its depth
        node_depths[phylogeny_id] = {"depth": depth, "name": agent_id}

    return tree, node_depths


def count_leaves(tree_part: dict, leaf_counts: dict, prefix: str = "") -> int:
    """
    Count the number of leaves for all nodes in the tree. Mutates ``leaf_counts``.

    Args:
        tree_part: Tree structure
        leaf_counts: Dictionary to store leaf counts
        prefix: Current node prefix
    """
    if not tree_part:  # Leaf node
        leaf_counts[prefix] = 1
        return 1

    count = 0
    for key, subtree in tree_part.items():
        count += count_leaves(subtree, leaf_counts, key)

    leaf_counts[prefix] = count
    return count


def calculate_positions(
    tree_part: dict,
    start_angle: float,
    angle_range: float,
    node_positions: dict[str, tuple[float, float]],
    leaf_counts: dict[str, int],
    max_depth: int,
    prefix: str = "",
    depth: int = 0,
):
    """
    Calculate r and theta positions for each node in the tree. Mutates ``node_positions``.

    Args:
        tree_part: Tree structure
        start_angle: Starting angle for this node
        angle_range: Angle range for this node
        node_positions: Dictionary to store node positions
        leaf_counts: Dictionary of leaf counts
        max_depth: Maximum depth of the tree
        prefix: Current node prefix
        depth: Current depth in the tree
    """
    # Position for this node
    node_positions[prefix] = (depth / max_depth, start_angle + angle_range / 2)

    if not tree_part:  # Leaf node
        return

    current_angle = start_angle
    sorted_keys = sorted(tree_part.keys())

    for key in sorted_keys:
        subtree = tree_part[key]
        # Calculate angle allocation based on proportion of leaves
        leaf_count = leaf_counts.get(key, 1)
        total_leaves = leaf_counts.get(prefix, 1) if prefix else leaf_counts.get("", 1)
        angle_allocation = angle_range * (leaf_count / total_leaves)

        # Recurse for children
        calculate_positions(
            subtree,
            current_angle,
            angle_allocation,
            node_positions,
            leaf_counts,
            max_depth,
            key,
            depth + 1,
        )
        current_angle += angle_allocation


def create_newick_string(tree, node_depths, node_id=""):
    """
    Create a Newick format string representation of the tree.

    Args:
        tree (dict): Tree structure
        node_depths (dict): Node depth information
        node_id (str): Current node ID

    Returns:
        str: Newick format string
    """
    if not tree:
        return node_depths.get(node_id, {}).get("name", node_id) + ":1"

    parts = []
    for key, subtree in sorted(tree.items()):
        parts.append(create_newick_string(subtree, node_depths, key))

    result = "(" + ",".join(parts) + ")"
    result += node_depths.get(node_id, {}).get("name", node_id) + ":1"

    return result


def plot_ampc_phylo(data):
    """
    Create a circular phylogenetic tree with AmpC concentrations using Matplotlib with polar projection.
    Children are placed equidistant from their parent's angle.

    Args:
        data (pd.DataFrame): Input data containing agent information.
    """
    agent_ids = data.loc[:, "Agent ID"].unique().tolist()
    final_agents = data.loc[data.loc[:, "Time"] == 26000, "Agent ID"].unique()
    dead_agents = [
        agent_id
        for agent_id in agent_ids
        if (agent_id + "0" not in agent_ids) and (agent_id not in final_agents)
    ]

    # Color nodes by AmpC concentration
    data["AmpC conc"] = data.loc[:, "AmpC monomer"] / (data.loc[:, "Volume"] * 0.2)
    agent_data = data.groupby("Agent ID").mean()

    # Set up colormap and normalization
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", [(0, 0, 0), (0, 0.4, 1), (1, 1, 1)]
    )
    min_conc = agent_data["AmpC conc"].min()
    max_conc = agent_data["AmpC conc"].max()
    norm = matplotlib.colors.Normalize(vmin=min_conc, vmax=max_conc)

    # Build tree structure
    tree, node_depths = build_tree_structure(agent_ids)

    # Count leaf nodes to determine angle allocation
    leaf_counts = {}
    count_leaves(tree, leaf_counts)
    # Assign angles based on leaf counts
    node_positions = {}
    max_depth = max(info["depth"] for info in node_depths.values())
    calculate_positions(tree, 0, 2 * np.pi, node_positions, leaf_counts, max_depth)

    # Create figure with polar projection
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.axis("off")

    # Map node_id to their children
    children_map = {}
    for node_id in node_positions:
        if len(node_id) > 0:
            parent_id = node_id[:-1]
            if parent_id not in children_map:
                children_map[parent_id] = []
            children_map[parent_id].append(node_id)

    # Draw connections: stem -> arc -> stems
    stem_length = 0.1  # Length of the radial stems

    for parent_id, children in children_map.items():
        if parent_id not in node_positions:
            continue

        # Get parent position
        parent_r, parent_theta = node_positions[parent_id]

        # Draw parent stem (radially outward)
        outer_r = parent_r + stem_length
        ax.plot(
            [parent_theta, parent_theta],
            [parent_r, outer_r],
            color="black",
            linewidth=0.8,
            zorder=1,
        )

        child_angles = [node_positions[child][1] for child in children]
        min_angle, max_angle = min(child_angles), max(child_angles)

        # Draw arc connecting children
        arc_r = outer_r
        theta = np.linspace(min_angle, max_angle, 50)
        ax.plot(theta, [arc_r] * len(theta), color="black", linewidth=0.8, zorder=1)

        # Draw stems from arc to each child
        for child in children:
            child_r, child_theta = node_positions[child]
            ax.plot(
                [child_theta, child_theta],
                [arc_r, child_r],
                color="black",
                linewidth=0.8,
                zorder=1,
            )

    # Draw nodes
    for node_id, (r, theta) in node_positions.items():
        full_name = node_depths[node_id]["name"]
        if full_name in agent_data.index:
            color = (
                "lightgray"
                if full_name in dead_agents
                else cmap(norm(agent_data.loc[full_name, "AmpC conc"]))
            )
            ax.scatter(theta, r, color=color, edgecolor="black", s=50, zorder=2)

    # Add colorbar
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cbar_ax,
        orientation="horizontal",
    )
    cbar.set_label("AmpC (periplasm, nM)", size=8)
    cbar.ax.tick_params(labelsize=7)

    # Adjust ticks on colorbar
    xticks = [int(np.round(min_conc, 1)), int(np.round(max_conc, 1))]
    cbar.ax.set_xticks(xticks)
    cbar.ax.set_xticklabels(xticks, size=7)

    # Save figure
    os.makedirs("out/analysis/paper_figures/", exist_ok=True)
    fig.savefig(
        "out/analysis/paper_figures/ampc_phylo.svg", bbox_inches="tight", dpi=300
    )
    plt.close()

    # Export data for downstream analysis
    newick_str = create_newick_string(tree, node_depths) + ";"
    with open("out/analysis/paper_figures/amp_tree.nw", "w") as f:
        f.write(newick_str)

    leaf_names = [
        info["name"]
        for info in node_depths.values()
        if info["name"] in agent_data.index
    ]
    agent_data.loc[leaf_names, :].to_csv("out/analysis/paper_figures/agent_data.csv")
