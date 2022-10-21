from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

METADATA_AGG = {
    "Color": "first",
    "Condition": "first",
    "Time": "max",
    "Division": "max",
    "Death": "max",
    "Agent ID": "first",
    "Boundary": "first"
}

def plot_death_dists(
    data: pd.DataFrame,
    out: str,
    axes: List[plt.Axes] = None,
) -> None:
    '''Plot distributions of variables for cells that die in their generation
    of death and cells that survive until the end of the simulation.

    Args:
        data: DataFrame where each column is a variable to plot and each row
            is an agent. Data from all replicates is concatenated into this
            single DataFrame and labelled with a different hex color in
            the "color" column. The DataFrame also has a "Condition" column
            that labels each experimental condition with a unique string.
        out: Prefix for output plot filename. Separate plots will be created
            and saved for each column in data.
        axes: If supplied, columns are plotted sequentially on these Axes.
    '''
    # Plot data averaged over each agent's lifespan (each color is a replicate)
    grouped_agent_data = data.groupby(by=["Color", "Agent ID"])
    columns_to_plot = set(data.columns) - METADATA_AGG.keys()
    column_agg = {column: np.mean for column in columns_to_plot}
    avg_agent_data = grouped_agent_data.aggregate(
        {**METADATA_AGG, **column_agg}
    )

    # Get data for cells that will die in their current generation
    dead_cells = grouped_agent_data.loc[grouped_agent_data["Death"], :]
    dead_agent_ids = set(dead_cells.index)
    imminent_death_data = []
    for agent_id in dead_agent_ids:
        avg_agent_data.loc[agent_id, "Death"] = True
        imminent_death_data.append(avg_agent_data.loc[agent_id, :])

    # Get data for cells that live to the end of the simulation
    end_time = grouped_agent_data.loc[:, "Time"].max()
    final_agent_ids = set(grouped_agent_data.loc[
        data["Time"]==end_time, :].index)
    final_gen = max(len(agent_id[1]) for agent_id in final_agent_ids)
    # Pull from penultimate generation to get averages for full cell cycle
    live_cell_data = []
    for agent_id in final_agent_ids:
        agent_id = (agent_id[0], agent_id[1][:final_gen])
        avg_agent_data.loc[agent_id, "Death"] = False
        live_cell_data.append(avg_agent_data.loc[agent_id, :])
    data = pd.concat(imminent_death_data + live_cell_data)

    if not axes:
        nrows = int(np.ceil(len(columns_to_plot) / 2))
        _, fresh_axes = plt.subplots(nrows=nrows, ncols=2, 
            sharex=True, figsize=(8, 3*nrows))
        axes = np.ravel(fresh_axes)
    # Use single color for all replicates for a given condition
    colors = data.groupby(by="Condition").aggregate({"Color": "first"})["Color"]
    palette = colors.to_dict()
    ax_idx = 0
    for column in columns_to_plot:
        curr_ax = axes[ax_idx]
        ax_idx += 1
        sns.histplot(
            data=data, x=column, hue="Condition", legend=False,
            palette=palette, element="step", ax=curr_ax)
    if out:
        fig = curr_ax.get_figure()
        plt.tight_layout()
        fig.savefig('out/analysis/antibiotics_colony/' + 
            f'{out}_{column.replace("/", "_")}.png')
        plt.close(fig)
