from typing import List, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ecoli.plots.snapshots import plot_snapshots

METADATA_AGG = {
    "Color": "first",
    "Condition": "first",
    "Time": "max",
    "Division": "max",
    "Death": "max",
    "Agent ID": "first",
    "Boundary": "first"
}

def plot_lineage_trace(
    data: pd.DataFrame,
    out: str = None,
    axes: List[plt.Axes] = None,
    bounds: List[float] = None,
    fields: Dict[float, Dict[str, List[List[float]]]] = None,
    agent_id: str = None
) -> None:
    '''Plot multi-generational trace for a single lineage.

    Args:
        data: DataFrame where each column is a variable to plot and each row
            is an agent. Data from all replicates is concatenated into this
            single DataFrame and labelled with a different hex color in
            the "color" column. The DataFrame also has a "Condition" column
            that labels each experimental condition with a unique string.
        out: Prefix for output plot filename. Separate plots will be created
            and saved for each column in data. Do not use with ``axes``.
        axes: If supplied, columns are plotted sequentially on these Axes.
        bounds: Height and width of spatial environment
        fields: Dictionary mapping timepoints to sub-dictionaries that map
            molecules to their environmental concentration fields at that time
        agent_id: Agent ID from final generation of lineage to plot
    '''
    experimental_condition = set(data["Condition"].unique()) - {"Glucose"}
    hex_palette = {
        # Gray for control
        "Glucose": "#708090",
        # Maroon for experimental
        list(experimental_condition)[0]: "#700000",
    }
    rgb_palette = {
        "Glucose": (111/256, 127/256, 143/256),
        list(experimental_condition)[0]: (112/256, 0, 0),
    }
    
    conditions_to_plot = data.groupby(["Condition"])
    columns_to_plot = set(data.columns) - METADATA_AGG.keys()
    if not axes:
        n_variables = len(columns_to_plot)
        _, fresh_axes = plt.subplots(nrows=n_variables, ncols=1, 
            sharex=True, figsize=(8, 2*n_variables))
        axes = np.ravel(fresh_axes)
    final_data_to_plot = []
    for condition, condition_to_plot in conditions_to_plot:
        # Plot timeseries for first replicate for each condition by default
        grouped_data = condition_to_plot.groupby(by=["Color"])
        replicates = condition_to_plot["Color"].unique()
        rep_to_plot = grouped_data.get_group(replicates[0])

        if agent_id == None:
            final_time = rep_to_plot["Time"].max()
            data_at_final_time = rep_to_plot.loc[
                rep_to_plot["Time"]==final_time]
            # Plot timeseries of lineage for first surviving lineage by default
            agent_id = data_at_final_time["Agent ID"].unique()[0]
        row_mask = rep_to_plot["Agent ID"]==agent_id
        for generation in range(1, len(agent_id)):
            row_mask = row_mask | (rep_to_plot["Agent ID"]==agent_id[:generation])
        agent_to_plot = rep_to_plot.loc[row_mask, :]
        final_data_to_plot.append(agent_to_plot)
        
        # Convert back to dictionary for snapshot plot function
        snapshot_data = {}         
        for time, agent_id, boundary in zip(
            rep_to_plot["Time"], rep_to_plot["Agent ID"], rep_to_plot["Boundary"]
        ):
            data_at_time = snapshot_data.setdefault(time, {})
            agent_at_time = data_at_time.setdefault(agent_id, {})
            agent_at_time["boundary"] = boundary
        color = rgb_palette[condition]
        snapshots_palette = {
            # Gray for control
            agent_id[:generation]: color
            for generation in range(1, len(agent_id))
        }
        fields = fields[replicates[0]]
        relevant_fields = [
            molecule
            for molecule in fields[max(fields.keys())]
            if np.array(fields[max(fields.keys())][molecule]).sum() > 0
        ]
        snapshots_palette[agent_id] = color
        snapshots_fig = plot_snapshots(
            agents=snapshot_data,
            fields=fields,
            bounds=bounds,
            agent_colors=snapshots_palette,
            include_fields=relevant_fields
        )
        snapshots_fig.savefig('out/analysis/antibiotics_colony/' + 
            f'{out}_snapshots.png')
        plt.close(snapshots_fig)
    final_data_to_plot = pd.concat(final_data_to_plot)
    ax_idx = 0
    for column in columns_to_plot:
        curr_ax = axes[ax_idx]
        ax_idx += 1
        sns.lineplot(
            data=final_data_to_plot, x="Time", y=column,
            hue="Condition", palette=hex_palette, ax=curr_ax,
            legend=False)
    if out:
        fig = curr_ax.get_figure()
        plt.tight_layout()
        fig.savefig('out/analysis/antibiotics_colony/' + 
            f'{out}_lineage.png')
        plt.close(fig)
