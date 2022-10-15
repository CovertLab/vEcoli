from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_final_dists(
    data: pd.DataFrame,
    out: str = None,
    axes: List[plt.Axes] = None,
) -> None:
    '''Plot data at the end of each simulation.

    Args:
        data: DataFrame where each column is a variable to plot and each row
            is an agent. Data from all replicates is concatenated into this
            single DataFrame and labelled with a different hex color in
            the "color" column. The DataFrame also has a "Condition" column
            that labels each experimental condition with a unique string.
        out: Prefix for output plot filename. Separate plots will be created
            and saved for each column in data. Do not use with ``axes``.
        axes: If supplied, columns are plotted sequentially on these Axes.
    '''
    metadata_columns = ["Color", "Condition", "Time", "Division", "Death", "Agent ID"]
    colors = data["Color"].unique()
    palette = {color: color for color in colors}
    grouped_data = data.groupby(by=["Condition", "Color"])
    final_data = []
    n_variables = len(data.columns) - len(metadata_columns)
    if not axes:
        _, fresh_axes = plt.subplots(nrows=n_variables, ncols=1, 
            sharex=True, figsize=(2*n_variables, 8))
        axes = np.ravel(fresh_axes)
    for rep_data in grouped_data:
        # Bring "Condition" and "Color" columns back
        rep_data = rep_data.reset_index()
        # Only plot data from the very last timestep for each condition
        max_time = data["Time"].max()
        final_agents = rep_data.loc[data["Time"]==max_time, :]
        final_data.append(final_agents)
    final_data = pd.concat(final_data, ignore_index=True)
    ax_idx = 0
    for column in final_data.columns:
        if column not in metadata_columns:
            curr_ax = axes[ax_idx]
            ax_idx += 1
            sns.histplot(
                data=final_data, x=column, hue="Color", legend=False,
                palette=palette, element="step", ax=curr_ax)
            if out:
                fig = curr_ax.get_figure()
                plt.tight_layout()
                fig.savefig('out/analysis/antibiotics_colony/' + 
                    f'{out}_{column.replace("/", "_")}.png')
                plt.close(fig)
