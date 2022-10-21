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

def plot_timeseries(
    data: pd.DataFrame,
    out: str = None,
    axes: List[plt.Axes] = None,
) -> None:
    '''For each generation of cells (identified by length of their agent ID),
    all generation start times are aligned and averaged to produce a bulk
    timeseries plot spanning multiple generations. The end time point for
    each generation is the median generation time, giving each cell a .

    Args:
        data: DataFrame where each column is a variable to plot and each row
            is an agent. Data from all replicates is concatenated into this
            single DataFrame and labelled with a different hex color in
            the "Color" column. The DataFrame also has a "Condition" column
            that labels each experimental condition with a unique string.
        out: Prefix for output plot filename. Do not use with ``axes``.
        axes: If supplied, columns are plotted sequentially on these Axes.
    '''
    agent_id_lengths = np.array([len(agent_id) for agent_id in data["Agent ID"]])
    n_generations = max(agent_id_lengths)
    previous_end_time = 0
    aligned_data = []
    # For each generation, assume start time = previous generation's end time
    # (0 for first gen.). Get median generation length and filter out all data
    # that exceeds that. Start time + median gen. length = new gen. start time
    for generation in range(1, n_generations +  1):
        generation_data = data.loc[agent_id_lengths == generation, :]
        grouped_data = generation_data.groupby(["Condition", "Color", "Agent ID"])
        real_start_time = generation_data.aggregate({"Time": "min"})
        target_start_time = previous_end_time
        real_gen_length = grouped_data.aggregate({"Time": "max"}) - real_start_time
        target_end_time = int(target_start_time + real_gen_length.median())
        for _, agent_data in grouped_data:
            agent_start_time =  int(agent_data.aggregate({"Time": "min"}))
            agent_data.loc[:, "Time"] -= agent_start_time - target_start_time
            agent_data = agent_data.loc[agent_data["Time"] <= target_end_time, :]
            agent_data = agent_data.reset_index()
            aligned_data.append(agent_data)
        previous_end_time = target_end_time
    aligned_data = pd.concat(aligned_data)

    colors = data["Color"].unique()
    palette = {color: color for color in colors}
    columns_to_plot = set(data.columns) - METADATA_AGG.keys()
    n_variables = len(columns_to_plot)
    if not axes:
        _, fresh_axes = plt.subplots(nrows=n_variables, ncols=1, 
            sharex=True, figsize=(8, 2*n_variables))
        axes = np.ravel(fresh_axes)
    ax_idx = 0
    for column in columns_to_plot:
        curr_ax = axes[ax_idx]
        ax_idx += 1
        g = sns.lineplot(
            data=aligned_data, x="Time", y=column, hue="Color", palette=palette,
            errorbar=("pi", 50), legend=False, estimator=np.median, ax=curr_ax)
        if "Death" in data.columns:
            death = aligned_data.loc[aligned_data["Death"], :]
            g = sns.scatterplot(
                data=death, x="Time", y=column, hue="Color", ax=g,
                palette=palette, markers=["X"], style="Death", legend=False)
    if out:
        fig = g.get_figure()
        plt.tight_layout()
        fig.savefig('out/analysis/antibiotics_colony/' + 
            f'{out}_generational_timeseries.png')
        plt.close(fig)
