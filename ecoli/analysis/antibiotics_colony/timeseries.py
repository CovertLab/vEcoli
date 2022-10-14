from typing import Union, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Color = Union[str, Tuple[float, float, float, float]]


def plot_timeseries(
    data: pd.DataFrame,
    out: str,
) -> None:
    '''Plot data as a collection of strip plots with overlaid boxplots.

    Args:
        data: DataFrame where each column is a variable to plot and each row
            is an agent. Data from all replicates is concatenated into this
            single DataFrame and labelled with a different hex color in
            the "color" column. The DataFrame also has a "Condition" column
            that labels each experimental condition with a unique string.
        out: Prefix for ouput filenames in out/analysis/antibiotics_colony/
    '''
    metadata_columns = ["Color", "Condition", "Time", "Division", "Death"]
    colors = data["Color"].unique()
    palette = {color: color for color in colors}
    for column in data.columns:
        if column not in metadata_columns:
            g = sns.lineplot(
                data=data, x="Time", y=column, hue="Color", palette=palette,
                    errorbar=("pi", 50), legend=False, estimator=np.median)
            if "Death" in data.columns:
                death = data[data["Death"]]
                g = sns.scatterplot(
                    data=death, x="Time", y=column, hue="Color", ax=g,
                    palette=palette, markers=["X"], style="Death", legend=False)
            if "Division" in data.columns:
                divide = data[data["Division"]]
                g = sns.scatterplot(
                    data=divide, x="Time", y=column, hue="Color", ax=g, 
                    palette=palette, markers=["P"], style="Division", legend=False)
            fig = g.get_figure()
            plt.tight_layout()
            fig.savefig('out/analysis/antibiotics_colony/' + 
                f'{out}_{column.replace("/", "_")}.png')
            plt.close(fig)
