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

def plot_stripbox(
    data: pd.DataFrame,
    out: str = "stripbox"
) -> None:
    '''Plot data as a collection of strip plots with overlaid boxplots.

    Args:
        data: DataFrame where each column is a variable to plot and each row
            is an agent. Data from all replicates is concatenated into this
            single DataFrame and labelled with a different hex color in
            the "Color" column. The DataFrame also has a "Condition" column
            that labels each experimental condition with a unique string.
        out: Prefix for output plot filename.
    '''
    colors = data["Color"].unique()
    palette = {color: color for color in colors}
    columns_to_plot = set(data.columns) - METADATA_AGG.keys()
    for column in columns_to_plot:
        g = sns.catplot(
            data=data, kind="box",
            x="Condition", y=column, col="Time",
            boxprops={'facecolor':'None'}, showfliers=False,
            aspect=0.5, legend=False)
        g.map_dataframe(sns.stripplot, x="Condition", y=column,
            hue="Color", palette=palette, alpha=0.5, size=3)
        plt.tight_layout()
        g.savefig('out/analysis/antibiotics_colony/' + 
            f'{out}_{column.replace("/", "_")}.png')
        plt.close(g)
