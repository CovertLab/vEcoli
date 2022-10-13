import os
import argparse
from typing import Union, Tuple

import pandas as pd
import seaborn as sns

from ecoli.analysis.antibiotics_colony.plot_utils import retrieve_data

Color = Union[str, Tuple[float, float, float, float]]


def plot_boxstrip(
    data: pd.DataFrame,
    out: str
) -> None:
    '''Plot data as a collection of strip plots with overlaid boxplots.

    Args:
        data: DataFrame where each column is a variable to plot and each row
            is an agent. Data from all replicates is concatenated into this
            single DataFrame and labelled with a different hex color in
            the "Color" column. The DataFrame also has a "Condition" column
            that labels each experimental condition with a unique string.
        out: Prefix for ouput filenames in out/analysis/antibiotics_colony/
    '''
    colors = data["Color"].unique()
    palette = {color: color for color in colors}
    for column in data.columns:
        if column not in ["Color", "Condition", "Time", "Division", "Death"]:
            g = sns.catplot(
                data=data, kind="box",
                x="Condition", y=column, col="time",
                boxprops={'facecolor':'None'}, showfliers=False,
                aspect=0.5, legend=False)
            g.map_dataframe(sns.stripplot, x="Condition", y=column,
                hue="color", palette=palette, alpha=0.5, size=3)
            g.savefig('out/analysis/antibiotics_colony/' + 
                f'{out}_{column.replace("/", "_")}.png')


def main():
    sns.set_style("whitegrid")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--configs', '-c', type=str, nargs='+', required=True,
        help="""Filename of JSON config in ecoli/analysis/antibiotics_colony/
            plot_configs (omit ".json" extension). Each JSON file configures
            data retrieval for one replicate. If multiple experiment IDs are
            provided for a single replicate, the data is collected sequentially
            and merged (Note: data from later experiments overwrites data
            from earlier ones if they share any timepoints)."""
    )
    parser.add_argument(
        '--baseline_configs', '-b', type=str, nargs='+', default=[],
        help="""Filename of JSON config in ecoli/analysis/antibiotics_colony/
            plot_configs (omit ".json" extension) with data access options for
            baseline simulations. If present, baseline data will be plotted
            in grey next to the experimental data, which will be in cyan."""
    )
    parser.add_argument(
        '--out', '-o', type=str, default="snapshot_ridgeline",
        help="""Prefix for output plot filenames. All plots are saved in
            out/analysis/antibiotics_colony."""
    )
    args = parser.parse_args()
    # Shades of grey for baseline distributions (up to 3 replicates)
    baseline_colors = ('#333333', '#777777', '#BBBBBB')
    # Shades of blue-green for experimental distributions (up to 3 replicates)
    colors = ('#5F9EA0', '#088F8F', '#008080')
    if args.baseline_configs:
        data = retrieve_data(args.baseline_configs, baseline_colors)
        exp_data = retrieve_data(args.configs, colors)
        data = pd.concat([data, exp_data], ignore_index=True)
    else:
        data = retrieve_data(args.configs, baseline_colors)

    os.makedirs('out/analysis/antibiotics_colony/', exist_ok=True)
    plot_boxstrip(data, args.out)


if __name__ == "__main__":
    main()
