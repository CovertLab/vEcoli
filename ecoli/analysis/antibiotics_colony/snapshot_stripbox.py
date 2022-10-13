import os
import json
import argparse
from typing import (
    Dict, Union, Tuple, List, Any)

import pandas as pd
import seaborn as sns
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import remove_units
from vivarium.library.dict_utils import get_value_from_path

from ecoli.analysis.db import access_counts

Color = Union[str, Tuple[float, float, float, float]]

SPECIAL_PATH_PREFIXES = {
    'monomer': 'monomer_names',
    'mrna': 'mrna_names',
    'rna_init': 'rna_init',
    'data': 'outer_paths'
}

def agent_data_table(
    raw_data: Dict[float, Dict[str, Any]],
    paths_dict: Dict[str, Dict[str, Any]]
) -> Dict[float, pd.DataFrame]:
    """Combine data from all agents into a nested dictionary with
    list leaf values.
    
    Args:
        raw_data: Raw data from a single timepoint.
        paths_dict: Dictionary mapping paths within each agent to names
            that will be used the keys in the returned dictionary.
    
    Returns:
        Dataframe where each column is a path and each row is an agent."""
    collected_data = {}
    agents_at_time = raw_data['agents']
    for name, path in paths_dict.items():
        if name not in collected_data:
            collected_data[name] = []
        for agent_at_time in agents_at_time.values():
            value_in_agent = get_value_from_path(agent_at_time, path)
            # Replace missing values with 0
            if value_in_agent == None:
                value_in_agent = 0
            collected_data[name].append(value_in_agent)
    return pd.DataFrame(collected_data)


def retrieve_data(
    configs: str,
    colors: List[str]):
    """Retrieves data for each replicate (config file).
    
    Args:
        configs: Filename in ecoli/analysis/antibiotics_colony/plot_configs
            (omit .json extension)
        colors: List of ordered hex colors (one for each config file)
    
    Returns:
        DataFrame where each row is an agents and each column is a variable
        to plot, with the following exceptions. The "color" column labels each
        replicate with a different hex color. The "Condition" column labels
        all entries in the DataFrame with the value of the "condition" key in
        the configs (should be consistent across configs if specified in >1).
    """
    data = pd.DataFrame()
    condition = None
    for i, config_file in enumerate(configs):
        config_path = "ecoli/analysis/antibiotics_colony/" + \
            f"plot_configs/{config_file}.json"
        with open(config_path, 'r') as config_json:
            config = json.load(config_json)
        if condition == None:
                condition = config.get("condition", None)
        if config.get("condition", None):
            assert condition == config.get("condition", None)
        paths_dict = config.pop('paths_dict', None)
        if not paths_dict:
            raise Exception(f'No paths_dict in {config_file}')
        rep_data = {}
        for data_config in config['experiments']:
            for path in paths_dict.values():
                # If path does not start with a special prefix, assume it is a
                # path to some store inside each agent (an "inner" path)
                path_key = SPECIAL_PATH_PREFIXES.get(path[0], 'inner_paths')
                if path_key not in data_config:
                    data_config[path_key] = []
                if path_key == 'inner_paths':
                    data_config[path_key].append(tuple(path))
                else:
                    data_config[path_key].append(path[-1])
            # Should only be retrieving a couple time points so this is fine
            data_config["cpus"] = 1
            exp_data = remove_units(
                deserialize_value(access_counts(**data_config)))
            rep_data.update(exp_data)
        for time, rep_data_at_time in rep_data.items():
            data_combined = agent_data_table(
                rep_data_at_time, paths_dict)
            data_combined["color"] = [colors[i]] * len(data_combined)
            data_combined["time"] = [time] * len(data_combined)
            data = pd.concat([data, data_combined], ignore_index=True)
        data["Condition"] = [condition] * len(data)
    return data


def plot_boxstrip(
    data: pd.DataFrame,
    out: str
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
    colors = data["color"].unique()
    palette = {color: color for color in colors}
    for column in data.columns:
        if column not in ["color", "Condition", "time"]:
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
