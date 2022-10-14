import argparse
from functools import partial
import json
import os
from typing import Dict, List, Any, Tuple, Callable
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import remove_units
from vivarium.library.dict_utils import get_value_from_path

from ecoli.analysis.db import access_counts
from ecoli.analysis.antibiotics_colony.timeseries import plot_timeseries
from ecoli.analysis.antibiotics_colony.snapshot_stripbox import plot_stripbox

SPECIAL_PATH_PREFIXES = {
    'monomer': 'monomer_names',
    'mrna': 'mrna_names',
    'rna_init': 'rna_init',
    'data': 'outer_paths'
}


PLOT_NAME_TO_FUN = {
    "timeseries": plot_timeseries,
    "stripbox": plot_stripbox,
}


def get_config_names(prefix):
    return [f'{prefix}_seed_{seed}' for seed in ('0', '100', '10000')]


def agent_data_table(
    raw_data: Tuple[float, Dict[float, Dict[str, Any]]],
    paths_dict: Dict[str, Dict[str, Any]],
    color: str
) -> Dict[float, pd.DataFrame]:
    """Combine data from all agents into a nested dictionary with
    list leaf values.
    
    Args:
        raw_data: Tuple of (time, dictionary at time for one replicate).
        paths_dict: Dictionary mapping paths within each agent to names
            that will be used the keys in the returned dictionary.
        color: Hex color for all data from this replicate.
    
    Returns:
        Dataframe where each column is a path and each row is an agent."""
    time = raw_data[0]
    raw_data = raw_data[1]
    collected_data = {'Agent ID': []}
    agents_at_time = raw_data['agents']
    for agent_id, agent_at_time in agents_at_time.items():
        collected_data['Agent ID'].append(agent_id)
        for name, path in paths_dict.items():
            if name not in collected_data:
                collected_data[name] = []
            value_in_agent = get_value_from_path(agent_at_time, path)
            # Replace missing values with 0
            if value_in_agent == None:
                value_in_agent = 0
            collected_data[name].append(value_in_agent)
    collected_data = pd.DataFrame(collected_data)
    collected_data["Color"] = [color] * len(collected_data)
    collected_data["Time"] = [time] * len(collected_data)
    return collected_data


def mark_death_and_division(data: pd.DataFrame):
    """Uses the "Color" (replicate), "Agent ID", and "Time" columns to identify
    the last timestep that a given agent is alive for a given replicate. If
    that agent has daughter cells, mark that timepoint as the time of division
    in a new "Division" column. If not, mark that timepoint as the time of
    death in a new "Death" column."""
    grouped_reps = data.groupby("Color")
    data["Death"] = np.zeros(len(data), dtype=np.bool_)
    data["Division"] = np.zeros(len(data), dtype=np.bool_)
    for rep_color, rep_data in grouped_reps:
        grouped_agents = rep_data.groupby("Agent ID")
        final_times = grouped_agents["Time"].max().reset_index().to_numpy()
        for agent_id, final_time in final_times:
            mask = ((data["Color"]==rep_color) & (data["Agent ID"]==agent_id)
                & (data["Time"]==final_time))
            if agent_id+'0' in rep_data["Agent ID"].to_list():
                data.loc[mask, "Division"] = True
            else:
                data.loc[mask, "Death"] = True
    return data


def retrieve_data(
    configs: str,
    colors: List[str],
    sampling_rate: int,
    div_and_death: bool = False,
    cpus: int = 8):
    """Retrieves data for each replicate (config file).
    
    Args:
        configs: Filename in ecoli/analysis/antibiotics_colony/plot_configs
            (omit .json extension)
        colors: List of ordered hex colors (one for each config file)
        div_and_death: Adds boolean "Division" and "Death" columns to mark
            timestep when agent divides or dies.
    
    Returns:
        DataFrame where each row is an agents and each column is a variable
        to plot, with the following exceptions. The "Color" column labels each
        replicate with a different hex color. The "Condition" column labels
        all entries in the DataFrame with the value of the "condition" key in
        the configs (should be consistent across configs if specified in >1).
        The "Time" column indicates the timestep that the data for that agent
        came from. The "Division" column indicates the timestep that an agent
        divides. The "Death" column indicates the timestep that an agent dies.
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
            if "cpus" not in data_config:
                # Assume only retrieving a few timepoints
                data_config["cpus"] = 1
            data_config["sampling_rate"] = sampling_rate
            exp_data = remove_units(
                deserialize_value(access_counts(**data_config)))
            rep_data.update(exp_data)
        agent_df_paths = partial(agent_data_table, paths_dict=paths_dict,
            color=colors[i])
        with ProcessPoolExecutor(cpus) as executor:
            data = list(tqdm(executor.map(agent_df_paths, rep_data.items())))
        data = pd.concat(data, ignore_index=True)
        data["Condition"] = [condition] * len(data)
    if div_and_death:
        data = mark_death_and_division(data)
    return data


def make_plots(
    baseline_configs: List[str],
    exp_configs: List[str],
    baseline_colors: List[str],
    exp_colors: List[str],
    sampling_rate: int,
    plot_fun: Callable,
    div_and_death: bool = False,
    cpus: int = 8,
    out: str = 'stripbox'
):
    """Helper function to retrieve and plot timeseries data.
    
    Args:
        baseline_configs: List of JSON filenames to configure
            baseline data retrieval (each is a replicate).
        exp_configs: List of JSON filenames to configure
            experimental data retrieval (each is a replicate).
        baseline_colors: Hex colors for baseline replicates.
        exp_colors: Hex colors for experimental replicates.
        sampling_rate: Plot data for times equal to 0 mod this value.
        plot_fun: Plotting function that takes two inputs, a DataFrame
            as generated by ``retrieve_data`` and a prefix for the
            output plot filename
        div_and_death: Mark death and division on plots (X and +)
        cpus: Number of CPU cores to use.
        out: Prefix for output plot filenames
    """
    data = retrieve_data(
        configs=baseline_configs,
        colors=baseline_colors, 
        sampling_rate=sampling_rate,
        div_and_death=div_and_death,
        cpus=cpus)
    exp_data = retrieve_data(
        configs=exp_configs,
        colors=exp_colors, 
        sampling_rate=sampling_rate,
        div_and_death=div_and_death,
        cpus=cpus)
    data = pd.concat([data, exp_data], ignore_index=True)
    os.makedirs('out/analysis/antibiotics_colony/', exist_ok=True)
    plot_fun(data, out)


def main():
    sns.set_style("white")
    amp_configs = get_config_names('ampicillin')
    tet_configs = get_config_names('tetracycline')
    glc_tet_configs = get_config_names('glucose_tet')
    glc_amp_configs = get_config_names('glucose_amp')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot_type', '-n', type=str, default="timeseries",
        help="""Type of plot. See keys of ``PLOT_NAME_TO_FUN``
        dictionary in this file for valid types."""
    )
    parser.add_argument(
        '--tetracycline', '-t', type=bool, default=True,
        help="""Compare tetracycline sims with glucose sims."""
    )
    parser.add_argument(
        '--ampicillin', '-a', type=bool, default=True,
        help="""Compare ampicillin sims with glucose sims."""
    )
    parser.add_argument(
        '--division_and_death', '-d', type=bool, default=True,
        help="""Mark values for agents right before cell division with
            a "+" sign. Mark values for agents right before cell death
            with a "x" sign."""
    )
    parser.add_argument(
        '--cpus', '-p', type=int, default=1,
        help="""Number of CPU cores to use."""
    )
    args = parser.parse_args()
    # Shades of grey for baseline distributions (up to 3 replicates)
    baseline_colors = ('#333333', '#777777', '#BBBBBB')
    # Shades of blue-green for experimental distributions (up to 3 replicates)
    colors = ('#5F9EA0', '#088F8F', '#008080')
    plot_fun = PLOT_NAME_TO_FUN[args.plot_type]
    if args.tetracycline:
        make_plots(
            baseline_configs=glc_tet_configs,
            exp_configs=tet_configs,
            baseline_colors=baseline_colors,
            exp_colors=colors,
            sampling_rate=args.sampling_rate,
            plot_fun=plot_fun,
            div_and_death=args.division_and_death,
            cpus=args.cpus,
            out='tetracycline_sb')
    if args.ampicillin:
        make_plots(
            baseline_configs=glc_amp_configs,
            exp_configs=amp_configs,
            baseline_colors=baseline_colors,
            exp_colors=colors,
            sampling_rate=args.sampling_rate,
            plot_fun=plot_fun,
            div_and_death=args.division_and_death,
            cpus=args.cpus,
            out='ampicillin_sb')


if __name__ == "__main__":
    main()
