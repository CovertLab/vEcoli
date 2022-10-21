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
from vivarium.library.dict_utils import get_value_from_path

from ecoli.analysis.db import access_counts
from ecoli.analysis.snapshots_video import deserialize_and_remove_units
from ecoli.analysis.antibiotics_colony.generational_timeseries import plot_timeseries
from ecoli.analysis.antibiotics_colony.stripbox import plot_stripbox
from ecoli.analysis.antibiotics_colony.final_dists import plot_final_dists
from ecoli.analysis.antibiotics_colony.lineage import plot_lineage_trace
from ecoli.analysis.antibiotics_colony.death_dists import plot_death_dists


SPECIAL_PATH_PREFIXES = {
    'monomer': 'monomer_names',
    'mrna': 'mrna_names',
    'rna_init': 'rna_init',
    'rna_synth_prob': 'rna_synth_prob',
    'data': 'outer_paths'
}

PLOT_NAME_TO_FUN = {
    "timeseries": plot_timeseries,
    "stripbox": plot_stripbox,
    "final_dists": plot_final_dists,
    "lineage": plot_lineage_trace,
    "death_dists": plot_death_dists
}

PLOT_NAME_TO_SAMPLING_RATE = {
    "timeseries": 2,
    "stripbox": 2600,
    "final_dists": 26000,
    "lineage": 2,
    "death_dists": 2
}


def get_config_names(prefix):
    return [f'{prefix}_seed_{seed}' for seed in ('0', '100', '10000')]


def agent_data_table(
    raw_data: Tuple[float, Dict[float, Dict[str, Any]]],
    paths_dict: Dict[str, Dict[str, Any]],
    color: str
) -> Dict[float, pd.DataFrame]:
    """Combine data from all agents into DataFrames for each timestep.
    
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
        # Cells that exist at last timestep have no daughters but are alive
        max_time = rep_data["Time"].max()
        data.loc[data["Time"] == max_time, "Death"] = False
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
        data: DataFrame where each row is an agents and each column is a variable
            to plot, with the following exceptions. The "Color" column labels each
            replicate with a different hex color. The "Condition" column labels
            all entries in the DataFrame with the value of the "condition" key in
            the configs (should be consistent across configs if specified in >1).
            The "Time" column indicates the timestep that the data for that agent
            came from. The "Division" column indicates the timestep that an agent
            divides. The "Death" column indicates the timestep that an agent dies.
            The "Boundary" column contains dictionaries that holds the data in the
            "boundary" store (e.g. for snapshot plots)
        bounds: Bounds of spatial environment
        fields: Dictionary containing concentration arrays for each molecule in
            the spatial environment
    """
    data = pd.DataFrame()
    condition = None
    bounds = None
    fields = {}
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
        paths_dict["Boundary"] = ["boundary"]
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
                data_config["cpus"] = 24
            data_config["sampling_rate"] = sampling_rate
            exp_data = access_counts(**data_config)
            rep_data.update(exp_data)
        agent_df_paths = partial(agent_data_table, paths_dict=paths_dict,
            color=colors[i])
        with ProcessPoolExecutor(cpus) as executor:
            print("Deserializing data and removing units...")
            data_deserialized = list(tqdm(executor.map(
                deserialize_and_remove_units, rep_data.values()),
                total=len(rep_data)))
            rep_data = dict(zip(rep_data.keys(), data_deserialized))
        print("Extracting spatial environment data...")
        if bounds == None:
            # Assume all experiments use same spatial environment dimensions
            bounds = rep_data[min(rep_data)]["dimensions"]["bounds"]
        fields.setdefault(colors[i], {})
        fields[colors[i]].update({
            time: data_at_time["fields"] for time, data_at_time in rep_data.items()
        })
        with ProcessPoolExecutor(cpus) as executor:
            print("Converting data to DataFrame...")
            data = list(tqdm(executor.map(agent_df_paths, rep_data.items()),
                total=len(rep_data)))
        data = pd.concat(data, ignore_index=True)
        data["Condition"] = [condition] * len(data)
    if div_and_death:
        print("Marking agent death and division...")
        data = mark_death_and_division(data)
    return data, bounds, fields


def apply_sampling_rate(
    data: pd.DataFrame,
    sampling_rate: int
):
    """Takes a DataFrame from ``retrieve_data`` and applies further
    downsampling."""
    return data.loc[(data["Time"] % sampling_rate) == 0, :]


def dict_sampling_rate(
    data: Dict,
    sampling_rate: int
):
    """Takes a timeseries dictionary and applies further downsampling."""
    current_times = list(data.keys())
    timepoints_to_keep = [
        time for time in current_times if float(time) % sampling_rate == 0]
    downsampled_data = {
        time: data[time]
        for time in timepoints_to_keep
    }
    return downsampled_data


def make_plots(
    baseline_configs: List[str],
    exp_configs: List[str],
    baseline_colors: List[str],
    exp_colors: List[str],
    sampling_rates: List[int],
    plot_funcs: List[Callable],
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
    sampling_rate = min(sampling_rates)
    ctl_data, ctl_bounds, ctl_fields = retrieve_data(
        configs=baseline_configs,
        colors=baseline_colors, 
        sampling_rate=sampling_rate,
        div_and_death=div_and_death,
        cpus=cpus)
    exp_data, exp_bounds, exp_fields = retrieve_data(
        configs=exp_configs,
        colors=exp_colors, 
        sampling_rate=sampling_rate,
        div_and_death=div_and_death,
        cpus=cpus)
    assert exp_bounds == ctl_bounds
    data = pd.concat([ctl_data, exp_data], ignore_index=True)
    fields = {**exp_fields, **ctl_fields}
    os.makedirs('out/analysis/antibiotics_colony/', exist_ok=True)
    for sampling_rate, plot_func in zip(sampling_rates, plot_funcs):
        print(f"Calling {plot_func.__name__}...")
        downsampled_data = apply_sampling_rate(data, sampling_rate)
        if plot_func == plot_lineage_trace:
            downsampled_fields = {
                color: dict_sampling_rate(fields[color], sampling_rate)
                for color in fields
            }
            plot_func(
                data=downsampled_data, 
                out=f"{out}_{plot_func.__name__}",
                bounds=ctl_bounds,
                fields=downsampled_fields)
        else:
            plot_func(
                data=downsampled_data,
                out=f"{out}_{plot_func.__name__}")


def main():
    sns.set_style("white")
    amp_configs = get_config_names('ampicillin')
    tet_configs = get_config_names('tetracycline')
    glc_tet_configs = get_config_names('glucose_tet')
    glc_amp_configs = get_config_names('glucose_amp')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot_types', '-p', type=str, nargs="+", default=["timeseries"],
        help="""Types of plots to make. See keys of ``PLOT_NAME_TO_FUN``
        dictionary in this file for valid types."""
    )
    parser.add_argument(
        '--tetracycline', '-t', type=bool, default=True,
        help="""Compare tetracycline sims with glucose sims."""
    )
    parser.add_argument(
        '--ampicillin', '-a', type=bool, default=False,
        help="""Compare ampicillin sims with glucose sims."""
    )
    parser.add_argument(
        '--sampling_rates', '-s', type=int, nargs="+", default=[],
        help="""Custom sampling rates. See default sampling rates
        by plot type in ``PLOT_NAME_TO_SAMPLING_RATE``"""
    )
    parser.add_argument(
        '--division_and_death', '-d', type=bool, default=True,
        help="""Mark values for agents right before cell division with
            a "+" sign. Mark values for agents right before cell death
            with a "x" sign."""
    )
    parser.add_argument(
        '--cpus', '-c', type=int, default=1,
        help="""Number of CPU cores to use."""
    )
    args = parser.parse_args()
    # Shades of grey for baseline distributions (up to 3 replicates)
    baseline_colors = ('#333333', '#777777', '#BBBBBB')
    # Shades of blue-green for experimental distributions (up to 3 replicates)
    colors = ('#5F9EA0', '#088F8F', '#008080')
    plot_funcs = []
    sampling_rates = []
    for plot_type in args.plot_types:
        plot_funcs.append(PLOT_NAME_TO_FUN[plot_type])
        sampling_rates.append(PLOT_NAME_TO_SAMPLING_RATE[plot_type])
    if args.sampling_rates:
        sampling_rates = args.sampling_rates
    make_plots(
        baseline_configs=["local_test"],
        exp_configs=["local_test"],
        baseline_colors=baseline_colors,
        exp_colors=colors,
        sampling_rates=sampling_rates,
        plot_funcs=plot_funcs,
        div_and_death=args.division_and_death,
        cpus=args.cpus,
        out=f'local')
    # if args.tetracycline:
    #     make_plots(
    #         baseline_configs=glc_tet_configs,
    #         exp_configs=tet_configs,
    #         baseline_colors=baseline_colors,
    #         exp_colors=colors,
    #         sampling_rates=sampling_rates,
    #         plot_funcs=plot_funcs,
    #         div_and_death=args.division_and_death,
    #         cpus=args.cpus,
    #         out=f'tetracycline')
    # if args.ampicillin:
    #     make_plots(
    #         baseline_configs=glc_amp_configs,
    #         exp_configs=amp_configs,
    #         baseline_colors=baseline_colors,
    #         exp_colors=colors,
    #         sampling_rates=sampling_rates,
    #         plot_funcs=plot_funcs,
    #         div_and_death=args.division_and_death,
    #         cpus=args.cpus,
    #         out=f'ampicillin')


if __name__ == "__main__":
    main()
