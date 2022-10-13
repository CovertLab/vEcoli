import json
from typing import Dict, List, Any

import numpy as np
import pandas as pd
from vivarium.core.serialize import deserialize_value
from vivarium.library.units import remove_units
from vivarium.library.dict_utils import get_value_from_path

from ecoli.analysis.db import access_counts

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
    collected_data = {'Agent ID': []}
    agents_at_time = raw_data['agents']
    for name, path in paths_dict.items():
        if name not in collected_data:
            collected_data[name] = []
        for agent_id, agent_at_time in agents_at_time.items():
            collected_data['Agent ID'].append(agent_id)
            value_in_agent = get_value_from_path(agent_at_time, path)
            # Replace missing values with 0
            if value_in_agent == None:
                value_in_agent = 0
            collected_data[name].append(value_in_agent)
    return pd.DataFrame(collected_data)


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
            mask = (data["Color"]==rep_color and data["Agent ID"]==agent_id
                and data["Time"]==final_time)
            if agent_id+'0' in rep_data["Agent ID"]:
                data.loc[mask, "Division"] = True
            else:
                data.loc[mask, "Death"] = True
    return data

def retrieve_data(
    configs: str,
    colors: List[str],
    div_and_death: bool = False):
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
            exp_data = remove_units(
                deserialize_value(access_counts(**data_config)))
            rep_data.update(exp_data)
        for time, rep_data_at_time in rep_data.items():
            data_combined = agent_data_table(
                rep_data_at_time, paths_dict)
            data_combined["Color"] = [colors[i]] * len(data_combined)
            data_combined["Time"] = [time] * len(data_combined)
            data = pd.concat([data, data_combined], ignore_index=True)
        data["Condition"] = [condition] * len(data)
    if div_and_death:
        data = mark_death_and_division(data)
    return data
