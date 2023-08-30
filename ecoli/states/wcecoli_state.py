import ast
import json
import numpy as np
import concurrent.futures

from vivarium.core.serialize import deserialize_value
from vivarium.library.units import units

def load_states(path):
    with open(path, "r") as states_file:
        states = json.load(states_file)
    return states


def numpy_molecules(states):
    """
    Loads unique and bulk molecule data as Numpy structured arrays
    """
    if 'bulk_dtypes' in states:
        bulk_dtypes = ast.literal_eval(states.pop('bulk_dtypes'))
        bulk_tuples = [tuple(mol) for mol in states['bulk']]
        states['bulk'] = np.array(bulk_tuples, dtype=bulk_dtypes)
        # Numpy arrays are read-only outside of updater
        states['bulk'].flags.writeable = False
    if 'unique_dtypes' in states:
        for key, dtypes in states.pop('unique_dtypes').items():
            dtypes = ast.literal_eval(dtypes)
            unique_tuples = [tuple(mol) for mol in states['unique'][key]]
            states['unique'][key] = np.array(unique_tuples, dtype=dtypes)
            states['unique'][key].flags.writeable = False
    return states 


def get_state_from_file( 
    path="data/wcecoli_t0.json",
):
    serialized_state = load_states(path)
    # Parallelize deserialization of colony states
    if 'agents' in serialized_state:
        agents = serialized_state.pop('agents')
        n_agents = len(agents)
        with concurrent.futures.ProcessPoolExecutor(n_agents) as executor:
            deserialized_agents = executor.map(
                deserialize_value, agents.values())
        numpy_agents = []
        for agent in deserialized_agents:
            agent.pop('first_update', None)
            numpy_agents.append(numpy_molecules(agent))
        agents = dict(zip(agents.keys(), numpy_agents))
        states = deserialize_value(serialized_state)
        states['agents'] = agents
        return states
    
    deserialized_states = deserialize_value(serialized_state)
    states = numpy_molecules(deserialized_states)
    # TODO: Add timeline process to set up media ID
    states.setdefault("environment", {})["media_id"] = "minimal"
    return states
