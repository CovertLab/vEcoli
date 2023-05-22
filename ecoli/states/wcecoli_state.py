import ast
import json
import numpy as np
import concurrent.futures

from vivarium.core.serialize import deserialize_value

from wholecell.utils import units


def infinitize(value):
    if value == "__INFINITY__":
        return float("inf")
    else:
        return value


def load_states(path):
    with open(path, "r") as states_file:
        states = json.load(states_file)
    # Apply infinitize() to every value in each agent's environment state
    if 'agents' in states.keys():
        for agent_state in states['agents'].values():
            agent_state['environment'] = {
                key: infinitize(value)
                for key, value in agent_state.get("environment", {}).items()
            }
    else:
        states['environment'] = {
            key: infinitize(value)
            for key, value in states.get("environment", {}).items()
        }
    return states


def colony_initial_state(states):
    """
    colony_initial_state modifies the states of a loaded colony simulation
    to be suitable for initializing a colony simulation.
    """
    for agent_state in states['agents'].values():
        # If evolvers_ran is False, we can get an infinite loop of
        # neither evolvers nor requesters running. No saved state should
        # include evolvers_ran=False.
        assert states.get('evolvers_ran', True)
        agent_state['environment']['exchange_data'] = {
            'unconstrained': {'CL-[p]', 'FE+2[p]', 'CO+2[p]', 'MG+2[p]',
                'NA+[p]', 'CARBON-DIOXIDE[p]', 'OXYGEN-MOLECULE[p]', 'MN+2[p]',
                'L-SELENOCYSTEINE[c]', 'K+[p]', 'SULFATE[p]', 'ZN+2[p]',
                'CA+2[p]', 'Pi[p]', 'NI+2[p]', 'WATER[p]', 'AMMONIUM[c]'},
            'constrained': {
                'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}}
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
            agent.pop('deriver_skips', None)
            numpy_agents.append(numpy_molecules(agent))
        agents = dict(zip(agents.keys(), numpy_agents))
        states = deserialize_value(serialized_state)
        states['agents'] = agents
        return colony_initial_state(states)
    
    deserialized_states = deserialize_value(serialized_state)
    states = numpy_molecules(deserialized_states)
    # If evolvers_ran is False, we can get an infinite loop of
    # neither evolvers nor requesters running. No saved state should
    # include evolvers_ran=False.
    assert states.get('evolvers_ran', True)
    # Shallow copy for processing state into correct form
    initial_state = states.copy()
    # process environment state
    env_data = states.get("environment", {})
    exchange_data = env_data.pop("exchange", {})
    initial_state["environment"] = {
        "media_id": "minimal",
        # TODO(Ryan): pull in environmental amino acid levels
        "amino_acids": {},
        "exchange_data": {
            'unconstrained': {'CL-[p]', 'FE+2[p]', 'CO+2[p]', 'MG+2[p]',
                'NA+[p]', 'CARBON-DIOXIDE[p]', 'OXYGEN-MOLECULE[p]', 'MN+2[p]',
                'L-SELENOCYSTEINE[c]', 'K+[p]', 'SULFATE[p]', 'ZN+2[p]',
                'CA+2[p]', 'Pi[p]', 'NI+2[p]', 'WATER[p]', 'AMMONIUM[c]'},
            'constrained': {
                'GLC[p]': 20.0 * units.mmol / (units.g * units.h)}},
        "external_concentrations": env_data,
        "exchange": exchange_data
    }
    initial_state["process_state"] = {"polypeptide_elongation": {}}
    return initial_state
