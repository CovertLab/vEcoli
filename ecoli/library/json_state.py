import ast
import json
import numpy as np
import concurrent.futures

from vivarium.core.serialize import deserialize_value
from wholecell.utils import units


def load_states(path):
    with open(path, "r") as states_file:
        states = json.load(states_file)
    return states


def numpy_molecules(states):
    """
    Loads unique and bulk molecule data as Numpy structured arrays
    """
    if "bulk_dtypes" in states:
        bulk_dtypes = ast.literal_eval(states.pop("bulk_dtypes"))
        bulk_tuples = [tuple(mol) for mol in states["bulk"]]
        states["bulk"] = np.array(bulk_tuples, dtype=bulk_dtypes)
        # Numpy arrays are read-only outside of updater
        states["bulk"].flags.writeable = False
    if "unique_dtypes" in states:
        for key, dtypes in states.pop("unique_dtypes").items():
            dtypes = ast.literal_eval(dtypes)
            unique_tuples = [tuple(mol) for mol in states["unique"][key]]
            states["unique"][key] = np.array(unique_tuples, dtype=dtypes)
            states["unique"][key].flags.writeable = False
    if "environment" in states:
        if "exchange_data" in states["environment"]:
            states["environment"]["exchange_data"]["constrained"] = {
                mol: units.mmol / (units.g * units.h) * rate
                for mol, rate in states["environment"]["exchange_data"][
                    "constrained"
                ].items()
            }
        else:
            # Load aerobic minimal media exchange data by default
            states["environment"]["exchange_data"] = {
                "unconstrained": sorted(
                    [
                        "CL-[p]",
                        "FE+2[p]",
                        "FE+3[p]",
                        "CO+2[p]",
                        "MG+2[p]",
                        "NA+[p]",
                        "CARBON-DIOXIDE[p]",
                        "OXYGEN-MOLECULE[p]",
                        "MN+2[p]",
                        "L-SELENOCYSTEINE[c]",
                        "K+[p]",
                        "SULFATE[p]",
                        "ZN+2[p]",
                        "CA+2[p]",
                        "Pi[p]",
                        "NI+2[p]",
                        "WATER[p]",
                        "AMMONIUM[c]",
                    ]
                ),
                "constrained": {"GLC[p]": 20.0 * units.mmol / (units.g * units.h)},
            }
    return states


def get_state_from_file(
    path="data/wcecoli_t0.json",
):
    serialized_state = load_states(path)
    # Parallelize deserialization of colony states
    if "agents" in serialized_state:
        agents = serialized_state.pop("agents")
        n_agents = len(agents)
        with concurrent.futures.ProcessPoolExecutor(n_agents) as executor:
            deserialized_agents = executor.map(deserialize_value, agents.values())
        numpy_agents = []
        for agent in deserialized_agents:
            numpy_agents.append(numpy_molecules(agent))
        agents = dict(zip(agents.keys(), numpy_agents))
        states = deserialize_value(serialized_state)
        states["agents"] = agents
        return states

    deserialized_states = deserialize_value(serialized_state)
    states = numpy_molecules(deserialized_states)
    # TODO: Add timeline process to set up media ID
    states.setdefault("environment", {})["media_id"] = "minimal"
    return states
