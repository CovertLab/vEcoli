"""
This folder is used to store frequently used JSON configuration files for
:py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim`.
"""

import copy
import json
import os

from vivarium.library.dict_utils import deep_merge

from ecoli.processes import process_registry
from ecoli.processes.registries import topology_registry

CONFIG_DIR_PATH = os.path.dirname(os.path.abspath(__file__)) + "/"

# Load default processes and topology
with open(os.path.join(CONFIG_DIR_PATH, "default.json")) as default_file:
    config = json.load(default_file)

processes = config["processes"]
topology = config["topology"]

# Compute effect of add_processes, exclude_processes, swap_processes
add_processes = config["add_processes"]
exclude_processes = config["exclude_processes"]
swap_processes = config["swap_processes"]

ECOLI_DEFAULT_PROCESSES = {}
"""
At runtime, the process classes corresponding to the process names 
listed under the ``processes`` key of the default configuration 
(``default.json``) are retrieved from the process registry (see 
``ecoli/processes/__init__.py``) and cached in this dictionary.

.. note::
    If the default configuration includes non-empty ``swap_processes`` 
    or ``exclude_processes`` fields, the specififed swaps/exclusions 
    are performed during the construction of this dictionary.
"""
for process_name in processes + list(add_processes):
    if process_name in exclude_processes:
        continue

    if process_name in swap_processes:
        process_name = swap_processes[process_name]

    process_class = process_registry.access(process_name)

    if not process_class:
        raise ValueError(
            f"Unknown process with name {process_name}. "
            "Did you call process_registry.register() in ecoli/processes/__init__.py?"
        )

    ECOLI_DEFAULT_PROCESSES[process_name] = process_class


ECOLI_DEFAULT_TOPOLOGY = {}
"""
At runtime, the topologies for the processes in 
:py:data:`~configs.ECOLI_DEFAULT_PROCESSES` 
are retrieved from the topology registry (most processes in ``ecoli/processes`` 
register their topology near the top of their source files). 

.. note::
    The topologies and overrides for swapped processes are handled as described 
    in :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim._retrieve_topology`.
"""

original_processes = {v: k for k, v in swap_processes.items()}
for process in ECOLI_DEFAULT_PROCESSES:
    # Start from default topology if it exists
    original_process = (
        process
        if process not in swap_processes.values()
        else original_processes[process]
    )

    process_topology = topology_registry.access(original_process)
    if process_topology:
        process_topology = copy.deepcopy(process_topology)
    else:
        process_topology = {}

    # Allow the user to override default topology
    if original_process in topology.keys():
        deep_merge(
            process_topology,
            {k: tuple(v) for k, v in topology[original_process].items()},
        )

    # For swapped processes, do additional overrides if they are provided
    if process != original_process and process in topology.keys():
        deep_merge(
            process_topology, {k: tuple(v) for k, v in topology[process].items()}
        )

    ECOLI_DEFAULT_TOPOLOGY[process] = process_topology
