import json
from ecoli.processes import process_registry
from ecoli.processes.registries import topology_registry

CONFIG_DIR_PATH = 'ecoli/composites/ecoli_configs/'

# Load default processes and topology
with open(CONFIG_DIR_PATH + 'default.json') as default_file:
    config = json.load(default_file)

processes = config['processes']
topology = config['topology']

# Compute effect of add_processes, exclude_processes, swap_processes
add_processes = config['add_processes']
exclude_processes = config['exclude_processes']
swap_processes = config['swap_processes']

ECOLI_DEFAULT_PROCESSES = {}
for process_name in processes + list(add_processes):
    if process_name in exclude_processes:
        continue

    if process_name in swap_processes:
        process_name = swap_processes[process_name]

    process_class = process_registry.access(process_name)

    if not process_class:
        raise ValueError(f"Unknown process with name {process_name}. "
                            "Did you call process_registry.register() in ecoli/processes/__init__.py?")

    ECOLI_DEFAULT_PROCESSES[process_name] = process_class


ECOLI_DEFAULT_TOPOLOGY = {}

original_processes = {v: k for k, v in swap_processes.items()}
for process in ECOLI_DEFAULT_PROCESSES:
    # Start from default topology if it exists
    original_process = (process
                        if process not in swap_processes.values()
                        else original_processes[process])

    process_topology = topology_registry.access(original_process)
    if process_topology:
        process_topology = dict(process_topology)
    else:
        process_topology = {}

    # Allow the user to override default topology
    if original_process in topology.keys():
        deep_merge(process_topology, {k: tuple(v)
                    for k, v in topology[original_process].items()})

    # For swapped processes, do additional overrides if they are provided
    if process != original_process and process in topology.keys():
        deep_merge(process_topology, {k: tuple(v)
                    for k, v in topology[process].items()})

    ECOLI_DEFAULT_TOPOLOGY[process] = process_topology

# Add log_update ports if log_updates is True
if config['log_updates']:
    for process, ports in result.items():
        result[process]['log_update'] = ('log_update', process,)

# add division
if config['divide']:
    ECOLI_DEFAULT_TOPOLOGY['division'] = {
        'variable': ('listeners', 'mass', 'cell_mass'),
        'agents': self.agents_path}
