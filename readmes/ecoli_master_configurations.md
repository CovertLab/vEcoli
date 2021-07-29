# Configuring Vivarium-ecoli Simulations

Simulations of the Ecoli composite can be configured using the command-line interface (CLI), or through `.json` configuration files. These configuration files live in `data/ecoli_master_configs`.

Configuration through the CLI affords easy access to the most important settings, whereas the JSON approach offers access to all settings.

Default settings, which are overridden by the CLI and custom JSONs, may be found in `data/ecoli_master_configs/default.json`. This file should be edited with care if at all, since missing defaults may result in an error.


## CLI

Running a simulation through the CLI gives access to most commonly used settings, which are summarized below.

For the most up-to-date documentation, run `ecoli_master_sim.py` with the help (`-h`) flag:

```
python ecoli/experiments/ecoli_master_sim.py -h
```

```
usage: ecoli_master_sim.py [-h] [--config CONFIG] [--experiment_id EXPERIMENT_ID] [--emitter {timeseries,database,print}]
                           [--seed SEED] [--initial_time INITIAL_TIME] [--total_time TOTAL_TIME] [--generations GENERATIONS]
                           [--log_updates] [--timeseries]
                           [path_to_sim_data [path_to_sim_data ...]]
```

### Positional arguments

- `path_to_sim_data`: Path to the sim_data to use for this experiment.

### Optional arguments

- `--config CONFIG`, `-c CONFIG`: path to a config file (`.json`) to use. Defaults to `data/ecoli_master_configs/default.json`. Settings from this file can be overridden with further command-line arguments.
- `--experiment_id EXPERIMENT_ID`, `-id EXPERIMENT_ID`: ID for this experiment. A UUID will be generated if this argument is not used and `"experiment_id"` is `null` in the configuration file.
- `--emitter`, `-e`: Emitter to use. Options are `timeseries`, `database`, and `print`. Timeseries uses RAMEmitter, database emits to MongoDB, and print emits to stdout.
- `--seed SEED`, `-s SEED`  Random seed.
- `--initial_time INITIAL_TIME`, `-t0 INITIAL_TIME`: Time of the initial state to load from (corresponding inital state file must be present in `data` folder).
- `--total_time TOTAL_TIME`, `-t TOTAL_TIME`: Time to run the simulation for.
- `--generations GENERATIONS`, `-g GENERATIONS`: Number of generations to run the simulation for.
- `--log_updates`, `-u`: Save updates from each process if this flag is set, e.g. for use with blame plot.
- `--raw_output`: Whether to return data in raw format (dictionary where keys are times, values are states). Some analyses may not work with this flag set.

## JSON

Configuring a simulation from JSON allows access much more complete access to simulation settings. This includes the ability to add/remove/swap processes, modify topology, and even change how individual molecules are updated or emitted (see **Schema Overrides**).

Not all settings need to be provided in a custom configuration file. Settings not given default to values from `data/ecoli_master_configs/default.json`.

### Basic Settings

The following settings correspond exactly to the options available through the CLI (described above).

- `"experiment_id"`
- `"sim_data_path"`
- `"emitter"`
- `"log_updates"`
- `"raw_output"`
- `"seed"`
- `"initial_time"`
- `"time_step"`
- `"total_time"`
- `"generations"`


### Processes and Topology

Processes to be used in a simulation are listed under the `"processes"` key, and these are wired to stores as specified with the `"topology"` key. One can configure a custom set of processes with custom topology using by overriding the values for `"processes"` and `"topology"` in `default.json`.

```{json}
{
    "processes": [
        "ecoli-tf-binding",
        "ecoli-transcript-initiation",
        ...
        "ecoli-mass"
    ],
    "topology": {
        "ecoli-tf-binding": {
            "promoters": ["unique", "promoter"],
            "active_tfs": ["bulk"],
            "inactive_tfs": ["bulk"],
            "listeners": ["listeners"]
        },
        "ecoli-transcript-initiation": {
            "environment": ["environment"],
            "full_chromosomes": ["unique", "full_chromosome"],
            "RNAs": ["unique", "RNA"],
            "active_RNAPs": ["unique", "active_RNAP"],
            "promoters": ["unique", "promoter"],
            "molecules": ["bulk"],
            "listeners": ["listeners"]
        },
        ...
        "ecoli-mass": {
            "bulk": ["bulk"],
            "unique": ["unique"],
            "listeners": ["listeners"]
        }
    }
}
```

However, typically one wishes to modify these only slightly, e.g. by adding a process, removing a process, or swapping a process for an alternative version of itself. As such, the following keys can be used to modify the processes as declared in `default.json`:

- `"add_processes"` : List of processes to add to the simulation
- `"exclude_processes"` : List of processes to remove from the simulation
- `"swap_processes"` : Dictionary where keys are processes in the default configuration, and values are processes to replace these with.

> ***Note:*** In order for `EcoliSimulation` to use new or alternative processes, two requirements need to be met. First, the new process should have its `.name` set (to something that does not conflict with existing processes). Second, the new process needs to be *registered* with the *process registry* (do this in `ecoli/processes/__init__.py`).

Adding a process requires adding a corresponding topology to the topology dictionary. Luckily, due to the way EcoliSimulation merges user settings with the default, one can simply specify topology of added processes without restating topology of processes kept from the default (check this??). For example:

```
{
    "add_processes" : [
        "clock"
    ],
    
    "topology" : [
        "clock" : {
            "global_time" : ["global_time"]
        }
    ]
}
```

would add a `Clock` process to the default configuration, without affecting the wiring of other processes.

Removing a process removes the corresponding topology entry automatically, and swapping a process keeps the same topology as the original process (unless overridden by the user) (check this).

### Schema Overrides

One powerful feature of the JSON configuration approach is the ability to override the port schemas specified by processes. To do so, one simply adds a `"_schema"` key to the configuration. In the following example, we have overwritten the schema for how the `"ecoli-equilibrium"` process affects the `"PD00413[c]"` molecule, in this case to temporarily avert a bug in which `"PD00413[c]"` goes below zero.

```
"_schema": {
        "ecoli-equilibrium": {
            "molecules": {
                "PD00413[c]": {"_updater": "nonnegative_accumulate"}
            }
        }
    },
```

Schema overrides can also be used to emit data that would normally not be emitted, by setting `"_emit"` to `True`.

```
"_schema": {
        "ecoli-equilibrium": {
            "molecules": {
                "PD00413[c]": {"_emit": True}
            }
        }
    },
```

### Additional Settings

- `"agent_id"`
- `"parallel"`
- `"daughter_path"`
- `"agents_path"`
- `"division"`
- `"divide"`