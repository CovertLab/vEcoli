# Configuring Vivarium-ecoli Simulations

Simulations of the Ecoli composite can be configured using the command-line interface (CLI), `.json` configuration files, or programmatically through an object-oriented interface. JSON configuration files live in `data/ecoli_master_configs`.

Configuration through the CLI affords easy access to the most important settings, whereas the programmatic and JSON approaches offer access to all settings.

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
                           [--log_updates] [--raw_output]
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

Processes to be used in a simulation are listed under the `"processes"` key, and these are wired to stores as specified with the `"topology"` key (or default topology in the `topology_registry`, see note below). One can configure a custom set of processes with custom topology using by overriding the values for `"processes"` and `"topology"` in `default.json`.

```{json}
{
    "processes": [
        "ecoli-tf-binding",
        "ecoli-transcript-initiation",
        ...
        "ecoli-mass"
    ],
    "topology": {
    }
}
```

> ***Note: Topology Registry***
>
> The topology key in `default.json` is actually empty, because default topologies come from the `topology_registry` (in `ecoli.processes.registries`). This is essentially a dictionary which associates the name of a process with its typical topology. Canonical processes register their default topology towards the top of the file.

However, typically one wishes to modify these only slightly, e.g. by adding a process, removing a process, or swapping a process for an alternative version of itself. As such, the following keys can be used to modify the processes as declared in `default.json`:

- `"add_processes"` : List of processes to add to the simulation
- `"exclude_processes"` : List of processes to remove from the simulation
- `"swap_processes"` : Dictionary where keys are processes in the default configuration, and values are processes to replace these with.

> ***Note: Adding Processes***
>
> In order for `EcoliSimulation` to use new or alternative processes, a few requirements need to be met. First, the new process should have its `.name` set (to something that does not conflict with existing processes). Second, the new process needs to be *registered* with the *process registry* (do this in `ecoli/processes/__init__.py`). Finally, one needs to specify the configuration of this process using the `"process_configs"` key (see below). If not specified, vivarium-ecoli will default to trying to load the process configuration from `sim_data` (see `LoadSimData.get_config_by_name`).

Adding a process requires adding a corresponding topology to the topology dictionary. Luckily, due to the way EcoliSimulation merges user settings with the default, one can simply specify topology of added processes without restating topology of processes kept from the default. For example:

```
{
    "add_processes" : ["clock"],
    "topology" : {
        "clock" : {
            "global_time" : ["global_time"]
        }
    },
    "process_configs" :{
        "clock" : {
            "time_step" : 2.0,
        }
    }
}
```

would add a `vivarium.processes.clock.Clock` process to the default configuration, without affecting the wiring of other processes.

Note that we used the `"process_configs"` key to initialize the `Clock` process with a timestep of 2. Besides providing an explicit configuration as we did above, we could also have written

```
"process_configs" :{
    "clock" : "default"
}
```

to use the default configuration for `Clock`, or 

```
"process_configs" :{
    "clock" : "sim_data"
}
```

to attempt to load a configuration for `Clock` from sim_data using `LoadSimData.get_config_by_name()` (which would fail in this case). Attempting to load from sim_data is the default behavior if a process config is not specified. 

> ***Note:*** when specifying an explicit process config, as in the first case where we set the timestep to 2, this explicit override actually gets deep-merged with (a) the config from sim_data, if it exists, or (b) the default process config, if it does not. This allows one to override only specific entries in the config.

Removing a process removes the corresponding topology entry automatically, and swapping a process keeps the same topology as the original process (unless overridden by the user).

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

- `"partition"` : (boolean) whether to use partitioning model (NOT YET IMPLEMENTED)
- `"description"` : (string) description of the experiment
- `"suffix_time"` : (boolean) whether to suffix custom experiment IDs with time of simulation, to avoid conflict in the database
- `"progress_bar"` : (boolean) whether to show the progress bar
- `"agent_id"`
- `"parallel"`
- `"daughter_path"`
- `"agents_path"`
- `"division"`
- `"divide"`

## Programmatic Interface

Running simulations within code, one should use the `EcoliSim` class from `ecoli.experiments.ecoli_master_sim`. This class represents a simulation of the whole-cell *E. coli* model, along with its settings. 

```
# Make simulation with default.json
sim = EcoliSim.from_file()  # Can also pass in a path to JSON config

# Modify simulation settings
sim.experiment_id = "Demo"
sim.total_time = 10
...

data_out = sim.run()
```

All of the settings available to be modified from JSON are also accessible as fields of the `EcoliSim` object. If at any point you wish to access the full simulation config, `sim.config` offers an up-to-date configuration including all changes made through this OOP interface.

After running a simulation, the `Ecoli` composite generated can be accessed with `sim.ecoli`.