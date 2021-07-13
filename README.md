# vivarium-ecoli


## pyenv

Install Python using pyenv. pyenv lets you install and switch between multiple Python releases and multiple "virtual environments", each with its own pip packages.

PYTHON_CONFIGURE_OPTS="--enable-shared" pyenv install 3.8.5


## pip version 20.0.2 is required for the current multiple setup in setup.py
TODO: fix this to support most recent pip!

```
pip install pip==20.0.2
```

## set up

First install numpy and Cython:
```
$ pip install numpy
$ pip install Cython
```

Then the remaining requirements:

```
$ python setup.py install
```

## background

This project is a port of https://github.com/CovertLab/wcEcoli to the Vivarium framework: https://github.com/vivarium-collective/vivarium-core

The scope of this project is to migrate the simulation processes, and therefore takes as a starting point the output of wcEcoli's "Parca" - `sim_data`, the large configuration object created by the parameter calculator. For this reason the `reconstruction/` and `wholecell/utils/` folders have been duplicated here as they are necessary to unpickle the serialized `sim_data` object. If a new `sim_data` object is required to be read, the corresponding wcEcoli folders will have to be synchronized.

The only aspects of the wcEcoli project beyond those required to load `sim_data` required for `vivarium-ecoli` are the simulation's process classes. All state handling (previously handled by Bulk- and UniqueMolecules states/containers/views) and the actual running of the simulation (previously `wholecell.sim.Simulation`) are now handled entirely by Vivarium's core engine and process interface. 

The new process classes can be found in `ecoli/processes/*` and a composites that links them together using a Vivarium topology lives in `ecoli/composites/ecoli_master`. To run the simulation first compile the `wholecell/utils` cython files:

    > make clean compile

then simply invoke

    > python ecoli/composites/ecoli_master.py

## migration

The main motivation behind the migration is to modularize the processes and allow them to be run in new contexts or reconfigured or included in different biological models. 

There are three main aspects to the migration:

* Decoupling each process from `sim_data` so they can be instantiated and run independently of the assumptions made in the structure of the `sim_data` object. This allows each process to be reconfigured into new models as well as ensuring each process has the ability to run on its own. Each process now has parameters declared for each class that can be provided at initialization time. 

* Refactoring access to `Bulk-` and `UniqueMolecules` into the `ports_schema()` declaration used by Vivarium. This turns each process's state interactions into explicit declarations of properties and attributes. These declarations allow the processes to be recombined and to be expanded with new properties as needed. It also frees the processes from having to conform to the `Bulk/Unique` dichotomy and provides a means for other kinds of state (previously implemented by reading from listeners or accessing other process class's attributes), further promoting decoupling. This declared schema is then used as a structure to provide the current state of the simulation to each process's `next_update(timestep, states)` function during the simulation.

* Translating all state mutation from inside the processes into the construction of a Vivarium `update` dictionary which is returned from the `next_update(timestep, states)`. The structure of this update mirrors the provided `states` dictionary and the declared `ports_schema()` structure. This allows each process to operate knowing that values will not be mutated by other processes before it sees them, a fundamental requirement for modular components.

The way Vivarium state updates work all states are provided as they appear at the beginning of each process's update timestep, and then are applied at the end of that timestep. This ensures all state is synchronized between processes.

All effort has been made to translate these processes as faithfully as possible. This means previous behavior is intact, and even ports are kept the same (`cell_mass` is still under a `listeners/mass` port instead of something more suitable like a `global` port). All listener output is still emitted, though listeners have been renamed from CamelCase to under_score, which will need to be addressed to get the existing analyses working. 

## current state

As of June 2020, 
The sim_data is generated with wcEcoli branch [vivarium-ecoli-52021](https://github.com/CovertLab/wcEcoli/tree/vivarium-ecoli-52021)

Remaining are:

* ChromosomeStructure - These are substantial and deal mostly with the chromosome, but don't present any challenge that the other processes didn't already provide. They use UniqueMolecules extensively so other processes like `transcript_initiation.py` may act as a good reference for migration.

* CellDivision - there is a branch (`division`) with a WIP on this. That said, cell division works differently in Vivarium from wcEcoli, so this will need to be a more substantial change. In wcEcoli, the simulation is ended on division and two files with daughter states are written to the filesystem, which need to be started again separately. In Vivarium, the simulation can continue running on division, simulating each daughter in a new compartment in a shared environment. This will need to be addressed.

In order to finish this migration, here are a few tips for migrating a process from wcEcoli to Vivarium:

* Copy the process file over from `wcEcoli/model/ecoli/processes/*` into `vivarium-ecoli/ecoli/processes/*` and convert all the tabs to spaces, also importing `from vivarium.core.process import Process` instead of the `wholecell.processes.process` version.
* In `ecoli/composites/ecoli_master.py`, create an `initialize_*` function taking `sim_data` to create the configuration for the process. 
* In the process file, extract from `initialize` any reference to `sim_data` or `sim._*` and create an entry in the config dictionary for it pointing to this path in `sim_data` (or a default for the `sim._*` option).
* Collapse `__init__` and `initialize`, then convert `calculateRequest` and `evolveState` into a single function `next_update(self, timestep, states)`.  
* In the process file, declare the `defaults` parameters dictionary and make each place that refers to `sim_data`
in `__init__` refer to these new parameters instead.
* Anywhere that creates a `BulkMoleculesView` or `UniqueMoleculesView` add an entry in the new process's `ports_schema()` instead. If they are a `UniqueMoleculesView`, declare any attributes that are referred to in the process's `next_update`.
* Go through `next_update` and fix any references to views to read from the `states` dictionary passed to `next_update`.
* Anywhere that a "request" is made, note this value to use instead of the value from `states` (the request is the state `evolveState` expected to see).
* Create an `update` dictionary to hold state updates. Anywhere the view is mutated, construct an entry in the `update` dictionary instead.
* Anywhere a listener is referred to, create an entry in `ports_schema` instead and add that value to the growing `update` dictionary.
* At the end, return the `update` dictionary.
* Put a trace right before return `update` from `next_update` and compare it to the values you expect from wcEcoli.

## remaining considerations

A few things remain:

* Calculating mass - some effort went into ensuring all the unique molecules that track submass still do so, but the overall calculation of mass was done in a listener and will need to be replicated as a process here. Also, the masses will have to be added as attributes to all the values in the `bulk` and `unique` stores to correspond to the mass for each molecule. 
* The collapse of `calculateRequest` and `evolveState` means partitioning no longer occurs. This may create a deviation in behavior and may need to be addressed.
* The processes were migrated but not entirely validated. Overall model behavior remains to be validated.
* In order to get analyses to run again, an adapter for the previous `TableReader` will have to read from the database instead. I made the (possibly misguided) decision to convert the CamelCase listener names to under_score, so either they need to be switched back or updated in the analyses, or simpler the conversion made in the new `TableDBReader` interface. 