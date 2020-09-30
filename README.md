# vivarium-ecoli

## background

This project is a port of https://github.com/CovertLab/wcEcoli to the Vivarium framework: https://github.com/vivarium-collective/vivarium-core

The scope of this project is to migrate the simulation processes, and therefore takes as a starting point the output of wcEcoli's "Parca" - `sim_data`, the large configuration object created by the parameter calculator. For this reason the `reconstruction/` and `wholecell/utils/` folders have been duplicated here as they are necessary to unpickle the serialized `sim_data` object. If a new `sim_data` object is required to be read, the corresponding wcEcoli folders will have to be synchronized.

The only aspects of the wcEcoli project beyond those required to load `sim_data` required for `vivarium-ecoli` are the simulation's process classes. All state handling (previously handled by Bulk- and UniqueMolecules states/containers/views) and the actual running of the simulation (previously `wholecell.sim.Simulation`) are now handled entirely by Vivarium's core engine and process interface. 

The new process classes can be found in `ecoli/processes/*` and a compartment that links them together using a Vivarium topology lives in `ecoli/compartments/ecoli_compartment`. To run the simulation simply invoke

    > python ecoli/compartments/ecoli_compartment.py

## migration

The main motivation behind the migration is to modularize the processes and allow them to be run in new contexts or reconfigured or included in different biological models. 

There are three main aspects to the migration:

* Decoupling each process from `sim_data` so they can be instantiated and run independently of the assumptions made in the structure of the `sim_data` object. This allows each process to be reconfigured into new models as well as ensuring each process has the ability to run on its own. Each process now has parameters declared for each class that can be provided at initialization time. 

* Refactoring access to `Bulk-` and `UniqueMolecules` into the `ports_schema()` declaration used by Vivarium. This turns each process's state interactions into explicit declarations of properties and attributes. These declarations allow the processes to be recombined and to be expanded with new properties as needed. It also frees the processes from having to conform to the `Bulk/Unique` dichotomy and provides a means for other kinds of state (previously implemented by reading from listeners or accessing other process class's attributes), further promoting decoupling. This declared schema is then used as a structure to provide the current state of the simulation to each process's `next_update(timestep, states)` function during the simulation.

* Translating all state mutation from inside the processes into the construction of a Vivarium `update` dictionary which is returned from the `next_update(timestep, states)`. The structure of this update mirrors the provided `states` dictionary and the declared `ports_schema()` structure. This allows each process to operate knowing that values will not be mutated by other processes before it sees them, a fundamental requirement for modular components.

The way Vivarium state updates work all states are provided as they appear at the beginning of each process's update timestep, and then are applied at the end of that timestep. This ensures all state is synchronized between processes.

