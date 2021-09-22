# Vivarium-Ecoli

![vivarium](doc/_static/ecoli_master_topology.png)

## Background

Vivarium-ecoli project is a port of the Covert Lab's 
[E. coli Whole Cell Model](https://github.com/CovertLab/wcEcoli) 
to the [Vivarium framework](https://github.com/vivarium-collective/vivarium-core).

The scope of this project is to migrate the Whole Cell Model's processes, and therefore takes 
wcEcoli's `sim_data` as its starting point in the simulation pipeline.
`sim_data` is a large configuration object created by the parameter calculator (ParCa). 
For this reason the `reconstruction/` and `wholecell/utils/` folders have been duplicated 
here as they are necessary to unpickle the serialized `sim_data` object. If a new `sim_data` 
object is required to be read, the corresponding wcEcoli folders will have to be synchronized.

All state handling (previously handled by Bulk- and UniqueMolecules states/containers/views) 
and the actual running of the simulation (previously `wholecell.sim.Simulation`) are now 
handled entirely by Vivarium's core engine and process interface. 

The new process classes can be found in `ecoli/processes/*` and are linked together using 
a Vivarium topology that is generated in `ecoli/experiments/ecoli_master_sim.py`.

## Pyenv Installation

pyenv lets you install and switch between multiple Python releases and multiple "virtual 
environments", each with its own pip packages. Using pyenv, create a virtual environment 
and install Python 3.8.5. For a tutorial on how to create a virtual environment, follow 
the instructions [here](https://github.com/CovertLab/wcEcoli/blob/master/docs/create-pyenv.md) 
and stop once you reach the "Create the 'wcEcoli3' python virtual environment" step. Then, 
run the following command in your terminal:

    $ pyenv virtualenv 3.8.5 viv-ecoli && pyenv local viv-ecoli

#TODO: Create a pyproject.toml so the project can install with a single pip command (see the [PEP 517/518 solution](https://stackoverflow.com/questions/54117786/add-numpy-get-include-argument-to-setuptools-without-preinstalled-numpy))
Now, install numpy and Cython:

    $ pip install numpy
    $ pip install Cython

Then install the remaining requirements:

    $ pip install -r requirements.txt 
    $ python setup.py install

## Conda install

To install using conda (first line creates a conda environment with numpy and Cython, the second activates the environment and the third installs WCM and remaining dependencies).

    $ conda create -n WCM numpy Cython
    $ conda activate WCM
    $ conda install --file requirements.txt
    $ python setup.py install

## Running the simulation

To run the simulation, simply invoke:

    $ python ecoli/experiments/ecoli_master_sim.py

For details on configuring simulations through either the command-line interface or .json files, 
see the [Ecoli-master configurations readme](readmes/ecoli_configurations.md).

## wcEcoli Migration

The main motivation behind the migration is to modularize the processes and allow them to be run 
in new contexts or reconfigured or included in different biological models. 

There are three main aspects to the migration:

* Decoupling each process from `sim_data` so they can be instantiated and run independently of 
the assumptions made in the structure of the `sim_data` object. This allows each process to be 
reconfigured into new models as well as ensuring each process has the ability to run on its own. 
Each process now has parameters declared for each class that can be provided at initialization time. 

* Refactoring access to `Bulk-` and `UniqueMolecules` into the `ports_schema()` declaration used 
by Vivarium. This turns each process's state interactions into explicit declarations of properties 
and attributes. These declarations allow the processes to be recombined and to be expanded with 
new properties as needed. It also frees the processes from having to conform to the `Bulk/Unique` 
dichotomy and provides a means for other kinds of state (previously implemented by reading from 
listeners or accessing other process class's attributes), further promoting decoupling. This declared 
schema is then used as a structure to provide the current state of the simulation to each process's 
`next_update(timestep, states)` function during the simulation.

* Translating all state mutation from inside the processes into the construction of a Vivarium 
`update` dictionary which is returned from the `next_update(timestep, states)`. The structure of 
this update mirrors the provided `states` dictionary and the declared `ports_schema()` structure. 
This allows each process to operate knowing that values will not be mutated by other processes before 
it sees them, a fundamental requirement for modular components.

The way Vivarium state updates work, all states are provided as they appear at the beginning of each 
process's update timestep, and then are applied at the end of that timestep. This ensures all states 
are synchronized between processes.

## Causality Network

After running a simulation, you can explore the Causality visualization tool (see 
[CovertLab/causality](https://github.com/CovertLab/causality)) to examine the model's causal links and 
simulation output correlations.

## Current state

As of September 2021, 
The sim_data is generated with wcEcoli branch [vivarium-ecoli-52021](https://github.com/CovertLab/wcEcoli/tree/vivarium-ecoli-52021)

All effort has been made to translate these processes as faithfully as possible. This means previous 
behavior is intact, including the partitioning assumption from the original model. 
