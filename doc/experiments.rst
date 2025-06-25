===========
Experiments
===========

:py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` is the primary
interface for configuring and running single-cell simulations. We refer
to simulations as experiments, and all simulations (or batches
of simulations run in a single workflow, see :ref:`/workflows.rst`) are
identified via a unique experiment ID. 

.. warning::
    If data is being persisted to disk (see :ref:`parquet_emitter`), simulations
    or workflows will overwrite data from any past simulations or workflows with
    the same experiment ID.

When running workflows with :py:mod:`runscripts.workflow` (see :ref:`/workflows.rst`),
users are prevented from accidentally overwriting data by a check that ensures
``{out_dir}/{experiment_id}/nextflow`` does not already exist.


.. _sim_config:

-------------
Configuration
-------------

:py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` offers three methods
for configuring simulations:

#. Using a JSON configuration file via the command-line option ``--config``
#. Using the object-oriented interface
#. Using command-line options

In general, we recommend that you use the JSON configuration interface as much
as possible. This is because the JSON configuration format is standardized across
all of the main interfaces for the model (the scripts in :py:mod:`runscripts`
and :py:mod:`~ecoli.experiments.ecoli_master_sim`). The object-oriented interface
allows users to programatically set simulation options and is mainly intended for
small ad-hoc test simulations or for creating your own experiment file (see
:ref:`create_experiment`). The command-line interface is much more limited than
the other two and only offers access to a few key configuration options. It is
mainly intended for internal use (e.g. in Nextflow workflow scripts).

.. _json_config:

-----------------
JSON Config Files
-----------------

The :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` class relies upon
the helper :py:class:`~ecoli.experiments.ecoli_master_sim.SimConfig` class to load
configuration options from JSON files and merge them with options specified via
the command line. The configuration options are always loaded in the following order,
with options loaded later on overriding those from earlier sources:

#. The options in the default JSON config file (located at
   :py:data:`~ecoli.experiments.ecoli_master_sim.SimConfig.default_config_path`)
#. The options in the JSON config file specified via ``--config``
   in the command line.
#. The options specified via the command line.

In most cases, configuration options that appear in more than one
of the above sources are successively overriden in their entirety. The sole
exceptions are configuration options listed in
:py:attr:`~runscripts.workflow.LIST_KEYS_TO_MERGE`. These
options hold lists of values that are concatenated with one another instead
of being wholly overriden.

Notice that the options in the default JSON config file are always loaded
first. This means that if you would like to run a simulation or workflow
that leaves some of these options alone, you can simply omit those options
from the JSON config file that you create and pass to your runscript of choice
via ``--config``.

Below is an annotated copy of the default simulation-related configuration
options from the default JSON config file (see the file located at
:py:data:`~ecoli.experiments.ecoli_master_sim.SimConfig.default_config_path`
for the most up-to-date defaults). Note that JSON configuration files passed
as input to the scripts in :py:mod:`runscripts` accept additional keys that are
documented in :ref:`/workflows.rst`.

.. code-block::

    {
        # List of JSON filenames in the "configs" directory (include ".json").
        # These files are loaded in order and merged into this configuration.
        # Avoid overly complex inheritance chains if possible.
        "inherit_from": [],
        # String that uniquely identifies simulation (or workflow if passed
        # as input to runscripts/workflow.py). Special characters and spaces
        # are not allowed (hyphens are OK).
        "experiment_id": "experiment_id_one"
        # Whether to append date and time to experiment ID in the following format
        # experiment_id_%Y%m%d-%H%M%S.
        "suffix_time": true,
        # Optional string description of simulation
        "description": "",
        # Whether to display vivarium-core progress bar
        "progress_bar" : true,
        # Path to pickle file output from parameter calculator (runscripts/parca.py).
        # Only used for single sim run with ecoli/experiments/ecoli_master_sim.py.
        # Ignored when run with runscripts/workflow.py because each simulation is
        # automatically run with the appropriate variant/baseline simulation data.
        "sim_data_path": "reconstruction/sim_data/kb/simData.cPickle",
        # Pick between "timeseries" to save simulation output in-memory (good
        # for single-cell ad-hoc analysis) or "parquet" to save output persistently
        # to Parquet files on disk (good for workflows and more in-depth analyses)
        "emitter" : "timeseries",
        # If choosing "parquet" emitter, must provide "out_dir" with path (relative
        # or absolute) to output folder OR "out_uri" with URI for Google Cloud Storage
        # bucket. Only provide one of the above. Other Parquet emitter options are
        # documented under the Parquet Emitter section in the Output page.
        "emitter_arg": {"out_dir": "out"},
        # See API documentation on vivarium-core for vivarium.core.engine.Engine.
        # Can usually leave as false.
        "emit_topology" : false,
        "emit_processes" : false,
        "emit_config" : false,
        # Whether to emit data from all molecules under ("unique",). Should only be
        # used for debugging purposes because this will emit a lot of data. Prefer
        # a dedicated listener to extract unique molecule information at simulation
        # runtime instead.
        "emit_unique": false,
        # Whether to save process updates to log_update stores. Should only be used
        # if choosing "timeseries" emitter. See "Log Updates" heading in "Composites"
        # documentation for more information.
        "log_updates" : false,
        # Controls output format for ecoli.experiments.ecoli_master_sim.EcoliSim.query.
        # Should only be used if choosing "timeseries" emitter. See API documentation
        # for the query function for more information.
        "raw_output" : true,
        # Initial seed used to generate the seeds that are used to initialize
        # the psuedorandom number generators in the model. Only used for single
        # simulations run using ecoli/experiments/ecoli_master_sim.py. Workflows
        # run with runscripts/workflow.py generate initial seeds using the value
        # of a different configuration option named "lineage_seed".
        "seed": 0,
        # Special flags to enable mechanisms related to antibiotic resistance.
        # See API documentation for ecoli.library.sim_data.LoadSimData for more
        # information.
        "mar_regulon": false,
        "amp_lysis": false,
        # String name of file inside "data" folder containing saved JSON initial
        # state (omit .json extension). See "Initialization" headings in "Store"
        # documentation and ecoli.composites.ecoli_master.Ecoli.initial_state
        # documentation for more details.
        "initial_state_file": "",
        # List of string file names inside "data" folder (can be nested like
        # "data/overrides/*") containing manual overrides for targeted values
        # in initial state (whether that initial state came from "initial_state"
        # or "initial_state_file"). Omit .json extension. See API documentation
        # for ecoli.composites.ecoli_master.Ecoli.initial_state.
        "initial_state_overrides": [],
        # Dictionary of values to populate initial state with. Supersedes any file
        # names specified in "initial_state_file". See API documentation
        # for ecoli.composites.ecoli_master.Ecoli.initial_state for more details,
        # including what happens if neither "initial_state" nor "initial_state_file"
        # are provided (as is the case here).
        "initial_state": {},
        # Global time step for all simulation processes. See "Time Step" heading
        # in "Processes" documentation for more details, including extra steps that
        # one must take to add a process with a different time step. MUST BE FLOAT.
        "time_step": 1.0,
        # Maximum time to run simulation for. By default, we only run simulations
        # until reaching division with ecoli/experiments/ecoli_master_sim.py
        # and runscripts/workflow.py. Most of the time, division occurs well before
        # 10800 seconds have elapsed. However, if this is not the case, this time
        # sets a hard stopping point for the simulation. MUST BE FLOAT.
        "max_duration": 10800.0,
        # The value to initialize the ("global_time",) store with. Mainly used for
        # simulations run with runscripts/workflow.py, which frequently entail
        # simulating daughter cells after a mother cell divides. MUST BE FLOAT.
        # Note that the "max_duration" option is applied on top of this value.
        # For example, for an "initial_global_time" of 3000.0 and a "max_duration"
        # of 10000.0, the simulation will have a hard stopping point at 13000.0 s.
        "initial_global_time": 0.0,
        # Whether to raise ecoli.experiments.ecoli_master_sim.TimeLimitError when
        # a simulation reaches the hard stopping point or to gracefully stop with
        # no error raised.
        "fail_at_max_duration": false,
        # String identifier for single cell simulation. For workflows run with
        # runscripts/workflow.py, subsequent generations will append "0" and "1"
        # to this initial agent ID for each daughter cell (only "0" if not
        # simulating both daughter cells, see "Workflow" documentation).
        "agent_id": "0",
        # Whether to add processes and associated topologies for cell
        # division. See "Division Modifications" heading in "Composites" docs.
        "divide": true,
        # Local or absolute path to directory where initial states for daughter
        # cells are saved as JSONs named ``daughter_state_0.json`` and
        # ``daughter_state_1.json``. These can be moved to the ``data``
        # folder and passed as ``initial_state_file`` to run simulations
        # of the daughter cells.
        "daughter_outdir": "out",
        # Whether to add process and associated topology for triggering division
        # after a D period has elapsed following the completion of chromosome
        # replication. If False, division is triggered when the store located
        # at the path for "division_variable" reaches "division_threshold".
        "d_period": true,
        # Threshold that "division_variable" must reach in order for division
        # to be triggered. When "d_period" is True, this must be set to True
        # and "division_variable" must be set to ["divide"] because the
        # ecoli.processes.cell_division.MarkDPeriod process sets the ["divide"]
        # store to True one D period after chromosome replication finishes.
        # To use a mass doubling threshold, "d_period" must be False,
        # "division_variable" must be set to ["listeners", "mass", "dry_mass"],
        # and "division_threshold" must be set to either a hard-coded float
        # (in femtograms) or "mass_distribution". The latter will trigger division
        # after dry mass has increased by an amount dependent on environmental
        # conditions (e.g. no oxygen, basal, with AA, etc.) multiplied by a
        # Gaussian noise factor N(1, 0.1). See ecoli.processes.cell_division.Division.
        "division_threshold": true,
        # Path to store containing value that triggers division upon reaching
        # "division_threshold".
        "division_variable": ["divide"],
        # Path to store containing full chromosome unique molecules. Used by
        # division process to ensure that a cell contains two complete
        # chromosomes before replicating (can occur when "d_period" is False
        # and "division_variable" is cell mass for example). Will wait for
        # there to be two complete full chromosomes before dividing even
        # if "division_variable" hits "division_threshold".
        "chromosome_path": ["unique", "full_chromosome"],
        # Whether to simulate cell inside a binned 2D spatial environment
        # with support for reaction diffusion. See API documentation for
        # ecoli.composites.environment.lattice.Lattice composite. This is
        # mainly useful for colony simulations.
        "spatial_environment": false,
        # Configuration options for Lattice composite. See the JSON config
        # file at configs/spatial.json for an example.
        "spatial_environment_config": {},
        # Whether to serialize the simulation state to JSON and save it to
        # files at the times listed in "save_times". See the API documentation
        # for ecoli.experiments.ecoli_master_sim.EcoliSim.save_states. This can
        # be useful to save and reload the simulation at a certain time for
        # debugging purposes.
        "save": false,
        "save_times": [],
        # List of process names to add to model on top of defaults.
        "add_processes" : [],
        # List of process names to remove from defaults (or processes added
        # by other JSONs in the "inherit_from" hierarchy).
        "exclude_processes" : [],
        # Mapping of process names to names of processes to replace them with.
        # For example, {"ecoli-metabolism" : "ecoli-metabolism-redux-classic"}
        # replaces the default metabolism process with one registered in
        # ecoli/processes/__init__.py as "ecoli-metabolism-redux-classic"
        "swap_processes" : {},
        # Whether to print profiling statistics for simulation run.
        # TODO: Check whether this still works.
        "profile": false,
        # List of names of processes to include in model. The blank lines between
        # process names here indicate the boundaries between successive execution
        # layers as described in the "Steps and Flows" sub-heading in the "Stores"
        # documentation (with the exception of "global_clock" which inherits from
        # Process and not Step). You can verify that this is the case by working
        # through the dependencies in the "flow" below.
        "processes": [
            "post-division-mass-listener", # Run and apply update

            "bulk-timeline", # Once layer above finishes, run and
            "media_update", # apply updates in arbitrary order
            "exchange_data",

            "ecoli-tf-unbinding", # Once layer above finishes, run and update

            "ecoli-equilibrium", # Once layer above finishes, run Requesters,
            "ecoli-two-component-system", # then Allocator, then Evolvers,
            "ecoli-rna-maturation", # then UniqueUpdate (see "Partitioning")

            "ecoli-tf-binding",

            "ecoli-transcript-initiation",
            "ecoli-polypeptide-initiation",
            "ecoli-chromosome-replication",
            "ecoli-protein-degradation",
            "ecoli-rna-degradation",
            "ecoli-complexation",

            "ecoli-transcript-elongation",
            "ecoli-polypeptide-elongation",

            "ecoli-chromosome-structure",

            "ecoli-metabolism",

            "ecoli-mass-listener",
            "RNA_counts_listener",
            "rna_synth_prob_listener",
            "monomer_counts_listener",
            "dna_supercoiling_listener",
            "replication_data_listener",
            "rnap_data_listener",
            "unique_molecule_counts",
            "ribosome_data_listener",
            
            "global_clock"
        ],
        # Mapping of process names to dictionaries of parameters to override
        # defaults with, if any. Processes that do not have a registered
        # function in ecoli.library.sim_data.LoadSimData.get_config_by_name
        # MUST specify either "default" or a dictionary of parameters here.
        # See ecoli.composites.ecoli_master.Ecoli.generate_processes_and_steps
        # for more details.
        "process_configs": {
            "global_clock": {},
            "replication_data_listener": {"time_step": 1}
        },
        # Mapping of process names to topology dictionaries. Processes that
        # did not register their topology in ecoli.processes.registry.topology_registry
        # by importing it and calling topology_registry.register(NAME, TOPOLOGY)
        # MUST specify a topology dictionary here.
        "topology": {
            "bulk-timeline": {
                "bulk": ["bulk"],
                "global": ["timeline"],
                "media_id": ["environment", "media_id"]
            },
            "global_clock": {
                "global_time": ["global_time"],
                "next_update_time": ["next_update_time"]
            }
        },
        # Mapping of Step names to paths to Step dependencies. See the
        # "Steps and Flows" sub-heading in the "Stores" documentation.
        "flow": {
            "post-division-mass-listener": [],
            "media_update": [["post-division-mass-listener"]],
            "exchange_data": [["media_update"]],

            "ecoli-tf-unbinding": [["media_update"]],

            "ecoli-equilibrium": [["ecoli-tf-unbinding"]],
            "ecoli-two-component-system": [["ecoli-tf-unbinding"]],
            "ecoli-rna-maturation": [["ecoli-tf-unbinding"]],

            "ecoli-tf-binding": [["ecoli-equilibrium"]],

            "ecoli-transcript-initiation": [["ecoli-tf-binding"]],
            "ecoli-polypeptide-initiation": [["ecoli-tf-binding"]],
            "ecoli-chromosome-replication": [["ecoli-tf-binding"]],
            "ecoli-protein-degradation": [["ecoli-tf-binding"]],
            "ecoli-rna-degradation": [["ecoli-tf-binding"]],
            "ecoli-complexation": [["ecoli-tf-binding"]],

            "ecoli-transcript-elongation": [["ecoli-complexation"]],
            "ecoli-polypeptide-elongation": [["ecoli-complexation"]],

            "ecoli-chromosome-structure": [["ecoli-polypeptide-elongation"]],

            "ecoli-metabolism": [["ecoli-chromosome-structure"]],

            "ecoli-mass-listener": [["ecoli-metabolism"]],
            "RNA_counts_listener": [["ecoli-metabolism"]],
            "rna_synth_prob_listener": [["ecoli-metabolism"]],
            "monomer_counts_listener": [["ecoli-metabolism"]],
            "dna_supercoiling_listener": [["ecoli-metabolism"]],
            "replication_data_listener": [["ecoli-metabolism"]],
            "rnap_data_listener": [["ecoli-metabolism"]],
            "unique_molecule_counts": [["ecoli-metabolism"]],
            "ribosome_data_listener": [["ecoli-metabolism"]]
        }
    }

Here are some general rules to remember when writing your own JSON config files:

- Strings must be enclosed in double quotes (not single quotes)
- Booleans are lowercase
- None values are written as (unquoted) ``null``
- Trailing commas are not allowed
- Comments are not allowed
- Tuples (e.g. in topologies or flows) are written as lists (``["bulk"]`` instead of ``("bulk",)``)
- ``~`` and environment variables like ``$HOME`` are not expanded (see warning at :doc:`workflows`)

.. note::
    It is strongly recommended that ``fail_at_max_duration`` be set to ``True``
    when running multi-generation workflows. If a simulation reaches max duration
    without dividing, this results in a more informative error message instead
    of a Nextflow error about missing daughter cell states.

------
Output
------

If ``emitter`` was set to ``parquet``, then folders containing the simulation output are
created as described in :ref:`parquet_emitter`.

If ``division`` is set to True, :py:mod:`~ecoli.experiments.ecoli_master_sim` will
save the initial states of the two daughter cells resulting from cell division
in ``daughter_outdir`` as JSON files. These files can be moved to the ``data``
folder and passed as ``initial_state_file`` to simulate the daughter cells.
Additionally, the file ``division_time.sh`` will be created in the folder where
you started the simulation. This script, when run, sets the environment variable
``division_time`` to the time at which the cell divided. It is intended for internal
use when running a simulation workflow with :py:mod:`runscripts.workflow`, allowing
Nextflow to correctly set the ``initial_global_time`` for daughter cell simulations.

----------------
Schema Overrides
----------------

One powerful feature of the JSON configuration approach is the ability to override the port schemas
specified by processes. To do so, one simply adds a ``_schema`` key to the config for a process
under the ``process_configs`` option. In the following example, we have overridden the schema for
how the `"ecoli-mass-listener"` process divides the cell mass.

.. code-block::

    "process_configs": {
        "ecoli-mass-listener": {
            "_schema": {
                "listeners": {
                    "mass": {"cell_mass": {"_divider": "set"}}
                }
            }
        }
    }


Another use of schema overrides is to emit data that would normally not be emitted
by setting ``_emit`` to ``True``.

.. code-block::

    "process_configs": {
        "ecoli-mass-listener": {
            "_schema": {
                "unique": {
                    "active_ribosome": {"_emit": true}
                }
            }
        }
    }

.. warning::
    Vivarium includes internal checks to ensure that all ports connected to a
    store give the same or compatible (no conflicting keys) schemas for that store.
    This means that if you would like to override the schema for a store with many
    connecting ports, you will need to override the schemas for all the relevant ports.

------------------
Colony Simulations
------------------

While :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` was only designed
to handle simulation of single cells in isolation,
:py:mod:`~ecoli.experiments.ecoli_engine_process` was made to simulate
multi-cell colonies in shared, dynamic spatial environments.

Engine Process
==============

In simple terms, instances of :py:class:`~ecoli.processes.engine_process.EngineProcess`
wrap an entire Vivarium simulation as a process that can be incremented time step by
time step and interact bidirectionally with the outer simulation. Refer to the API
documentation for :py:mod:`~ecoli.experiments.ecoli_engine_process` for more details.


Configuring Colony Simulations
==============================

All of the configuration
options listed above still apply to simulations started with
:py:mod:`~ecoli.experiments.ecoli_engine_process`. There are only three new options:

- ``engine_process_reports``: List of paths (e.g. ``["bulk"]`` for bulk store) inside
  each cell to save in final colony output.
- ``emit_paths``: List of paths in outer simulation (e.g. locations of each cell in
  spatial environment) to save in final colony output.
- ``parallel``: In :py:mod:`~ecoli.experiments.ecoli_engine_process`, each simulated
  cell is contained within a single process (specifically, an instance of
  :py:class:`~ecoli.processes.engine_process.EngineProcess`). Therefore, assuming
  cells only need to communicate a tiny amount of information between one another,
  interprocess overhead is low and running these cells in parallel can greatly speed
  up the colony simulation.

In addition to these new configuration options, several previously mentioned options
become much more useful in the context of colony simulations:

- ``save`` and ``save_times`` can be used to create snapshots of the colony state
  to start many colony simulations from, for example, a 16-cell state using
  ``initial_state_file`` without having to wait for 16 generations every time.
  The names of the files saved can be given an optional prefix configured via the
  ``colony_save_prefix`` option.
- ``spatial_environment`` and ``spatial_environment_config``: The benefit of running
  simulations inside a shared, dynamic spatial environment is only fully realized when
  many cells are interacting with one another inside this environment.

.. _create_experiment:

---------------
Create Your Own
---------------

For more control over a simulation than what is provided by the default
:py:mod:`~ecoli.experiments.ecoli_master_sim` experiment (as well as the
workflow runscript :py:mod:`runscripts.workflow`, see :ref:`/workflows.rst`),
you can create your own experiment file. Some examples of custom experiment
files in the ``ecoli/experiments`` folder include:

- :py:mod:`~ecoli.experiments.tet_amp_sim`: Modifies the initial state to add
  new bulk molecules (see :ref:`bulk`) for antibiotics-related molecules and
  adds two transcription factor binding sites to all promoters for MarA and MarR.
  Also adds command-line options for external concentration of tetracycline and
  ampicillin.
- :py:mod:`~ecoli.experiments.metabolism_redux_sim`: Replaces the default metabolism
  process (:py:class:`~ecoli.processes.metabolism.Metabolism`) with experimental
  alternatives (e.g. :py:class:`~ecoli.processes.metabolism_redux_classic.MetabolismReduxClassic`).
  Makes use of the object-oriented interface for sim configuration mentioned
  in :ref:`sim_config` (e.g. ``sim.max_duration = 100``).
