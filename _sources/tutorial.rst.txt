========
Tutorial
========

This guide will walk through the following steps in a typical model development cycle:

1. :ref:`Add a new simulation process<new-process>`
2. :ref:`Add a new simulation variant<new-variant>`
3. :ref:`Add a new analysis script<new-analysis>`
4. :ref:`Run workflow with new additions<run-workflow>`

.. _new-process:

-----------
New Process
-----------

:ref:`/processes.rst` are self-contained sub-models that simulate
a biological mechanism. To add a new process, start by creating a new Python
file in the ``ecoli/processes`` folder. Here is an annotated example of a
process that checks the count of a certain bulk molecule (see
:ref:`bulk`) and halts the simulation if a threshold is reached.

.. code-block:: python

    # Import base class that process inherits from. See the
    # "Processes" documentation for more options and details.
    from ecoli.processes.partition import PartitionedProcess
    # Import a fancy dictionary that we use to store process
    # topologies for automatic retrieval by our simulation
    # runscript (ecoli/experiments/ecoli_master_sim.py).
    # See the "Workflows" documentation for more details.
    from ecoli.processes.registries import topology_registry
    # Import some helper functions that allow us to read and
    # access the bulk counts that we desire from the bulk store.
    # Refer to the "Stores" documentation for more details.
    from ecoli.library.schema import bulk_name_to_idx, counts, numpy_schema

    # Give a unique string name to the process
    NAME = "death_threshold"
    # Define the stores that each port in the process connects to
    TOPOLOGY = {
        # There is a port called bulk connected to a store located
        # at a top-level store in the simulation state also called bulk
        "bulk": ("bulk",)
        # Topologies make our processes modular. If we wish to wire
        # the process differently, all we have to do is change
        # the topology. For example, changing the above to
        # "bulk": ("new_bulk", "sub_bulk") would connect the bulk
        # port of the process to a different store called sub_bulk
        # that is located inside the top-level new_bulk store. It is up
        # to you to ensure that whatever store the port is connected to
        # contains data in a format that the process expects from that
        # port and has an updater that can handle the updates that the
        # process passes through  that port.

        # Most of our current processes are required to run with the same
        # timestep (see "Partitioning" heading in "Stores" documentation).
        # As such, most processes connect their timestep ports to the
        # same top-level timestep store using "timestep": ("timestep",).
        # However, if we wish to run a process with its own timestep,
        # we could connect it to a separate dedicated store as follows.
        "timestep": ("death_threshold", "timestep"),
        # Time stepping for PartitionedProcesses and most Steps in our
        # model requires the process to have a port to the global time store.
        # See the "Time Steps" sub-heading in the "Processes" documentation.
        "global_time": ("global_time",)
    }
    topology_registry.register(NAME, TOPOLOGY)

    class DeathThreshold(PartitionedProcess):
        """
        Check the count of a molecule and stop the simulation
        upon reaching a certain threshold.
        """

        # Can optionally define default parameters for process. These will
        # be merged with any user-provided parameter dictionary and passed
        # to the __init__ method of the process. The `time_step` parameter
        # is a special one that, in the absence of a custom `calculate_timestep`
        # method, determines how often to run the process (once every X seconds).
        defaults = {"time_step": 1.0, "molecule_id": "WATER[c]", "threshold": 1e10}

        def __init__(self, parameters=None):
            # Run __init__ of base Process class to save all parameters as
            # instance variable self.parameters
            super().__init__(parameters)

            # Can extract and perform calculations on other values in ``parameters``
            # here to prepare process parameters.
            self.molecule_id = self.parameters["molecule_id"]
            self.threshold = self.parameters["threshold"]
            # Cache indices into bulk array for molecules of interest by creating
            # instance variable with initial value of None. This will be populated
            # the first time the Requester runs calculate_request.
            self.mol_idx = None

        def ports_schema(self):
            # Ports must match the ports connected to stores by the topology. Here
            # we make use of the ``numpy_schema`` helper function to standardize
            # the creation of schemas for ports connected to the bulk store. Since
            # ports connected to the same store must have non-conflicting (values
            # for shared keys must be the same) schemas, if you know you are connecting
            # to a store that already exists (already has a schema from a port from
            # in another process), you can just leave the schema as an empty dictionary
            # as we do for the global_time port here.
            return {
                "bulk": numpy_schema("bulk"),
                "global_time": {},
                "timestep": {"_default": self.parameters["time_step"]},
            }

        def calculate_request(self, timestep, states):
            # Since this is a PartitionedProcess, it will be turned into two Steps:
            # a Requester and an Evolver. The Requester Step will call calculate_request.

            # Cache molecule index so that Requester and Evolver can use it
            if self.mol_idx is None:
                self.mol_idx = bulk_name_to_idx(self.molecule_id, states["bulk"]["id"])
            # Request all counts of given bulk molecule. Updates to bulk store are
            # lists of 2-element tuples ``(index, count)``
            return {"bulk": [(self.mol_idx, counts(states["bulk"], self.mol_idx))]}
        
        def evolve_state(self, timestep, states):
            # The Evolver Step will call evolve_state after the Requesters in the execution
            # layer have called calculate_request and the Allocator has allocated counts
            # to processes
            mol_counts = counts(states["bulk"], self.mol_idx)
            if mol_counts > self.threshold:
                raise RuntimeError(f"Count threshold for {self.molecule_id} exceeded: "
                    f"{mol_counts} > {self.threshold}")

The main steps to add a new process are:

#. Create a file in the :py:mod:`ecoli.processes` folder with the process
   definition (should inherit from either :py:class:`~vivarium.core.process.Process`
   or :py:class:`~vivarium.core.process.Step`). The remainder of this Tutorial
   assumes you placed the above process file in ``ecoli/processes/death_threshold.py``.
#. Decide upon a string name for the process under which it is registered
   in ``ecoli/processes/__init__.py`` and its topology is registered in
   :py:attr:`ecoli.processes.registries.topology_registry`. This was done by
   importing the topology registry and registering the topology in the process file.
#. Add the process name to the list of process names under the ``processes``
   key in either the default JSON configuration file or your own JSON
   configuration file. For processes that inherit from :py:class:`~vivarium.core.process.Step`
   or :py:class:`~ecoli.processes.partition.PartitionedProcess`, the process
   must also be added to the ``flow``. 
#. For processes whose execution order matters, inherit from
   :py:class:`~vivarium.core.process.Step` instead of :py:class:`~vivarium.core.process.Process`
   and add the process along with its dependencies to the ``flow`` option.
#. For partitioned processes, inherit from :py:class:`~ecoli.processes.partition.PartitionedProcess`
   and implement the :py:meth:`~ecoli.processes.partition.PartitionedProcess.calculate_request`
   and :py:meth:`~ecoli.processes.partition.PartitionedProcess.evolve_state` methods instead
   of :py:meth:`~vivarium.core.process.Process.next_update` and
   add the process along with its dependencies to the ``flow`` option.
   
For example, if we want to run the example process above after all other Steps have run in a
timestep, we can add the following key-value pair to the ``flow``:
``"death_threshold": [("ribosome_data_listener",)]`` because ``ribosome_data_listener``
is currently in the last execution layer (see :ref:`partitioning`).

.. _new-variant:

-----------
New Variant
-----------

Variants are Python files containing an ``apply_variant`` function that
is used to generate modified versions of the
:py:class:`~reconstruction.ecoli.simulation_data.SimulationDataEcoli`
object (holds most model parameters). They can be used to generate a large
amount of variant simulation data objects using the :py:mod:`runscripts.create_variants`
interface as described in :ref:`variants`.
Here is an annotated example of a variant:

.. code-block:: python

    from typing import Any, TYPE_CHECKING

    if TYPE_CHECKING:
        from reconstruction.ecoli.simulation_data import SimulationDataEcoli

    def apply_variant(
        sim_data: "SimulationDataEcoli", params: dict[str, Any]
    ) -> "SimulationDataEcoli":
        """
        Modify sim_data to environmental condition from condition_defs.tsv.

        Args:
            sim_data: Simulation data to modify
            params: Parameter dictionary of the following format::

                {
                    # Environmental condition: "basal", "with_aa", "acetate",
                    # "succinate", "no_oxygen"
                    "condition": str,
                }

        Returns:
            Simulation data with the following attributes modified::

                sim_data.condition
                sim_data.external_state.current_timeline_id
        """
        # Set media condition by changing attributes of sim_data in accordance
        # with value of ``condition`` key in ``params``
        sim_data.condition = params["condition"]
        sim_data.external_state.current_timeline_id = params["condition"]
        sim_data.external_state.saved_timelines[params["condition"]] = [
            (0, sim_data.conditions[params["condition"]]["nutrients"])
        ]

        return sim_data

To add a new variant:

- Add Python file containing ``apply_variant`` function with the same signature
  as above in the ``ecoli/variants`` folder
- Add the name of the variant (name of Python file without ``.py``) to ``variants``
  key in the configuration JSON

.. _new-analysis:

------------
New Analysis
------------

Analysis scripts are Python files that contain a ``plot`` function which uses
DuckDB to read Hive-partitioned Parquet files containing simulation output
(see :ref:`/output.rst`) and calculates aggregates / makes plots. Here is an
annotated example of an analysis script:

.. code-block:: python

    import os
    from typing import Any, cast, TYPE_CHECKING

    if TYPE_CHECKING:
        from duckdb import DuckDBPyConnection
    # Can use polars to perform calculations on tabular data
    # returned by DuckDB.
    import polars as pl

    # Import helper functions to read data (see "Output" documentation).
    from ecoli.library.parquet_emitter import num_cells, read_stacked_columns

    def plot(
        params: dict[str, Any],
        conn: "DuckDBPyConnection",
        history_sql: str,
        config_sql: str,
        success_sql: str,
        sim_data_paths: dict[str, dict[int, str]],
        validation_data_paths: list[str],
        outdir: str,
        variant_metadata: dict[str, dict[int, Any]],
        variant_names: dict[str, str],
    ):
        # See "Analysis" sub-heading in "Workflows" documentation for description
        # of arguments for ``plot``

        # Use helper function to get number of cells in filtered data set
        # contained within DuckDB SQL query
        assert (
            num_cells(conn, config_sql) == 1
        ), "Mass fraction summary plot requires single-cell data."

        mass_columns = {
            "Protein": "listeners__mass__protein_mass",
            "tRNA": "listeners__mass__tRna_mass",
            "rRNA": "listeners__mass__rRna_mass",
            "mRNA": "listeners__mass__mRna_mass",
            "DNA": "listeners__mass__dna_mass",
            "Small Mol.s": "listeners__mass__smallMolecule_mass",
            "Dry": "listeners__mass__dry_mass",
        }
        # Use helper function to read simulation output data from
        # specified columns. Column names are derived by concatenating
        # the string keys that comprise the path of the store containing
        # the data stored in each column.
        mass_data = read_stacked_columns(
            history_sql, list(mass_columns.values()), conn=conn
        )
        fractions = {
            k: (mass_data[v] / mass_data["listeners__mass__dry_mass"]).mean()
            for k, v in mass_columns.items()
        }
        new_columns = {
            "Time (min)": (mass_data["time"] - mass_data["time"].min()) / 60,
            **{
                f"{k} ({fractions[k]:.3f})": mass_data[v] / mass_data[v][0]
                for k, v in mass_columns.items()
            },
        }
        # Convert Polars DataFrame to use their API
        mass_fold_change_df = pl.DataFrame(new_columns)

        # Altair requires long form data (also no periods in column names)
        melted_df = mass_fold_change_df.melt(
            id_vars="Time (min)",
            variable_name="Submass",
            value_name="Mass (normalized by t = 0 min)",
        )

        chart = (
            alt.Chart(melted_df)
            .mark_line()
            .encode(
                x=alt.X("Time (min):Q", title="Time (min)"),
                y=alt.Y("Mass (normalized by t = 0 min):Q"),
                color=alt.Color("Submass:N", scale=alt.Scale(range=COLORS)),
            )
            .properties(
                title="Biomass components (average fraction of total dry mass in parentheses)"
            )
        )
        chart.save(os.path.join(outdir, "mass_fraction_summary.html"))

.. warning::
    In order to be run as part of a workflow with :py:mod:`runscripts.workflow`,
    analysis scripts must write at least one file to ``outdir``.

To add a new analysis script:

- Add Python file containing analysis script containing ``plot`` function
  in ``ecoli/analysis/{analysis_type}`` folder
- Add analysis name (file name minus ``.py``) to appropriate analysis type
  key (e.g. ``single``, ``multidaughter``, etc) under ``analysis_options``
  in the configuration JSON

.. _run-workflow:

------------
Run Workflow
------------

Once you have finished adding new components to the model, you can run a
workflow containing all those changes by simply invoking :py:mod:`runscripts.workflow`
with a configuration JSON modified as described in the above sections.
