========
Tutorial
========

This guide will walk through the following steps in a typical model development cycle:

1. :ref:`Add a new simulation process<new-process>`
2. :ref:`Add a new simulation variant<new-variant>`
3. :ref:`Add a new analysis script<new-analysis>`
4. :ref:`Run a simulation workflow with new additions<run-workflow>`

.. _new-process:

-----------
New Process
-----------

:ref:`/processes.rst` are self-contained sub-models that simulate
a biological mechanism. To add a new process, start by creating a new Python
file in the ``ecoli/processes`` folder. Here is an annotated example of a
process that checks the concentration of a certain bulk molecule (see
:ref:`bulk`) and halts the simulation if a threshold is reached.

.. code-block:: python

    # Import base class that all processes inherit from. Many of our
    # current processes inherit from sub-classes of Process. See the
    # "Processes" documentation for more details.
    from vivarium.core.process import Process
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
        # port and 

        # Most of our current processes are required to run with the same
        # timestep (see "Partitioning" heading in "Stores" documentation).
        # As such, most processes connect their timestep ports to the
        # same top-level timestep store using "timestep": ("timestep",).
        # 
        "timestep": ("death_threshold", "timestep")
    }
    topology_registry.register(NAME, TOPOLOGY)

    class DeathThreshold(Process):
        """
        Check the concentration of a molecule and stop the simulation
        upon reaching a certain threshold.
        """

        # Can optionally define default parameters for process. These will
        # be merged with any user-provided parameter dictionary and passed
        # to the __init__ method of the process. The `time_step` parameter
        # is a special one that, in the absence of a custom `calculate_timestep`
        # method, determines how often to run the process (once every X seconds).
        defaults = {"time_step": 1.0}

        def __init__(self, parameters=None):


        def ports_schema(self):
            return {
                "global_time": {"_default": 0.0, "_updater": "accumulate"},
                "timestep": {"_default": self.parameters["time_step"]},
            }

        def calculate_timestep(self, states):
            return states["timestep"]

        def next_update(self, timestep, states):
            return {"global_time": timestep}

The main steps to add a new process are:

#. Create a file in the :py:mod:`ecoli.processes` folder with the process
   definition (should inherit from either :py:class:`~vivarium.core.process.Process`
   or :py:class:`~vivarium.core.process.Step`)
#. Decide upon a string name for the process under which it is registered
   in ``ecoli/processes/__init__.py`` and its topology is registered in
   :py:attr:`ecoli.processes.registries.topology_registry`
#. Add the process name to the list of process names under the ``process``
   key in either the default JSON configuration file or your own JSON
   configuration file



Processes are designed to run with a certain frequency controlled by a "time step"
that can be supplied upon instantiation with the ``time_step`` key of the ``config``
parameter (units of seconds). For example, a Process with a time step of 3 will run
once every 3 simulated seconds.


.. list-table::
    :widths: 50 50
    :header-rows: 1

    * - Processes
      - Steps
    * - Only runs once every time step (user-configurable, can be variable)
      - By default, runs at the end of every time step where a Process ran
    * - All Processes slated to run at a certain time do so independently
      - Steps can be made to depend on other Steps through "flows" 

.. _new-variant:

-----------
New Variant
-----------

Test new process.


.. _new-analysis:

------------
New Analysis
------------

Test new process.


.. _run-workflow:

------------
Run Workflow
------------

Test new process.
