=========
Processes
=========

.. note::
    This document assumes that you are familiar with the basic concepts
    underlying Vivarium processes as outlined in the
    `vivarium-core documentation <https://vivarium-core.readthedocs.io/en/latest/guides/processes.html>`_.
    Please read the linked page in full before continuing.

The vEcoli model consists of many sub-models that each represent a biological mechanism.
We refer to these sub-models as **processes**. At their core, processes are Python classes that inherit from either
:py:class:`~vivarium.core.process.Process`
or :py:class:`~vivarium.core.process.Step`.
For now, we will ignore the differences
between these two base classes and refer to both as processes.

----------
Parameters
----------

Nearly all processes in our model require parameters calculated from raw
experimental data via the parameter calculator or ParCa in order to function.
Refer to :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.build_ecoli` and
:py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_processes_and_steps`
for details on how process parameters are loaded from ParCa output (and elsewhere)
to configure the processes.


---------------------
Partitioned Processes
---------------------

While Vivarium processes are required to implement the
:py:meth:`~vivarium.core.process.Process.next_update` method, you may notice that many
processes in :py:mod:`ecoli.processes` do not have this method and inherit
from the :py:class:`~ecoli.processes.partition.PartitionedProcess` base class
instead of :py:class:`~vivarium.core.process.Process` or :py:class:`~vivarium.core.process.Step`.
During sim initialization, :py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_processes_and_steps`
uses each :py:class:`~ecoli.processes.partition.PartitionedProcess` to create two
processes that inherit from :py:class:`~vivarium.core.process.Step` and have the required
:py:meth:`~vivarium.core.process.Process.next_update` methods: a
:py:class:`~ecoli.processes.partition.Requester` and an
:py:class:`~ecoli.processes.partition.Evolver`. These processes share an initialized
:py:class:`~ecoli.processes.partition.PartitionedProcess` instance, meaning
both have access to all the same parameters that the partitioned process was
instantiated with and any changes made to instance variables are seen by both.
Refer to :ref:`partitioning` for more details.


--------------------
Connecting to Stores
--------------------

To be integrated into the broader model, all processes are required to implement
the :py:meth:`~vivarium.core.process.Process.ports_schema` method and define
a topology dictionary. The vEcoli model includes two convenience features
to help with this.

- Nearly all processes import the :py:attr:`ecoli.processes.registries.topology_registry`
  and register their topologies under their unique string name, allowing
  :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim._retrieve_topology`
  to automatically retrieve topologies for each process at runtime
  (called by :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.build_ecoli`).
- The three main types of stores in vEcoli (bulk molecules, unique molecules,
  and listeners) all have helper functions to concisely generate schemas
  for use in the :py:meth:`~vivarium.core.process.Process.ports_schema` methods
  of processes (see :ref:`/stores.rst`).

.. _timesteps:

----------
Time Steps
----------

Processes that inherit from :py:class:`~vivarium.core.process.Process` are
automatically able to run with a time step that the user can supply using
the ``time_step`` key in the parameter dictionary. However, most processes
in vEcoli inherit from :py:class:`~vivarium.core.process.Step` and not
:py:class:`~vivarium.core.process.Process`. Instead of running with a
certain time step, Steps, by default, are run at the end of every time
step where at least one :py:class:`~vivarium.core.process.Process`
ran. Since we only used Steps instead of Processes in order to enforce
a certain execution order within time steps, we need to somehow allow
Steps to run with something like a time step.

To achieve this, we:

#. Added a top-level store to hold the global simulation time step at ``("timestep",)``.
#. Added a top-level store to hold the global time at ``("global_time",)`` with a
   default value of this store is 0.
#. Added a store for each process located at ``("next_update_time", "process_name")``
   which has a default value of ``("timestep",)``.
#. Added logic to the :py:meth:`~vivarium.core.process.Process.next_update`
   methods (or :py:meth:`~ecoli.processes.partition.PartitionedProcess.calculate_request`
   or :py:meth:`~ecoli.processes.partition.PartitionedProcess.evolve_state`
   for partitioned processes) to increment ``("next_update_time", "process_name")``
   by ``("timestep",)`` every time the Step is run.
#. Added a :py:class:`~ecoli.processes.global_clock.GlobalClock` process
   that calculates the smallest difference between the current ``("global_time",)``
   and each Step's ``("next_update_time", "process_name")``. This process has a
   custom :py:meth:`~vivarium.core.process.Process.calculate_timestep` method
   to tell vivarium-core to only run this process after this minimum time
   difference has elapsed in vivarium-core's internal simulation clock. At that
   time, this process advances ``("global_time",)`` by that minimum time difference.
   Taken together, these actions guarantee that we never accidentally
   skip over a Step's scheduled update time and also that our manual
   time stepping scheme stays perfectly in sync with vivarium-core's built-in
   time stepping.
#. Added a custom :py:meth:`~vivarium.core.process.Process.update_condition`
   method to most Steps which tells vivarium-core to only run a given Step
   when ``("next_update_time", "process_name")`` is less than or equal to
   ``("global_time",)``.

This manual time stepping scheme highlights a guiding philosophy of models built
with vivarium-core: storing simulation values in stores wherever possible.
This is what makes our processes modular while still facilitating communication
between processes. For example, say we wanted to dynamically modulate the time
step over the course of a simulation. By storing the time step for all the relevant
Steps in the same ``("timestep",)`` store, a Process or Step only needs to modify
this store for all Steps to register this change. Conversely, say we wanted to have
each Step run with its own time step instead of a global time step.
We could implement this by simply changing the topologies of each Step to connect
to a dedicated time step store ``("timestep", "process_name")``, unlinking time steps
for each Step.

.. note::
   The above scheme is automatically implemented for processes that inherit
   from :py:class:`~ecoli.processes.partition.PartitionedProcess` when they
   are used to create :py:class:`~ecoli.processes.partition.Requester`
   and :py:class:`~ecoli.processes.partition.Evolver` Steps.
