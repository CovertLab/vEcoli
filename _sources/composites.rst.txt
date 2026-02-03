==========
Composites
==========

:py:class:`~ecoli.composites.ecoli_master.Ecoli` is a so-called composer
that is responsible for aggregating Processes, Steps, topologies,
and the flow for the Steps into a unified "composite" model that vivarium-core
is able to run. Unlike a typical Vivarium composer which simply collects all
these pieces, the :py:class:`ecoli.composites.ecoli_master.Ecoli` composer
automatically makes several modifications to these components in order to
support features unique to our model.

.. _composite_partitioning:

--------------------------
Partitioning Modifications
--------------------------

As described in :ref:`partitioning`, our model contains many processes that inherit
from :py:class:`~ecoli.processes.partition.PartitionedProcess` because they require
special handling to avoid overdrafting bulk molecules. These processes are not directly
included in the final composite but instead used to parameterize two Steps
that are included and run with the final model: a
:py:class:`~ecoli.processes.partition.Requester` and an
:py:class:`~ecoli.processes.partition.Evolver`.

The :py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_processes_and_steps`
method of the :py:class:`~ecoli.composites.ecoli_master.Ecoli` composer is responsible
for creating these two Steps, the :py:class:`~ecoli.processes.allocator.Allocator` steps
sandwiched between them in each execution layer, and the
:py:class:`~ecoli.processes.unique_update.UniqueUpdate` Steps that run at the very end
of each execution layer. It is also responsible for updating the flow to arrange
these Steps in the order described in :ref:`implementation`. As an end-user, all you
have to do to add a new partitioned process is ensure that it inherits from
:py:class:`~ecoli.processes.partition.PartitionedProcess` and is included in the flow
supplied to :py:class:`~ecoli.composites.ecoli_master.Ecoli` (see :ref:`/experiments.rst`
for a description of the interface for specifying simulation options).

The :py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_topology` method is responsible
for creating the topologies for each Requester and Evolver. It does so by first copying the
topology of the corresponding :py:class:`~ecoli.processes.partition.PartitionedProcess` then
adding and wiring the following additional ports:

- ``process``: Wired to store located at ``("process", "process_name")``. Contains the
  instance of :py:class:`~ecoli.processes.partition.PartitionedProcess` that is shared
  by both the Requester and the Evolver. Stored as a single-element tuple because
  vivarium-core has special handling for stores containing naked processes that is not
  applicable here.
- ``request`` (:py:class:`~ecoli.processes.partition.Requester` only): Wired to store
  located at ``("request", "process_name")``. Contains a single sub-store called ``bulk``
  that is updated every time the Requester runs with a list of tuples
  ``(Index of bulk molecule in structured Numpy array, count requested)``.
- ``allocate`` (:py:class:`~ecoli.processes.partition.Evolver` only): Wired to store
  located at ``("allocate", "process_name")``. Contains a single sub-store called ``bulk``
  that is updated every time the :py:class:`~ecoli.processes.allocator.Allocator` for the
  corresponding execution layer is run with a 1D array of partitioned bulk counts.
- ``global_time``, ``timestep``, ``next_update_time``: See :ref:`timesteps`.

----------------------
Division Modifications
----------------------

vEcoli has a variety of options related to cell division that, if enabled,
prompt :py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_processes_and_steps`
to add certain Steps and
:py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_topology` to add
their corresponding topologies to the final composite model.

- ``divide``: Adds :py:class:`~ecoli.processes.cell_division.Division` when ``True``
- ``d_period``: Adds :py:class:`~ecoli.processes.cell_division.MarkDPeriod` when ``True``
  but only if ``divide`` is ``True``
- ``generations``: Adds :py:class:`~ecoli.processes.cell_division.StopAfterDivision`
  when ``True`` but only if ``divide`` is ``True``

-----------
Log Updates
-----------

For debugging purposes, it may be useful to know exactly what updates are returned
by each Process/Step in the model at every timestep. The ``log_updates`` boolean
configuration option adds a ``log_update`` port to each process wired to a store
located at ``("log_update", "process_name")`` (see
:py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_topology`). It also wraps
each process using
:py:func:`~ecoli.library.logging_tools.make_logging_process` to make it write the
contents of its update to this log update store (see
:py:meth:`~ecoli.composites.ecoli_master.Ecoli.generate_processes_and_steps`).

The analysis plots located in :py:mod:`~ecoli.analysis.single.blame` can be used
to visualize these updates.

.. warning::
    This feature should only be turned for debugging purposes and
    only when using the in-memory emitter (see :ref:`ram_emitter`).

-------------
Initial State
-------------

The :py:meth:`~ecoli.composites.ecoli_master.Ecoli.initial_state` method is responsible
for generating the initial state used to populate many of the stores in the simulation
(see the "Initialization" sub-headings in :ref:`/stores.rst`).

It also allows users to manually override initial state values and populates the
``("process", "process_name")`` stores mentioned in :ref:`composite_partitioning`.
