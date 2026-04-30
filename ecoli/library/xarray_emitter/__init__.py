
r"""

Introduction
============

:py:class:`.XarrayEmitter` is an :py:class:`~vivarium.core.emitter.Emitter`
similar to :py:class:`~.ParquetEmitter`, but with a design optimized towards a
different flavour of downstream applications:

- :py:class:`~.ParquetEmitter` is geared towards emitting a significant fraction
  of the simulator state, in a format that supports flexible sparse selections,
  `data reductions`_ and time series visualizations, as used in :ref:`analysis
  scripts <analysis_scripts>`.
- :py:class:`.XarrayEmitter` is intended for emitting only a pre-selected subset
  of statically shaped tensor variables, in a format that supports numerical
  algorithms in the high-dimensional and large-sample regime.

The former type of computations is naturally expressed using `relational query`_
engines (e.g., :ref:`DuckDB <parquet_read>`), whereas the latter type is
naturally expressed using `array programming`_ libraries (e.g., `Cubed`_). Due
to the sheer size of the simulator state, both types may in general require
`out-of-core processing`_ algorithms.

.. _data reductions: https://en.wikipedia.org/wiki/Data_reduction
.. _relational query: https://en.wikipedia.org/wiki/Relational_database
.. _array programming: https://en.wikipedia.org/wiki/Array_programming
.. _Cubed: https://cubed-dev.github.io/cubed/why-cubed.html
.. _out-of-core processing: https://en.wikipedia.org/wiki/External_memory_algorithm

In order to facilitate downstream applications based on chunked array
processing, :py:class:`.XarrayEmitter` writes out to any persistent storage
supporting the `Zarr`_ specification, using an in-memory buffer comprised of
`Xarray`_ objects. For optimized throughput, the buffer implements temporal
subsampling, numerical type casting and compression codecs at emission time.
Furthermore, in order to simplify the export of simulation data into external
libraries, the hierarchy of the output `DataTree`_ is decoupled from the
hierarchy of simulation :py:class:`~vivarium.core.store.Store`\ s, using an
output :ref:`variable layout <variable_layout>` specified in the :ref:`simulator
configuration <sim_config>`.

.. _Xarray: https://xarray.dev/
.. _Zarr: https://zarr.dev/
.. _DataTree: https://docs.xarray.dev/en/stable/user-guide/data-structures.html#datatree


Comparison with :py:class:`~.ParquetEmitter`
============================================

Similarities
------------

- Currently only supports simulations of a *single-cell lineage* per
  :py:class:`.BufferedEmitter` instance.
- Executes at every time step.
- Buffers emissions into time chunks.
- Uses concurrent threads for writing buffers to persistent storage.
- Produces a hierarchically structured storage layout that supports selective
  reading in downstream applications.

Differences in usage
--------------------

- Supports the configuration of *emission predicates*.
- Currently only supports emitting a *static collection* of *statically shaped
  tensor variables*.
- Supports *renaming and rearranging* of output variables.
- Requires the configuration of *output data types*.
- Supports the configuration of backend-specific *compression codecs*.
- Supports :ref:`log_updates`, i.e., the emission of individual
  :py:class:`~vivarium.core.process.Process` update requests, before they are
  aggregated and reallocated by
  :py:func:`~ecoli.processes.allocator.calculatePartition` and then applied to
  the global cell state by
  :py:meth:`~ecoli.processes.partition.PartitionedProcess.evolve_state`.

.. note::
  See :py:class:`.XarrayEmitter` for an explanation of the JSON configuration
  syntax, and ``configs/test_configs/test_xarray_emitter.json`` for a complete
  example.

.. hint::
  As data structures, `DataTree`_\ s could support changes of variable names and
  dimensions across time steps. The constraints currently imposed by
  :py:class:`.XarrayEmitter` rather serve to enable I/O optimizations for the
  intended use cases. When access to variably sized simulation variables is
  desired, users have the choice either of implementing custom :ref:`listeners
  <listeners>` with static output coordinates, or otherwise of defaulting to the
  :py:class:`~.ParquetEmitter`.

Differences in implementation
-----------------------------

- Uses the `Xarray`_ API for serialization, buffering, and `metadata
  organization`_, including unit annotations (see :py:class:`.VariableSpec` and
  :py:class:`.XarrayTransducer` for details).
- Applies a "*process*-major" rather than a "*generation*-major" output layout,
  reflecting array variables directly in the output directory tree; this
  produces one file per *variable* time chunk, rather than one file per
  *simulation* time chunk (compare the :py:class:`.XarrayEmitter` :ref:`storage
  layout <storage_layout>` with :py:meth:`.ParquetEmitter.emit`; see
  :py:class:`.XarrayStoragePartition` for details).
- Defines the abstract interface :py:class:`.AsyncBufferWriter` for storage
  backends with *asynchronous* APIs (currently supported: `Zarr`_), realizing
  the opportunity for :ref:`concurrency <concurrency>` among multiple
  `DataArray`_\ s within an output buffer.
- Decouples the in-memory buffer size from the persistent chunk size, in order
  to simplify performance tuning of large-scale simulations (see
  :py:class:`.XarrayTransducer` and :py:class:`.AsyncBufferWriter` for details).
- Maintains `consolidated metadata`_ and updates it at the end of each simulated
  cell generation, in order to reduce the metadata loading latency for
  subsequent storage reads (see :py:class:`.AsyncZarrBufferWriter` for details).

.. _metadata organization: https://docs.xarray.dev/en/stable/get-help/faq.html#approach-to-metadata
.. _DataArray: https://docs.xarray.dev/en/stable/user-guide/data-structures.html#dataarray
.. _consolidated metadata: https://docs.xarray.dev/en/stable/user-guide/io.html#io-zarr-consolidated-metadata


.. _storage_layout:

Storage layout
==============

The workflow storage layout, which comprises many individual simulations, is
currently organized as follows --- where file paths in this example are specific
to the Zarr v3 storage backend::

  {store}                                         ;  <root>
  ├─ zarr.json                                    ;    metadata
  └─ experiment_id={}/variant={}/lineage_seed={}  ;    <independent substore>
     ├─ zarr.json                                 ;      consolidated metadata
     ├─ emitstep_gen={}                           ;      <time coordinate>
     │  ├─ zarr.json                              ;        metadata
     │  └─ c/...                                  ;        chunked array
     ├─ time_gen={}                               ;      <time values>
     │  ├─ zarr.json                              ;        metadata
     │  └─ c/...                                  ;        chunked array
     └─ {path/to/variable}                        ;      <variable layout>
        ├─ zarr.json                              ;        metadata
        ├─ id_{variable}                          ;        <variable coordinate>
        │  ├─ zarr.json                           ;          metadata
        │  └─ c/...                               ;          chunked array
        └─ generation={}                          ;        <variable values>
           ├─ zarr.json                           ;          metadata
           └─ c/...                               ;          chunked array

This design is motivated by the following considerations:

  - Using relative paths inside a global store simplifies the authentication,
    configuration and resource management of file system providers during highly
    parallel, long-running simulation workflows.
  - An :py:attr:`.XarrayStoragePartition.independent_path` locates a logically
    self-contained substore, which maintains its own `consolidated metadata`_
    without communicating to other substores.
  - Fine-grained control over storage footprint, variable selection, latency and
    throughput, both during emission and during post-processing, is enabled by
    distinguishing *at the file system level* among simulation variables, as
    well as between metadata, coordinate data and variable data. In particular:

    - Chunk sizes and compression codecs can be configured for each variable.
    - By leveraging `inheritance`_ inside an independent substore, each
      generation's time coordinate is shared across the generation's variables,
      and each variable coordinate is shared across all generations.

.. note::
  In order to fully benefit from consolidated metadata, downstream applications
  should open independent substores directly, i.e., based on their known
  relative file system paths, rather than by first loading all metadata for the
  global store. This can be achieved, e.g., by calling
  :py:func:`xarray.open_datatree` only on substores, or by using lower-level
  APIs such as :py:func:`zarr.open_group`.

  An alternative would be to perform metadata consolidation on the global store
  at the end of an entire workflow, e.g., by calling
  :py:func:`zarr.consolidate_metadata` for the Zarr storage backend. However,
  this may be an expensive operation accessing a large number of files, and
  would, for the `zarr-python`_ implementation at the time of writing, ignore
  the already incrementally consolidated metadata in independent substores.

.. _inheritance: https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html#alignment-and-coordinate-inheritance
.. _zarr-python: https://github.com/zarr-developers/zarr-python


.. _variable_layout:

Variable layout
===============

For each individual :py:class:`.EcoliSim` simulation, the mapping from the
Vivarium simulation hierarchy (:py:class:`~vivarium.core.store.Store`) to the
Xarray output hierarchy (:py:class:`~xarray.DataTree`) is configured using three
levels of grouping::

  ForestView      ;  specifies a full `xarray.DataTree`
  └─ TreeView     ;  specifies a partial `xarray.DataTree`
     └─ LeafView  ;  specifies a single `xarray.DataArray`

A :py:class:`.LeafView` corresponds to a single array variable emitted by
Vivarium, a :py:class:`.TreeView` collects :py:class:`.LeafView`\ s whose
metadata paths share a common root, and a :py:class:`.ForestView` is a group of
:py:class:`.TreeView`\ s within a single ``agent_id`` (see :ref:`Configuration
<sim_config>`).


Software architecture
=====================

:py:class:`.XarrayEmitter` is in essence a `finite-state transducer`_, and its
state factors through the following object ownership relations::

  .XarrayEmitter(BufferedEmitter)                     ;  <application layer>
  ├─ .XarrayTransducer                                ;    <presentation layer>
  │  ├─ .ConjunctiveEmitPredicate                     ;      emission criterion
  │  └─ .XarrayBuffer                                 ;      output buffer
  │     ├─ .ForestView                                ;        variable layout
  │     ├─ xarray.DataTree                            ;        memory layout
  │     └─ .XarrayStoragePartition(StoragePartition)  ;        storage layout
  └─ .AsyncZarrBufferWriter(AsyncBufferWriter)        ;    <session layer>
     └─ xarray.backends.ZarrStore                     ;      <transport layer>
        └─ zarr.Group                                 ;        persistent storage

.. _finite-state transducer: https://en.wikipedia.org/wiki/Finite-state_transducer


.. _concurrency:

Concurrency
===========

The current design employs two levels of concurrency per :py:class:`~.EcoliSim`
OS process::

  [main thread]       ;  XarrayEmitter.flush()
  └─ [writer thread]  ;  AsyncBufferWriter._write()
     └─ [coroutine]   ;  AsyncArrayWriter._async()

Each time the :py:class:`.XarrayBuffer` is filled up, the writer thread receives
a :py:meth:`~concurrent.futures.Executor.submit` call from the main thread (see
:py:meth:`.AsyncBufferWriter.write`). In turn, the writer thread executes a
`coroutine`_ that leverages a backend-specific API for asynchronously executing
the low-level write operations required for persisting the buffer contents (see
:py:class:`.AsyncArrayWriter`).

.. _coroutine: https://docs.python.org/3/howto/a-conceptual-overview-of-asyncio.html

This design is motivated by the following observations at the time of writing:

  - In the typical use case of :py:class:`.XarrayEmitter`,
    :math:`10^2`--:math:`10^5`
    :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` instances are
    executed in parallel, on a single core each.
  - The multiplicity of *bandwidth*-consuming arrays in each
    :py:class:`.XarrayBuffer` provides an opportunity for concurrency, even
    though on typical compute environments, the *latency* of the transport layer
    in the :py:class:`.AsyncBufferWriter` is assumed to be at least 3 orders of
    magnitude smaller than the simulation time required to fill an
    :py:class:`.XarrayBuffer`. This is particularly relevant if many parallel
    simulations in an HPC environment are flushing their buffers at correlated
    wall-clock times.
  - Since using many threads per
    :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` OS process would
    slow down the simulation, and since using many concurrent connections per
    :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` OS process would
    congest the transport layer, the optimal choice is expected to be a small
    integer for both resource parameters.
  - The :py:mod:`~zarr.api.asynchronous` API was `central to the design`_ of
    `zarr-python`_ 3.
  - In Xarray, library support for ``async`` calls to storage backends is `still
    in flux`_. In particular, :py:class:`xarray.backends.ZarrStore` does not yet
    fully reflect Zarr's ``async`` API, with current support for asynchronous
    writing `tied`_ to a `chunked array library`_ via
    :py:class:`xarray.namedarray.parallelcompat.ChunkManagerEntrypoint`,
    including in a recent `proof-of-concept`_.

.. hint::
  - The numbers of threads and concurrent connections in the transport layer
    have backend-specific configuration options (see
    :py:class:`~.zarr_writer.AsyncZarrBufferWriter`).
  - Threading in the session layer can be disabled for debugging purposes (see
    :py:class:`.AsyncBufferWriter`).

.. _central to the design: https://zarr.readthedocs.io/en/v3.0.8/developers/roadmap.html#async-api
.. _still in flux: https://github.com/pydata/xarray/issues/10622
.. _tied: https://github.com/pydata/xarray/pull/10625
.. _chunked array library: https://docs.xarray.dev/en/stable/internals/chunked-arrays.html
.. _proof-of-concept: https://github.com/pydata/xarray/pull/11171

"""
