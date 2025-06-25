======
Output
======

Simulation output can come in one of two different formats and contain data
from as many or few stores as you desire.

.. _emit_stores:

--------------
Stores to Emit
--------------

To indicate that you want to save the data in a simulation store for later,
set the ``_emit`` key to True in the port schema for all ports connecting
to that store. By default, we always emit data for:

- Bulk molecules store located at ``("bulk",)``: The
  :py:func:`~ecoli.library.schema.numpy_schema` helper function that we use
  to create the schema for ports to the bulk store automatically
  sets ``_emit`` to True when the ``name`` argument is ``bulk``.
- Listeners located at ``("listeners",)``: The
  :py:func:`~ecoli.library.schema.listener_schema` helper function that we use
  to create the schema for ports to stores located somewhere in the hierarchy
  under the ``listener`` store automatically sets ``_emit`` to True

.. _serializing_emits:

-----------------
Serializing Emits
-----------------

Serialization is the process of converting data into a format that can be stored
or transmitted then later reconstructed back into its original format. By default, both of
the two available data output formats in vEcoli serialize
data by first converting the store hierarchy to save (:ref:`emit_stores`) to JSON using
`orjson <https://github.com/ijl/orjson>`_, which natively serializes Python's built-in types
as well as basic 1D Numpy arrays. For stores containing data that is not one of these types,
vivarium-core allows users to specify custom serializers either on a per-store basis using the
``_serialize`` schema key or for all stores of a given type using the
:py:class:`~vivarium.core.registry.Serializer` API (see
`vivarium-core documentation <https://vivarium-core.readthedocs.io/en/latest/reference/api/vivarium.core.registry.html?highlight=serializer>`_).

For details about reading data back after it has been saved, refer to
:ref:`ram_read` for the in-memory data format and :ref:`parquet_read`
for the persistent storage format.

.. _ram_emitter:

-----------------
In-Memory Emitter
-----------------

When ``timeseries`` is specified using the ``emitter`` option in a configuration JSON,
simulation output is stored transiently in-memory in a dictionary keyed by time that
looks like the following:

.. code-block::

    {
        # Data for time = 0
        0.0: {
            # Store hierarchy as nested dictionary containing all stores and sub-stores
            # where ``_emit`` is True
            "store_1": value,
            "store_2": {
                "inner_store_1": value,
                ...
            },
            ...
        },
        # Data for time = 1
        1.0: {...},
        ...
    }

This data format is mainly intended for ad-hoc analysis scripts (e.g. Jupyter
notebooks) where a single-cell simulation is run and probed for model development.
Importantly, the data saved by this emitter is lost when the Python program
used to run the cell simulation terminates.

.. _ram_read:

Querying
========

Data can be read from the RAM emitter by calling
:py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.query`
on the :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` object used to run
the simulation. To deserialize data (reconstitute it after serialization),
the :py:func:`~vivarium.core.serialize.deserialize_value` function is called, which
calls the :py:meth:`~vivarium.core.registry.Serializer.deserialize` method
of the :py:class:`~vivarium.core.registry.Serializer` instance whose
:py:meth:`~vivarium.core.registry.Serializer.can_deserialize` method returns
True on the data to deserialize.

.. _parquet_emitter:

---------------
Parquet Emitter
---------------

When ``parquet`` is specified using the ``emitter`` option in a configuration JSON,
simulation output is stored in a tabular file format called Parquet inside a nested
directory structure called Hive partitioning.  For details on the available JSON
configuration options, see :py:class:`~ecoli.library.parquet_emitter.ParquetEmitter`.


Hive Partitioning
=================

In Hive partitioning, certain keys in data are used to partition the data into folders:

.. code-block::

    key_1=value_1/key_2=value_2/...

In the vEcoli Parquet emitter, the keys used for this purpose are the experiment ID,
variant index, lineage seed (initial seed for cell lineage), generation, and agent ID.
These keys uniquely identify a single cell simulation, meaning each simulation process
will write data to its own folder in the final output with a path like:

.. code-block::

    experiment_id={}/variant={}/lineage_seed={}/generation={}/agent_id={}

This allows workflows that run simulations with many variant simulation data objects,
lineage seeds, generations, and agent IDs to all write data to the same main output
folder without simulations overwriting one another.

Parquet Files
=============

Because Parquet is a tabular file format (think in terms of columns like a Pandas
DataFrame), additional serialization steps must be taken after the emit data
has been converted to JSON format in accordance with :ref:`serializing_emits`.
The Parquet emitter (:py:class:`~ecoli.library.parquet_emitter.ParquetEmitter`)
first calls :py:func:`~ecoli.library.parquet_emitter.flatten_dict` in order to
flatten the nested store hierarchy into unnested key-value pairs where keys
are paths to leaf values concatenated with double underscores and values are
leaf values. For example, take the following nested dictionary:

.. code-block::

    {
        "a": {
            "b": 1,
            "c": {
                "d": 2,
                "e": 3
            },
            "f": 4
        },
        "g": 5
    }

This is flattened to:

.. code-block::

    {
        "a__b": 1,
        "a__c__d": 2,
        "a__c__e": 3,
        "a__f": 4,
        "g": 5
    }

Then, :py:func:`~ecoli.library.parquet_emitter.np_dtype` is used to get the
the type of the Parquet column that will be created for each key-value pair in
the flattened dictionary, where each key is the column name and each value is one
entry in the column. Parquet files are strongly typed, so emitted store data
must always be serialized to the same type as they were in the first time step
(default or initial value). The exception to this rule are columns that can contain
null values or nested types containing null values (e.g. empty list). For these columns,
all values except the null entries must be the same type (e.g. column with lists
of integers where some entries are empty lists).

.. warning::
  The Parquet emitter is poorly suited for storing large listeners that have more
  than a single dimension per time step. We recommend splitting these listeners up
  if possible, especially if you plan to read specific indices along those dimensions.

The Parquet emitter saves the serialized tabular data to two Hive-partitioned
directories in the output folder (``out_dir`` or ``out_uri`` option under
``emitter_arg`` in :ref:`json_config`):

- ``configuration``: Copy of all configuration options (e.g. from JSON, CLI) that
  were used to run the simulation as well as store-specific metadata
- ``history``: Actual saved simulation output

.. _configuration_parquet:

``configuration``
-----------------

Each simulation will save a single Parquet file named ``config.pq`` inside
its corresponding Hive partition under the ``configuration`` folder.
Many of the columns inside this Parquet file come from flattening the configuration
JSON used to run the simulation and can be read back in analysis scripts (see
:ref:`analysis_scripts`) using the helper function
:py:func:`~ecoli.library.parquet_emitter.config_value`.

Additionally, this file can contain metadata for each store to emit. This metadata
can be specified under the ``_properties`` key in a port schema as follows:

.. code-block::

    {
        "_properties": {
            "metadata": Put anything here.
        }
    }

Schemas constructed with the :py:func:`~ecoli.library.schema.listener_schema` helper
function can populate this metdata concisely. These metadata values are compiled for
all stores in the simulation state hierarchy by
:py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.output_metadata`. In the
saved configuration Parquet file, the metadata values will be located in
columns with names equal to the double-underscore concatenated store path
prefixed by ``output_metadata__``. For convenience, the
:py:func:`~ecoli.library.parquet_emitter.field_metadata` can be used in
analysis scripts to read this metadata.

``history``
-----------

Each simulation will save Parquet files containing serialized simulation output data
inside its corresponding Hive partition under the ``history`` folder. The columns in
these Parquet files come from flattening the hierarchy of emitted stores. To leverage
Parquet's columnar compression and efficient reading, we batch many time steps worth
of emits into either NumPy arrays (constant dimensions) or lists of Polars Series (variable
dimensions). These batched emits are efficiently converted into a Polars DataFrame and
written to a Parquet file named ``{batch size * number of batches}.pq`` (e.g.
``400.pq``, ``800.pq``, etc. for a batch size of 400). The default batch size of
400 has been tuned for our current model but can be adjusted via ``batch_size``
under the ``emitter_arg`` option in a configuration JSON.

.. _parquet_read:

DuckDB
======

`DuckDB <https://duckdb.org>`_ is the main library that we use to read and query Parquet files.
It offers class-leading performance and a fairly user-friendly SQL dialect for constructing
complex queries. Refer to the `DuckDB documentation <https://duckdb.org/docs/>`_ to learn more.

We provide a variety of helper functions in :py:mod:`ecoli.library.parquet_emitter`
to read data using DuckDB. These include:

- :py:func:`~ecoli.library.parquet_emitter.dataset_sql`: Construct basic
  SQL queries to read data from ``history`` and ``configuration`` folders. This
  is mainly intended for ad-hoc Parquet reading (e.g. in a Jupyter notebook).
  Analysis scripts (see :ref:`analysis_scripts`) receive a ``history_sql`` and
  ``config_sql`` that reads data from Parquet files with filters applied when
  run using :py:mod:`runscripts.analysis`.
- :py:func:`~ecoli.library.parquet_emitter.union_by_name`: Modify SQL query
  from :py:func:`~ecoli.library.parquet_emitter.dataset_sql` to
  use DuckDB's `union_by_name <https://duckdb.org/docs/stable/data/multiple_files/combining_schemas.html#union-by-name>`_.
  This is useful when reading data from simulations with different columns.
- :py:func:`~ecoli.library.parquet_emitter.num_cells`: Quickly get a count of
  the number of cells whose data is included in a SQL query
- :py:func:`~ecoli.library.parquet_emitter.skip_n_gens`: Add a filter to an SQL
  query to skip the first N generations worth of data
- :py:func:`~ecoli.library.parquet_emitter.ndlist_to_ndarray`: Convert a
  column of nested lists read from Parquet into an N-D Numpy array (use
  ``polars.Series`` to do opposite conversion)
- :py:func:`~ecoli.library.parquet_emitter.ndidx_to_duckdb_expr`: Get a DuckDB SQL
  expression which can be included in a ``SELECT`` statement that uses Numpy-style
  indexing to retrieve values from a nested list Parquet column
- :py:func:`~ecoli.library.parquet_emitter.named_idx`: Get a DuckDB SQL expression
  which can be included in a ``SELECT`` statement that extracts values at certain indices
  from each row of a nested list Parquet column and returns them as individually named columns
- :py:func:`~ecoli.library.parquet_emitter.field_metadata`: Read saved store
  metadata (see :ref:`configuration_parquet`)
- :py:func:`~ecoli.library.parquet_emitter.config_value`: Read option from
  configuration JSON used to run simulation
- :py:func:`~ecoli.library.parquet_emitter.read_stacked_columns`: Main interface
  for reading simulation output from ``history`` folder. Can either immediately read
  all data in specified columns into memory by supplying ``conn`` argument or
  return a DuckDB SQL query that can be iteratively built upon (useful when data
  too large to read into memory all at once).

.. warning::
  Column names that contain special characters (e.g. spaces, dashes, etc.) must be
  enclosed in double quotes when used in DuckDB SQL queries. This is automatically
  handled by most of the helper functions above with the notable exception of
  :py:func:`~ecoli.library.parquet_emitter.read_stacked_columns`.

.. warning::
    Parquet lists are 1-indexed. :py:func:`~ecoli.library.parquet_emitter.ndidx_to_duckdb_expr`
    and :py:func:`~ecoli.library.parquet_emitter.named_idx` automatically add 1 to
    user-supplied indices.

Construct SQL Queries
---------------------

The true power of DuckDB is unlocked when SQL queries are iteratively constructed. This can be
accomplished in one of two ways:

- For simpler queries, you can wrap a complete DuckDB SQL expression in parentheses to use as
  the input table to another query. For example, to calculate the average cell and dry mass for
  over all time steps for all cells accessible to an analysis script:

    .. code-block:: sql

        SELECT avg(*) FROM (
            SELECT listeners__mass__dry_mass, listeners__mass__cell_mass FROM (
                history_sql
            )
        )
    
  ``history_sql`` can be slotted in programmatically using an f-string.
- For more advanced, multi-step queries, you can use
  `common table expressions <https://duckdb.org/docs/sql/query_syntax/with.html>`_ (CTEs).
  For example, to run the same query above but first averaging over all time steps
  for each cell before averaging the averages over all cells:

    .. code-block:: sql

        WITH cell_avgs AS (
            SELECT avg(listeners__mass__dry_mass) AS avg_dry_mass,
                avg(listeners__mass__cell_mass) AS avg_cell_mass
            FROM (history_sql)
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        )
        SELECT avg(*) FROM cell_avgs

.. tip::
  DuckDB will efficiently read only the rows and columns necessary to complete your query.
  However, if you are reading a column of lists (e.g. bulk molecule counts every time step)
  or nested lists, DuckDB reads the entire nested value for every relevant row in that column,
  even if you only care about a small subset of indices. To avoid repeatedly incurring this
  cost, we recommend using :py:func:`~ecoli.library.parquet_emitter.named_idx` to select all
  indices of interest to be read in one go. As long as the final result fits in RAM, this
  should be much faster than reading each index individually.

See :py:mod:`~ecoli.analysis.multivariant.new_gene_translation_efficiency_heatmaps`
for examples of complex queries, as well as helper functions to create SQL expressions
for common query patterns. These include:

- :py:func:`~ecoli.analysis.multivariant.new_gene_translation_efficiency_heatmaps.avg_ratio_of_1d_arrays_sql`
- :py:func:`~ecoli.analysis.multivariant.new_gene_translation_efficiency_heatmaps.avg_1d_array_sql`
- :py:func:`~ecoli.analysis.multivariant.new_gene_translation_efficiency_heatmaps.avg_sum_1d_array_sql`
- :py:func:`~ecoli.analysis.multivariant.new_gene_translation_efficiency_heatmaps.avg_1d_array_over_scalar_sql`
- :py:func:`~ecoli.analysis.multivariant.new_gene_translation_efficiency_heatmaps.avg_sum_1d_array_over_scalar_sql`


---------------------
Other Workflow Output
---------------------

We provide helper functions in :py:mod:`ecoli.library.parquet_emitter` to read other
workflow output.

- :py:func:`~ecoli.library.parquet_emitter.open_arbitrary_sim_data`: Intended for use
  in analysis scripts. Accepts the ``sim_data_paths`` dictionary given as input to
  analysis scripts by :py:mod:`runscripts.analysis` and picks a single arbitrary
  path in that dictionary to read and unpickle.
- :py:func:`~ecoli.library.parquet_emitter.open_output_file`: When opening any
  workflow output file in a Python script, use this function instead of the built-in
  ``open`` (e.g. ``with open_output_file({path}, "r") as f:``). This is mainly
  intended to future-proof analysis scripts for Google Cloud support.
