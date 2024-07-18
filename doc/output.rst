======
Output
======

For simulations run using :py:mod:`~ecoli.experiments.ecoli_master_sim`
and the ``emitter`` configuration option set to ``timeseries``, simulation
output is kept in memory and can be accessed by calling the
:py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.query` method
of the :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim` object
used to run the simulation. See the source code for
:py:func:`~ecoli.experiments.metabolism_redux_sim.save_sim_output`
for an example of how this data can be manually saved for custom
downstream analyses.

There are many helper functions located inside
:py:mod:`ecoli.library.parquet_emitter` to interactRead simulation data using helper functions located in
ecoli.library.parquet_emitter like read_stacked_columns
and/or write custom DuckDB queries that build upon history_sql
and/or config_sql to retrieve the desired data or aggregates.
See ecoli.analysis.multivariant.new_gene_translation_efficiency_heatmaps
for many examples of complex aggregations performed by iteratively
constructing DuckDB queries.

When querying using DuckDB, you can control the format of the
returned data. If data is in a tabular format (e.g. PyArrow Table,
Pandas/Polars DataFrame), you can make use of hvplot to create
plots using a variety of backends (including interactive plots).
See ecoli.analysis.single.mass_fraction_summary for an example.
matplotlib can always be used for low-level control over plots.
