=========
Workflows
=========

A typical simulation workflow has four main steps:

1. Run the parameter calculator (:py:mod:`runscripts.parca`) to generate simulation parameters from raw data.
2. Create one or more variants (:py:mod:`runscripts.create_variants`) of the simulation parameters.
3. Simulate cells (:py:mod:`runscripts.sim` wraps :py:mod:`ecoli.experiments.ecoli_master_sim`).
4. Aggregate simulation results with analysis scripts (:py:mod:`runscripts.analysis`).

While each of these steps can be run manually by invoking their associated scripts,
it is common to run them in an automatically coordinated Nextflow workflow
using :py:mod:`runscripts.workflow`.

As mentioned in :ref:`json_config`, the preferred method for supplying configuration
options to runscripts is via a JSON configuration file specified using the ``--config``
command-line option. Please check the configuration JSON located at
:py:attr:`~ecoli.experiments.ecoli_master_sim.SimConfig.default_config_path`
for the most up-to-date default configuration options. 

.. tip::

  Template configuration JSON files for running :py:mod:`runscripts.parca`,
  :py:mod:`runscripts.create_variants`, and :py:mod:`runscripts.analysis`
  standalone are located in ``configs/templates``. See ``configs/default.json``
  for a template :py:mod:`runscripts.sim`/:py:mod:`ecoli.experiments.ecoli_master_sim`
  configuration JSON.

.. note::
    Remember that when creating your own JSON configuration
    file, you only need to include the configuration options whose values are
    different from the defaults.

.. warning::
  ``~`` and environment variables like ``$HOME`` are not expanded in the
  configuration JSON. Use ``echo your_path`` to get a full path that you can
  use in the JSON. For example, ``echo ~/out`` or ``echo $HOME/test``.

Below, we explore each runscript in further detail, including the
configuration options unique to each.

-----
ParCa
-----

The parameter calculator (or ParCa) is a Python script that performs certain computations
on raw, curated experimental data (located in the ``reconstruction/ecoli/flat`` folder)
to generate the parameters expected by processes in our model. It packages these parameters
in a pickled :py:class:`~reconstruction.ecoli.simulation_data.SimulationDataEcoli` object
whose path must be given via the ``sim_data_path`` configuration option to all runscripts
in ``runscripts/`` and to experiment modules in ``ecoli/experiments`` (default used by
:py:mod:`runscripts.workflow` is :py:mod:`~ecoli.experiments.ecoli_master_sim`).

The code responsible for loading data from the raw flat files is contained in
:py:class:`~reconstruction.ecoli.knowledge_base_raw.KnowledgeBaseEcoli`. The actual logic
of the ParCa is mostly contained within a single file: :py:mod:`~reconstruction.ecoli.fit_sim_data_1`.
The main interface for running the ParCa is :py:mod:`runscripts.parca`.

Configuration
=============

Configuration options for the ParCa are all located in a dictionary under the
``parca_options`` key inside a JSON configuration file. They include:

- ``cpus``: Number of CPU cores to parallelize parts of the ParCa over
- ``outdir``: Path to directory (local or absolute) for output pickle files
- ``operons``: If True, calculate parameters with operon gene structure
- ``ribosome_fitting``: If False, ribosome expression is not fit to protein synthesis demands
- ``rnapoly_fitting``: If False, RNA polymerase expression is not fit to protein synthesis demands
- ``remove_rrna_operons``: If True, do not include rRNAs in operon gene structures.
- ``remove_rrff``: If True, remove rrfF gene that encodes for the extra 5S rRNA in the rrnD operon
- ``stable_rrna``: If True, set degradation rates of mature rRNAs
  to the values calculated from the half-life in sim_data.constants. If False,
  set degradation rates of mature rRNAs to the average reported degradation rates of mRNAs.
- ``new_genes``: String folder name in ``reconstruction/ecoli/flat/new_gene_data``
  containing necessary flat files to add new gene(s) to the model (see templates in
  ``reconstruction/ecoli/flat/new_gene_data/template``). By default, ``off`` does
  nothing (no new genes).
- ``debug_parca``: If True, fit only one arbitrarily-chosen transcription
  factor in order to speed up a debug cycle.
- ``save_intermediates``: Save intermediate pickle files for each major
  step of the ParCa (:py:func:`~reconstruction.ecoli.fit_sim_data_1.initialize`,
  :py:func:`~reconstruction.ecoli.fit_sim_data_1.input_adjustments`,
  :py:func:`~reconstruction.ecoli.fit_sim_data_1.basal_specs`,
  :py:func:`~reconstruction.ecoli.fit_sim_data_1.tf_condition_specs`,
  :py:func:`~reconstruction.ecoli.fit_sim_data_1.fit_condition`,
  :py:func:`~reconstruction.ecoli.fit_sim_data_1.promoter_binding`,
  :py:func:`~reconstruction.ecoli.fit_sim_data_1.adjust_promoters`,
  :py:func:`~reconstruction.ecoli.fit_sim_data_1.set_conditions`,
  :py:func:`~reconstruction.ecoli.fit_sim_data_1.final_adjustments`).
- ``intermediates_directory``: Path to folder where intermediate pickle files
  should be saved or loaded.
- ``load_intermediate``: The function name of the ParCa step to load
  sim_data and cell_specs from; functions prior to and including this one
  will be skipped but all subsequent functions will run. Can only be used
  if all ParCa steps up to and including named step were previously run
  successfully with ``save_intermediates`` set to True.
- ``variable_elongation_transcription``: If True, enable variable elongation
  for transcription.
- ``variable_elongation_translation``: If True, enable variable elongation
  for translation.

.. note::
  If the top-level ``sim_data_path`` option is not null, the ParCa is skipped
  in favor of the pickled simulation data at the specified path. This applies
  regardless of whether running with :py:mod:`runscripts.workflow` or
  :py:mod:`runscripts.parca`.

.. warning::
  If running :py:mod:`runscripts.parca` and :py:mod:`ecoli.experiments.ecoli_master_sim`
  manually instead of using :py:mod:`runscripts.workflow`, you must create two config JSON
  files: one for the ParCa with a null ``sim_data_path`` and an ``outdir``
  as described above and one for the simulation with
  ``sim_data_path`` set to ``{outdir}/kb/simData.cPickle``. This is intentional to
  reduce the chance that the incorrect simulation data is used.

.. _variants:

--------
Variants
--------

In many cases, we would like to use the model to answer biological questions that
require running the model with different parameters. For example, we may want to
see how a cell responds differently when grown in different media conditions.
Since most process parameters in our model come from the pickled
:py:class:`~reconstruction.ecoli.simulation_data.SimulationDataEcoli` generated by
the ParCa, we need an easy way to modify this object. The
:py:mod:`runscripts.create_variants` script was designed for this purpose.

Template
========

In essence, this script runs a "variant function" with one or more input
parameter combinations, with each invocation independently modifying the
baseline :py:class:`~reconstruction.ecoli.simulation_data.SimulationDataEcoli`
object in some way. Variant functions are contained within Python files
located in the ``ecoli.variants`` folder. They all have an ``apply_variant``
function that follows the following template:

.. code-block:: python

    from typing import Any, TYPE_CHECKING

    # This if statement prevents Python from unnecessarily importing this
    # object when it is only needed for type hinting
    if TYPE_CHECKING:
        from reconstruction.ecoli.simulation_data import SimulationDataEcoli

    def apply_variant(
        sim_data: "SimulationDataEcoli", params: dict[str, Any]
    ) -> "SimulationDataEcoli":
        """
        Modify sim_data using parameters from params dictionary.

        Args:
            sim_data: Simulation data to modify
            params: Parameter dictionary of the following format::

                {
                    "{name of parameter}": {type of parameter},
                    ...
                }

        Returns:
            Simulation data with the following attributes modified::

                {attributes of sim_data that this function changes}

        """
        # Modify sim_data as you see fit using params. Following is example
        sim_data.attribute = params["param_1"]
        return sim_data

Configuration
=============

When running :py:mod:`runscripts.create_variants`, users must specify the
variant function to use under the ``variants`` key in the configuration JSON
following the general template:

.. code-block::

    {
        "variants": {
            "{name of variant function}": {
                {variant function parameters}
            }
        }
    }

The name of each variant function is the name of the file containing its
``apply_variant`` function. For example, to use the variant function
:py:mod:`ecoli.variants.new_gene_internal_shift`, provide the name
``new_gene_internal_shift``. If the ``variants`` key points to an
empty dictionary (no variants), then only the only "variant" saved
by :py:mod:`runscripts.create_variants` is the unmodified simulation
data object. Thus, when running a workflow with :py:mod:`runscripts.workflow`,
at least one lineage of cells will always be run with the baseline
``sim_data``. To avoid this (e.g. when running many batches of simulations
with the same variant function), set the top-level ``skip_baseline`` option
to ``True``.

.. warning::
    Only one variant function is supported at a time.

If you would like to modify the simulation data object using multiple
variant functions, create a new variant function that invokes the desired
combination of ``apply_variants`` methods from other variant functions.

The format of the variant function parameters is described in
:py:func:`~runscripts.create_variants.parse_variants`. By using the
``op`` key, you can concisely generate a large array of parameter
combinations, each of which results in the creation of a variant of the
:py:class:`~reconstruction.ecoli.simulation_data.SimulationDataEcoli`
object.

When manually running :py:mod:`runscripts.create_variants` (as opposed to
running :py:mod:`runscripts.workflow`), the configuration file must also include:

- Top-level (not under ``variants`` key) ``outdir`` option: path to directory
  in which to save variant simulation data objects as pickle files
- Top-level (not under ``variants`` key) ``kb`` option: path to directory
  containing ParCa output pickle files

.. _variant_output:

Output
======

The generated variant simulation data objects are pickled and saved in the
directory given in the ``outdir`` key of the configuration JSON.
They all have file names of the format ``{index}.cPickle``, where
index is an integer. If the top-level ``skip_baseline`` option is not set
to ``True``, the unmodified simulation data object is always
saved as ``0.cPickle``. Otherwise, the 0 index is skipped. The identity of
the other indices can be determined by referencing the ``metadata.json``
file that is also saved in ``outdir``. This JSON maps the variant function
name to a mapping from each index to the exact parameter
dictionary passed to the variant function to create the
variant simulation data saved with that index as its file name. See
:py:func:`~runscripts.create_variants.apply_and_save_variants` for
more details.

-----------
Simulations
-----------

Refer to :ref:`/experiments.rst` for more information about the main
script for running single-cell simulations,
:py:mod:`~ecoli.experiments.ecoli_master_sim`.


.. _analysis_scripts:

--------
Analyses
--------

The :py:mod:`runscripts.analysis` script is the main interface for running
analyses on simulation output data. Importantly, to use this interface,
simulations must be run (whether with :py:mod:`~ecoli.experiments.ecoli_master_sim`
or :py:mod:`runscripts.workflow`) with the ``emitter`` option set to ``parquet``
and an output directory set using the ``out_dir`` or ``out_uri`` key under the
``emitter_arg`` option (see :ref:`json_config`). This tells vivarium-core to use
:py:mod:`~ecoli.library.parquet_emitter.ParquetEmitter` to save the simulation
output as Hive partitioned Parquet files. See :ref:`/output.rst` for more
details on Parquet and DuckDB, the primary library used to interact with the
saved files.

Analysis scripts must be one of the following types and placed into the
corresponding folder:

- :py:mod:`~ecoli.analysis.single`: Limited to data for a single simulated cell
- :py:mod:`~ecoli.analysis.multidaughter`: Limited to data for daughter cell(s)
  of a single mother cell
- :py:mod:`~ecoli.analysis.multigeneration`: Limited to data for all cells across
  many generations for a given initial seed, variant simulation data object, and
  workflow run (same experiment ID, see :ref:`/experiments.rst`)
- :py:mod:`~ecoli.analysis.multiseed`: Limited to data for all cells across many
  generations and initial seeds for a given variant simulation data object and
  workflow run
- :py:mod:`~ecoli.analysis.multivariant`: Limited to data for all cells across
  many generations, initial seeds, and variant simulation data objects for a given
  workflow run
- :py:mod:`~ecoli.analysis.multiexperiment`: Limited to data for all cells across
  many generations, initial seeds, variant simulation data objects, and workflow runs

.. note::
    These categories represent upper bounds on the data that can be accessed.

A ``multiseed`` analysis, for example, can choose to only read data
from cells between generations 4 and 8 from the cells with the same
experiment ID, variant simulation data object, and initial seed that
it has access to.

.. tip::
  If you would like to use an analysis script with many different scopes,
  instead of duplicating the entire script in each analysis type
  folder, you can just create stub files in the appropriate folders
  that simply import the ``plot`` function from a primary analysis script.

.. _analysis_config:

Configuration
=============

The :py:mod:`runscripts.analysis` script accepts the following configuration
options under the ``analysis_options`` key:

- ``single``, ``multidaughter``, ``multigeneration``, ``multiseed``, ``multivariant``
  ``multiexperiment``: Can pick one or more analysis types to run. Under each analysis
  type is a sub-dictionary of the following format:

    .. code-block::

        {
            "{analysis name}": {optional dictionary of analysis parameters},
            # Example:
            "mass_fraction_summary": {"font_size": 12}
        }
    
  The name of an analysis is simply its file name without the ``.py`` extension.
- ``experiment_id``, ``variant``, ``lineage_seed``, ``generation``, ``agent_id``:
  List of experiment IDs, variant indices, etc. to filter data to before running
  analyses. Note that experiment IDs and agent IDs are strings while the rest are
  integers. ``experiment_id`` is required while the others are optional. If not
  provided, :py:mod:`runscripts.analysis` simply does not filter data by variant
  indices, initial seeds, etc. before running analyses.
- ``variant_range``, ``lineage_seed_range``, ``generation_range``: List of length
  2 where the first element is the start and the second element is the end (exclusive)
  of a range of variant indices, initial seeds, or generations to filter data to
  before running analyses. Overrides corresponding non-range options.
- ``sim_data_path``: List of string paths to simulation data pickle files. If multiple
  variants are given via ``variant`` or ``variant_range``, you must provide same number
  of paths in the same order using this option. This option is mainly meant for internal use.
  For a simpler alternative that also works if multiple experiment IDs are given with
  ``experiment_id`` (variant indices may correspond to completely different variant
  simulation data objects in different workflow runs), see ``variant_data_dir``.
- ``variant_metadata_path``: String path to ``metadata.json`` file saved by
  :py:mod:`runscripts.create_variants` (see :ref:`variant_output`). This option is mainly
  intended for internal use. For a simpler alternative that also works if multiple
  experiment IDs are given via ``experiment_id``, see ``variant_data_dir``.
- ``variant_data_dir``: List of string paths to one or more directories containing
  variant simulation data pickles and metadata saved by :py:mod:`runscripts.create_variants`.
  Must provide one path for each experiment ID in ``experiment_ID`` and in the
  same order.
- ``validation_data_path``: List of string paths to validation data pickle files
  (generated by ParCa). Can pass any number of paths in any order and they will be
  passed as is to analysis script ``plot`` functions.
- ``outdir``: Local (relative or absolute) path to directory that serves as a prefix
  to the ``outdir`` argument for analysis script ``plot`` functions
  (see :ref:`analysis_template`). A copy of the configuration options
  used to run :py:mod:`runscripts.analysis` is saved as ``outdir/metadata.json``.
- ``cpus``: Number of CPU cores to let DuckDB use. DuckDB generally scales well
  with more cores at the cost of proportionally increased RAM usage (default: 1)
- ``analysis_types``: List of analysis types to run. By default (if this option
  is not used), all analyses provided under all the analysis type keys are run
  on all possible subsets of the data after applying the data filters given using
  ``experiment_id``, ``variant``, etc. For example, say 2 experiment IDs are
  given with ``experiment_id``, 2 variants with ``variant``, 2 seeds with ``lineage_seed``,
  and 2 generations with ``generation``. Assuming no simulations failed and ``single_daughter``
  was set to True, analyses under the ``multiexperiment`` key (if any) will each run once
  with all data passing this filter. ``multivariant`` analyses will each run twice, first
  with filtered data for one experiment ID then with filtered data for the other. ``multiseed``
  analyses will each run 4 times (2 exp IDs * 2 variants), ``multigeneration`` analyses
  8 times (4 * 2 seeds), ``multidaughter`` analyses 16 times (8 * 2 generations), and
  ``single`` analyses 16 times. If you only want to run the ``single`` and ``multivariant``
  analyses, specify ``["single", "multivariant"]`` using this option.


.. note::
  You must also have the ``emitter_arg`` key in your config JSON with a ``out_dir`` or
  ``out_uri`` set to the location where the analysis script will look for simulation
  data output.

.. _analysis_template:

Template
========

All analysis scripts must contain a ``plot`` function with the following signature:

.. code-block:: python

    from typing import Any, TYPE_CHECKING

    if TYPE_CHECKING:
        from duckdb import DuckDBPyConnection

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
        """
        Args:
            params: Dictionary of parameters given under analysis
                name in configuration JSON.
            conn: DuckDB database connection, automatically created
                by runscripts/analysis.py with appropriate settings.
            history_sql: DuckDB SQL query that filters simulation
                output data to subset appropriate for analysis type
                (e.g. single cell for ``single`` analyses).
            config_sql: DuckDB SQL query that filters simulation
                config data to subset appropriate for analyis type.
            success_sql: DuckDB SQL query to Hive-partitioned
                Parquet dataset which only contains successful sims.
            sim_data_paths: Mapping from experiment IDs to mapping
                from variant indices to variant simulation data
                pickle paths. Generated by runscripts/analysis.py
                either using:

                    - Combination of  ``sim_data_path``, ``variant``,
                      ``variant_metadata_path``, and ``experiment_id``
                      configuration options
                    - Traversing directories in ``variant_data_dir`` and
                      matching discovered variants with experiment IDs
                      given in ``experiment_id`` (preferred route for
                      most use cases)

            validation_data_paths: List of validation data pickle
                paths taken directly from ``validation_data_path``
                configuration option.
            outdir: String path equal to ``outdir`` configuration option
                prepended to Hive partitioned directory representing data
                filters applied to ``history_sql``. For example, a ``single``
                analysis script run on data for experiment ID "test",
                variant index 1, lineage seed 3, generation 2, and agent ID
                "00" will get: ``{outdir}/experiment_id=test/variant=1/
                lineage_seed=3/generation=2/agent_id=00``. By convention,
                analysis scripts should save their outputs in this folder.
            variant_metadata: Mapping from experiment IDs to mapping
                from variant indices to parameters used to create variant
                simulation data object. Generated by runscripts/analysis.py
                in one of the same two methods used for sim_data_paths.
            variant_names: Mapping from experiment IDs to name of variant
                function used to generate variant simulation data objects
                for workflow run. Generated by runscripts/analysis.py
                in one of the same two methods used for sim_data_paths.
        """

Refer to :ref:`/output.rst` for more information about how
to use DuckDB to read and analyze simulation output inside
analysis scripts.

---------
Workflows
---------

`Nextflow <https://www.nextflow.io>`_ is a piece of software that abstracts the
complexity of orchestrating complex workflows on a variety of supported
platforms, including personal computers, computing clusters, and even cloud
computing services. :py:mod:`runscripts.workflow` uses the template Nextflow
workflow scripts located in the ``runscripts/nextflow`` folder along with
an input configuration JSON to create and run a complete workflow with all of
the steps described above.

Configuration
=============

All the previously covered configuration options also apply to the configuration
JSON supplied to :py:mod:`runscripts.workflow`. Those options govern the behavior
of the corresponding step in the workflow. For example, running
:py:mod:`runscripts.workflow` with ``cpus`` under ``parca_options``
set to 4 will start the workflow by running the ParCa with 4 CPUs.

After creating some number of variant simulation data objects with
:py:mod:`runscripts.create_variants`, the workflow automatically
starts at least one cell simulation for each variant using
:py:mod:`~ecoli.experiments.ecoli_master_sim`. The exact number
of simulations started per variant is configured by the following
top-level configuration options:

- ``n_init_sims``: Number of replicate simulations to run for each variant,
  where replicates differ in the initial seed used to initialize them
- ``lineage_seed``: Each integer in the half-open interval
  ``[lineage_seed, lineage_seed + n_init_sims)`` is used to initialize
  the first generation of a lineage, where a lineage is defined
  as a group of cell simulations with the same first generation initial
  seed (called a lineage seed) and variant simulation data object
- ``generations``: Integer number of generations to run each cell lineage
- ``single_daughters``: If False, simulates both daughter cells (append ``0``
  to mother agent ID to get one daughter agent ID and ``1`` to get other) after
  cell division. Otherwise, continue lineage with one arbitrary daughter cell
  state (append ``0`` to mother agent ID to get daughter agent ID)

This means that if a workflow is run with ``n_init_sims`` set to 4, ``generations``
set to 10, ``single_daughters`` set to True, and ``variant_options``
configured to create 4 different variant simulation data objects (5 including
baseline, unmodified ``0.cPickle``, see :ref:`variant_output`),
``4 * 10 * 1 * 5 = 200`` total simulations will run. This is assuming no lineages fail
before reaching 10 generations due to ``fail_at_max_duration`` (see :ref:`json_config`)
or some other uncaught exception.

Unlike when running :py:mod:`runscripts.analysis` manually, the configuration JSON
supplied to :py:mod:`runscripts.workflow` only needs to provide the names and
parameters for analysis scripts to run using the analysis type options (e.g.
``single``, ``multivariant``, etc.) and can omit the other options documented
in :ref:`analysis_config`. This is because the Nextflow workflow is configured
to automatically pass the other required parameters like the paths to the variant
simulation data pickles created by :py:mod:`runscripts.create_variants` earlier
in the worklow.


.. _progress:

--------
Progress
--------

There are three main ways to monitor a workflow's progress.

#. Check the command-line output of the Nextflow orchestrator. On a
   personal computer, Nextflow will periodically print its progress
   to the command line. On Sherlock, this output is written to the
   ``slurm-{job ID}.out`` file in the directory you started the workflow from.
#. Open the file named ``trace--{experiment ID}--{timestamp}.csv``
   in the directory you started the workflow from. This file contains
   information about completed processes as they complete. Note that
   ``submit,start,complete,duration,realtime`` are reported in ms and
   ``rss,peak_rss`` are reported in bytes.
#. Open the file named ``.nextflow.log`` in the directory you started the
   workflow from. This is a fairly verbose and technical log
   that may be useful for debugging purposes.

.. danger::
    Any changes that are made to the cloned repository while a workflow is running
    **on a local computer** will immediately affect workflow jobs submitted after
    the change. For example, modifying ``runscripts/analysis.py`` will affect all
    subsequent analysis jobs in a running workflow. This does not apply to workflows
    run on :doc:`Google Cloud <../gcloud>` or :doc:`Sherlock <../hpc>`, where
    a snapshot of the repository is packaged into the container image used to
    run the workflow.

The warning above only applies to files in the repository that are actively executed or
used during a workflow (ParCa, variant creation, simulation, analysis). Notably,
you can freely create, modify, and delete configuration JSON files in the cloned
repository and use them to launch concurrent workflows.

If this is not sufficient, you can create additional clones of the repository
under different directory names, modify them, and use them to launch workflows.
To reduce the size of each clone, use ``git clone --filter=blob:none {URL} {output path}``
to create a blobless clone, which downloads file contents only for the latest
commit. File contents (blobs) for other commits are downloaded on-demand upon
checkout.

.. _fault_tolerance:

---------------
Fault Tolerance
---------------

Nextflow workflows can be configured to be highly fault tolerant. The following
is a list workflow behaviors enabled in our model to handle unexpected errors.

- When running on Sherlock, jobs that fail with exit codes 137 or 140 (job
  limits for RAM or runtime) or 143 (job was preempted by another user)
  are automatically retried up to a maximum of 3 times. For the resource
  limit exit codes, Nextflow will automatically request more RAM
  and a higher runtime limit with each attempt: ``4 * {attempt num}``
  GB of memory and ``1 * {attempt num}`` hours of runtime. See the
  ``sherlock`` profile in ``runscripts/nextflow/config.template``.
- Additionally, some jobs may fail on Sherlock due to issues submitting
  them to the SLURM scheduler. Nextflow was configured to limit the rate
  of job sumission and job queue polling to keep these failures to a
  minimum. Furthermore, jobs that fail to submit are automatically
  retried with a relatively long 5 minute delay to hopefully avoid
  any transient scheduler issues.
- Jobs that fail for any reason other than the Sherlock reasons described
  above are ignored. This is mainly to allow a workflow to finish running
  all programmed cell simulations even if some cells fail, terminating their
  corresponding lineages. For example, if generation 6 for a given initial
  seed and variant simulation data failed, then generation 7+ for that lineage
  cannot run but the lineages for different initial seeds and/or variant
  simulation data can still continue to run.
- If you realize that a code issue is the cause of job failure(s), stop
  the workflow run if it is not already (e.g. ``control + c``, ``scancel``,
  etc.), make the necessary code fixes, and rerun :py:mod:`runscripts.workflow`
  with the same configuration JSON and the ``--resume`` command-line argument,
  supplying the experiment ID (with time suffix if using ``suffix_time`` option).
  Nextflow will intelligently resume workflow execution from the last successful
  job in each chain of job dependencies (e.g. generation 7 of a cell lineage
  depends on generation 6, :py:mod:`runscripts.create_variants` depends on
  :py:mod:`runscripts.parca`, etc).

.. _output:

------
Output
------

A completed workflow will have the following directory structure underneath
the output directory specified via ``out_dir`` or ``out_uri`` under the
``emitter_arg`` option in the configuration JSON (see :ref:`json_config`):

- ``{experiment ID}``: Folder in output directory named the experiment ID
  for the workflow. Allows many workflows to use the same output directory
  without overwriting data as long as they have different experiment IDs.

    - ``history``: Hive-partitioned Parquet files of
      simulation output. See :ref:`/output.rst`.
    - ``configuration``: Hive-partitioned Parquet files
      of simulation configs. See :ref:`/output.rst`.
    - ``success``: Hive-partitioned Parquet files that
      only exist for successful simulations.
    - ``parca``: Pickle files saved by :py:mod:`runscripts.parca`.

        - ``simData.cPickle``: :py:class:`~reconstruction.ecoli.simulation_data.SimulationDataEcoli`
        - ``validationData.cPickle``: :py:class:`~validation.ecoli.validation_data.ValidationDataEcoli`
        - ``rawData.cPickle``: :py:class:`~reconstruction.ecoli.knowledge_base_raw.KnowledgeBaseEcoli`
        - ``rawValidationData.cPickle``: :py:class:`~validation.ecoli.validation_data_raw.ValidationDataRawEcoli`

    - ``variant_sim_data``: Output of :py:mod:`runscripts.create_variants`.

        - ``0.cPickle``: Unmodified, baseline simulation data object.
        - ``metadata.json``: Mapping from variant function name to mapping from variant
          indices to parameter dictionaries used to create them.
        - ``1.cPickle``, ``2.cPickle``, etc: Variant simulation data objects, if any.

    - ``daughter_states``: Hive-partitioned (with experiment ID partition omitted
      because all files are for the same experiment ID) directory structure containing
      daughter cell initial states as JSON files.
    - ``analysis``: Hive-partitioned (with experiment ID partition omitted because
      all files are for the same experiment ID) directory structure containing
      output of analysis scripts in folder named ``plot`` at the level corresponding
      to the analysis type (e.g. ``multigeneration`` analysis output will be
      in sub-folders of the format ``variant={}/lineage_seed={}/plot``). Each ``plot``
      folder also contains a ``metadata.json`` file with the configuration.
      options passed to :py:mod:`runscripts.analysis` for the output in that folder.
    - ``nextflow``: Nextflow-related files.

        - ``main.nf``: Nextflow workflow script.
        - ``nextflow.config``: Nextflow workflow configuration.
        - ``{experiment ID}_report.html``: Detailed information about workflow run.
        - ``workflow_config.json``: Configuration JSON passed to
          :py:mod:`runscripts.workflow`.
        - ``nextflow_workdirs``: Contains all working directories for Nextflow jobs.
          Required for resume functionality described in :ref:`fault_tolerance`. Can
          also go to work directory for a job (consult files described in :ref:`progress`
          or ``{experiment ID}_report.html``) for debugging. See :ref:`make_and_test`
          for more information.

.. tip::
  To save space, you can safely delete ``nextflow_workdirs`` after you are finished
  troubleshooting a particular workflow.


.. _troubleshooting:

---------------
Troubleshooting
---------------

To troubleshoot a workflow that was run with :py:mod:`runscripts.workflow`, you can
either inspect the HTML execution report ``{experiment ID}_report.html`` described
in :ref:`output` (nice summary and UI) or use the ``nextflow log`` command
(more flexible and efficient).

HTML Report
===========

Click "Tasks" in the top bar or scroll to the bottom of the page. Filter for failed
jobs by putting "failed" into the search bar. Find the work directory (``workdir`` column)
for each job. Navigate to the work directory for each failed job and
inspect ``.command.out`` (``STDOUT``), ``.command.err`` (``STDERR``), and
``.command.log`` (both) files.

CLI
===

Run ``nextflow log`` in the same directory in which you launched
the workflow to get the workflow name (should be of the form
``{adjective}_{famous last name}``). Use the ``-f`` and ``-F``
flags of ``nextflow log`` to show and filter the information that
you would like to see (``nextflow log -help`` for more info).

Among the fields that can be shown with ``-f`` are the ``stderr``,
``stdout``, and ``log``. This allows you to automatically retrieve
relevant output for all failed jobs in one go instead of manually
navigating to work directories and opening the relevant text files.

For more information about ``nextflow log``, see the
`official documentation <https://www.nextflow.io/docs/latest/reports.html#reports>`_.
For a description of some fields (non-exhaustive) that can be specified with
``-f``, refer to `this section <https://www.nextflow.io/docs/latest/reports.html#trace-fields>`_
of the official documentation.

As an example, to see the name, stderr, and workdir for all failed jobs
in a workflow called ``agitated_mendel``:

.. code-block:: bash

  nextflow log agitated_mendel -f name,stderr,workdir -F "status == 'FAILED'"


.. _make_and_test:

Make and Test Fixes
===================

If you need to further investigate an issue, the exact steps differ depending
on where you are debugging.

- Google Cloud: See :ref:`instructions here <interactive-containers>`
- Sherlock: See :ref:`instructions here <sherlock-interactive>`
- Local machine: Continue below

Add breakpoints to any Python file with the following line:

.. code-block:: python

  import ipdb; ipdb.set_trace()

Figure out the working directory (see :ref:`troubleshooting`) for a
failing process. Navigate to the working directory and run:

.. code-block:: bash

  uvenv bash .command.sh

This should re-run the job and pause upon reaching the breakpoints you
set. You should now be in an ipdb shell which you can use to examine
variable values or step through the code.

After fixing the issue, you can resume the workflow (avoid re-running
already successful jobs) by navigating back to the directory in which you
originally started the workflow and running :py:mod:`runscripts.workflow`
with the ``--resume`` option (see :ref:`fault_tolerance`).
