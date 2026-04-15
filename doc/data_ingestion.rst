=========================
Experimental Data Ingestion
=========================

This document describes how to provide custom experimental data to the model,
allowing users to substitute their own measurements for the reference data
shipped with the repository.

--------
Overview
--------

The vEcoli model ships with curated reference data in ``reconstruction/ecoli/flat/``.
This data was compiled from public databases and literature and represents a
"default" *E. coli* K-12 MG1655 grown in M9 minimal medium with glucose.

In many cases, users want to parameterize the model with their own experimental
measurements—for example, RNA-seq data from a different strain, growth condition,
or laboratory. The **experimental data ingestion** system provides a structured
way to do this without modifying the core reference files.

Philosophy
==========

The ingestion system follows these principles:

1. **Like-for-like substitution**: Custom data must match the format and semantics
   of the reference data it replaces. For RNA-seq, this means gene-level TPM values
   that can be mapped to the model's gene set.

2. **Schema validation**: All ingested data is validated against Pandera schemas
   (see :py:mod:`wholecell.io.schemas`) to catch formatting errors early.

3. **Manifest-based organization**: Datasets are registered in manifest files that
   provide metadata (source, strain, condition) alongside file paths. This keeps
   the data self-documenting.

4. **Config-driven selection**: Users specify which dataset to use via configuration
   options, making it easy to switch between datasets without code changes.

Currently Supported Data Types
==============================

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Data Type
     - Description
     - Status
   * - RNA-seq (transcriptome)
     - Gene-level TPM expression values
     - ✓ Supported
   * - Proteomics
     - Protein abundance measurements
     - Planned, near term
   * - Metabolomics
     - Metabolite concentrations
     - Under consideration
   * - Metabolic fluxes
     - Flux values
     - Under consideration
   * - Growth physiology
     - Growth rates, cell size, etc.
     - Under consideration

------
RNA-seq
------

RNA-seq data provides gene expression levels used by the ParCa (parameter calculator)
to set basal transcription rates. By default, ParCa uses expression data from the
reference files. With the ingestion system, you can substitute your own RNA-seq
measurements.

File Organization
=================

RNA-seq data is organized as:

.. code-block:: text

    reconstruction/
    └── ecoli/
        └── experimental_data/
            └── rnaseq/
                ├── manifest.tsv        # Lists all available datasets
                ├── ref_0001.tsv       # TPM table for dataset ref_0001
                ├── ref_0002.tsv       # TPM table for dataset ref_0002
                ├── gbw_0001.tsv       # TPM table for dataset gbw_0001
                └── ...

The ``manifest.tsv`` file is the entry point—it lists all available datasets and
their metadata. Each dataset has a corresponding TPM table file.

Manifest Schema
===============

The manifest is a tab-separated file validated against
:py:obj:`~wholecell.io.schemas.rnaseq.RnaseqSamplesManifestSchema`.

**Required columns:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Column
     - Type
     - Description
   * - ``dataset_id``
     - string
     - Unique identifier for this dataset (e.g., ``gbw_0001``). Referenced in config.
   * - ``dataset_description``
     - string
     - Human-readable description of the dataset.
   * - ``file_path``
     - string
     - Path to the TPM table file (relative to manifest or absolute).
   * - ``data_source``
     - string
     - Origin of the data (e.g., ``Ginkgo Bioworks``, ``PNNL``).

**Optional columns:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Column
     - Type
     - Description
   * - ``data_source_experiment_id``
     - string
     - Experiment identifier from the data source.
   * - ``data_source_date``
     - string
     - Date of the experiment (e.g., ``2026-01-15``).
   * - ``strain``
     - string
     - Strain descriptor (e.g., ``MG1655 rph+``).
   * - ``condition``
     - string
     - Cultivation condition (e.g., ``M9, Glucose, Aerobic, 37C``).

**Example manifest:**

.. code-block:: text

    dataset_id	dataset_description	file_path	data_source	strain	condition
    ref_0001	Reference M9 Glucose minus AAs	ref_0001.tsv	reference	MG1655	M9, Glucose, Aerobic
    gbw_0001	MG1655 rph+ in Modified M9	gbw_0001.tsv	Ginkgo Bioworks	MG1655 rph+	Modified_M9_N_Fe

TPM Table Schema
================

Each TPM table is a tab-separated file validated against
:py:obj:`~wholecell.io.schemas.rnaseq.RnaseqTpmTableSchema`.

**Required columns:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``gene_id``
     - string
     - Gene identifier matching the model's gene set (EcoCyc IDs, e.g., ``EG10001``).
   * - ``tpm_mean``
     - float
     - Mean TPM (transcripts per million) for this gene. Must be ≥ 0.

**Optional columns:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Column
     - Type
     - Description
   * - ``tpm_std``
     - float
     - Standard deviation of TPM across replicates. Must be ≥ 0.

**Example TPM table:**

.. code-block:: text

    gene_id	tpm_mean	tpm_std
    EG10001	1234.56	45.2
    EG10002	567.89	23.1
    EG10003	0.0	0.0
    ...

.. note::
   Gene IDs must match the EcoCyc identifiers used by the model. Genes not found
   in the model's gene set will be ignored with a warning.

Configuration
=============

To use custom RNA-seq data, add the following options under ``parca_options``
in your configuration JSON:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Type
     - Description
   * - ``rnaseq_manifest_path``
     - string
     - Path to the manifest TSV file.
   * - ``rnaseq_basal_dataset_id``
     - string
     - The ``dataset_id`` to use as the basal transcriptome.
   * - ``basal_expression_condition``
     - string
     - Modeled condition name (default: ``"M9 Glucose minus AAs"``).

**Example configuration:**

.. code-block:: json

    {
        "parca_options": {
            "cpus": 4,
            "outdir": "out/custom_rnaseq",
            "rnaseq_manifest_path": "$ECOLI_SOURCES/data/manifest.tsv",
            "rnaseq_basal_dataset_id": "vecoli_m9_glucose_minus_aas",
            "basal_expression_condition": "M9 Glucose minus AAs"
        }
    }

**Default behavior (backward compatible):**

If ``rnaseq_manifest_path`` is ``null`` or omitted, ParCa uses the legacy
reference data from ``reconstruction/ecoli/flat/rna_seq_data/``.

Private / non-public overlays
=============================

Additional RNA-seq datasets that cannot live in the public ``ecoli-sources``
repo (e.g. proprietary vendor data) can be kept in a sibling *overlay* repo
with the same ``data/`` layout (TSVs + a `manifest.tsv` validated by
``RnaseqSamplesManifestSchema``). Tell ParCa about overlays by exporting:

.. code-block:: bash

    export ECOLI_SOURCES=/path/to/ecoli-sources
    export ECOLI_SOURCES_OVERLAYS=/path/to/ecoli-sources-vegas/data/manifest.tsv

``ECOLI_SOURCES_OVERLAYS`` is a colon-separated list of overlay manifest
paths; each path may use ``$ECOLI_SOURCES`` or ``~`` expansion. The loader
concatenates the primary + overlay manifests; ``dataset_id`` must be unique
across the union (collisions raise ``ValueError``). Configs don't need to
change — ``"rnaseq_manifest_path": "$ECOLI_SOURCES/data/manifest.tsv"`` works
regardless of whether overlays are active, and dataset ids from overlays
simply resolve transparently.

Validation Errors
=================

The ingestion system validates data early and provides clear error messages:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Error
     - Cause
   * - ``ValueError: rnaseq_manifest_path is set but rnaseq_basal_dataset_id is None``
     - You specified a manifest but forgot to specify which dataset to use.
   * - ``FileNotFoundError: ...``
     - The manifest file or a TPM table file doesn't exist.
   * - ``KeyError: Dataset_id 'xyz' not found in manifest``
     - The ``dataset_id`` you specified isn't in the manifest.
   * - ``SchemaError: ...``
     - A file doesn't match the expected schema (missing columns, wrong types, etc.).

-----------
Python API
-----------

For programmatic access, use the functions in :py:mod:`wholecell.io.ingestion`:

.. code-block:: python

    from wholecell.io.ingestion import (
        ingest_rnaseq_manifest,
        ingest_rnaseq_tpm_table,
        ingest_transcriptome,
    )

    # Load and validate a manifest
    manifest = ingest_rnaseq_manifest("$ECOLI_SOURCES/data/manifest.tsv")

    # Load a single TPM table
    tpm_df = ingest_rnaseq_tpm_table("$ECOLI_SOURCES/data/vecoli_m9_glucose_minus_aas.tsv")

    # Convenience: load a dataset by ID (validates manifest + TPM table)
    tpm_df, metadata = ingest_transcriptome(
        "$ECOLI_SOURCES/data/manifest.tsv",
        dataset_id="vecoli_m9_glucose_minus_aas"
    )

-------------------
Adding Your Own Data
-------------------

To add your own RNA-seq data:

1. **Prepare your TPM table** as a tab-separated file with ``gene_id`` and ``tpm_mean``
   columns. Ensure gene IDs are EcoCyc identifiers.

2. **Place the file** in the sibling ``ecoli-sources/data/`` repo (or set ``$ECOLI_SOURCES`` to a directory that contains your manifest + TSVs).

3. **Add an entry to the manifest** with a unique ``dataset_id``, description,
   and the path to your file.

4. **Update your config** to point to the manifest and specify your ``dataset_id``.

5. **Run ParCa** to generate new simulation parameters using your data.

**Example workflow:**

.. code-block:: bash

    # 1. Add your TPM file
    cp my_experiment_tpm.tsv $ECOLI_SOURCES/data/my_exp_001.tsv

    # 2. Edit $ECOLI_SOURCES/data/manifest.tsv to add a row for my_exp_001

    # 3. Create a config file
    cat > configs/my_experiment.json << 'EOF'
    {
        "parca_options": {
            "cpus": 4,
            "outdir": "out/my_experiment",
            "rnaseq_manifest_path": "$ECOLI_SOURCES/data/manifest.tsv",
            "rnaseq_basal_dataset_id": "my_exp_001"
        }
    }
    EOF

    # 4. Run ParCa
    python runscripts/parca.py --config configs/my_experiment.json

----------
References
----------

- Schemas: :py:mod:`wholecell.io.schemas.rnaseq`
- Ingestion functions: :py:mod:`wholecell.io.ingestion`
- ParCa configuration: :ref:`/workflows.rst`
