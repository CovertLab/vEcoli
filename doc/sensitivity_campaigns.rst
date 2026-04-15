====================
Sensitivity Campaigns
====================

Sensitivity campaigns use the multi-parca workflow to systematically perturb
vEcoli's input data and map how simulation behavior responds. The target
question is: given a reference dataset that produces a healthy cell, what is
the *region of acceptable input* and how does the model fail outside it?

See :doc:`data_ingestion` for the underlying ingestion plumbing and
``.claude/plans/dataset-sensitivity-exploration.md`` for the research design
this infrastructure was built to support.

---------
Overview
---------

Three repos collaborate to run a campaign:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Repo
     - Role
     - Key code
   * - ``vEcoli`` (this repo)
     - Model, workflow generator, campaign meta-runner, analyses
     - ``runscripts/run_sensitivity_campaign.py``,
       ``runscripts/workflow.py``,
       ``ecoli/analysis/multivariant/sensitivity_overview.py``
   * - ``ecoli-sources`` (sibling, public)
     - Primary RNA-seq datasets + manifest + perturbation operators + schemas
     - ``processing/perturbations.py``, ``schemas/``, ``data/manifest.tsv``
   * - ``ecoli-sources-*`` (sibling, private overlays)
     - Datasets that cannot live in the public repo
     - ``data/manifest.tsv``

Each perturbed dataset is produced deterministically from
``(source_dataset, operator, params, seed)``, so the spec (not the TSVs) is
the canonical artifact. Perturbations can be regenerated at any time; they
are gitignored in ``ecoli-sources/data/perturbations/``.

----------
Data layer
----------

Primary + overlay manifests
===========================

vEcoli's ingestion consumes one or more manifest files. The primary manifest
is referenced in the config as ``rnaseq_manifest_path``; additional overlay
manifests are loaded from the ``ECOLI_SOURCES_OVERLAYS`` environment variable
(``;``- or ``:``-separated list, URI-safe). ``dataset_id`` must be unique
across the union.

Cloud URIs (``s3://``, ``gs://``, ``gcs://``) are handled natively through
fsspec — ``rnaseq_manifest_path`` can point at either a local path or a
cloud URL.

Typical local layout::

    ecoli-sources/                      (public, git-tracked)
      data/
        manifest.tsv                    primary datasets
        vecoli_*.tsv, precise_*.tsv
        perturbations/                  (gitignored, regenerated from specs)
          add_log_normal_noise/
            vecoli_m9_glucose_minus_aas/
              *.tsv
      schemas/                          pandera validators
      processing/
        perturbations.py                operator library
        post_processing.py              dataset variants

    ecoli-sources-vegas/                (private, git-tracked)
      data/
        manifest.tsv
        gbw_vegas_*.tsv

Environment::

    export ECOLI_SOURCES=$HOME/code/ecoli-sources
    export ECOLI_SOURCES_OVERLAYS=$HOME/code/ecoli-sources-vegas/data/manifest.tsv

Config::

    "parca_options": {
      "rnaseq_manifest_path": "$ECOLI_SOURCES/data/manifest.tsv",
      "rnaseq_basal_dataset_id": "vecoli_m9_glucose_minus_aas"
    }

Schemas
=======

All manifests + TSVs are validated by pandera schemas in
``ecoli-sources/schemas/`` (``RnaseqSamplesManifestSchema``,
``RnaseqTpmTableSchema``, plus adjustment / parameter / regulation / half-life
schemas covering most of what used to live under
``reconstruction/ecoli/flat/``). Validate standalone with::

    uv run python -m schemas.validate AdjustmentValueSchema path/to/file.tsv

---------------------
Perturbation operators
---------------------

Operators live in ``ecoli-sources/processing/perturbations.py``. Each is a
pure function on a TPM DataFrame and deterministic given its RNG seed.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Operator
     - Arity
     - What it tests
   * - ``add_log_normal_noise(sigma, seed)``
     - unary
     - Robustness to measurement noise
   * - ``scale_gene_set(gene_ids, factor)``
     - unary
     - Inverse of manual expression adjustments
   * - ``zero_genes(gene_ids)``
     - unary
     - Load-bearing-gene identification
   * - ``drop_and_fill(fraction, seed)``
     - unary
     - Fill-from-ref code-path stress
   * - ``interpolate_datasets(tpm_a, tpm_b, alpha)``
     - binary
     - Walk from working → failing dataset
   * - ``quantile_match(source, target)``
     - binary
     - Shape-only vs. gene-identity effects

The ``make_(binary_)perturbation_variant`` driver applies an operator, writes
the perturbed TSV under ``data/perturbations/<operator>/<source>/``, and
appends a provenance row to the manifest with ``parent_dataset_id``,
``operator``, ``operator_params_json``, ``seed``.

--------------
Campaign specs
--------------

A campaign spec is a JSON document under
``configs/campaigns/<name>.spec.json`` with this shape::

    {
      "name": "pilot_expression_noise",
      "source_dataset_id": "vecoli_m9_glucose_minus_aas",
      "operator": "add_log_normal_noise",
      "param_grid": {
        "sigma": [0.1, 0.2, 0.4, 0.8],
        "seed":  [0, 1, 2]
      },
      "include_source_as_baseline": true,
      "base_config": "configs/test_multi_parca.json",
      "sim": {
        "generations": 3,
        "n_init_sims": 3,
        "analysis_options": {
          "multiseed": { "cd1_higher_order_properties": {} },
          "multivariant": {
            "sensitivity_overview": {
              "campaign_sidecar": "configs/campaigns/pilot_expression_noise.campaign.json"
            }
          }
        }
      }
    }

``param_grid`` is Cartesian-product expanded. Binary operators additionally
require ``"binary_partner": "<dataset_id>"``.

Generating the Nextflow config
==============================

``runscripts/run_sensitivity_campaign.py`` reads the spec, materializes the
perturbed TSVs in ``$ECOLI_SOURCES/data/perturbations/``, appends rows to the
manifest, and emits a Nextflow-ready config plus a sidecar::

    uv run runscripts/run_sensitivity_campaign.py \
        --spec configs/campaigns/pilot_expression_noise.spec.json

    # → configs/campaigns/pilot_expression_noise.json           (Nextflow config)
    # → configs/campaigns/pilot_expression_noise.campaign.json  (sidecar: spec + generated_ids)

The meta-runner is **idempotent**: re-running reuses existing perturbations
(same ``(operator, params, seed)`` hash → same ``dataset_id``). Pass
``--regenerate`` to overwrite in place. ``--dry-run`` reports what would be
generated without writing anything.

-----------------
Running the campaign
-----------------

Locally
=======

::

    uv run runscripts/workflow.py \
        --config configs/campaigns/pilot_expression_noise.json

For a fast validation run that exercises all processes without actually
running parca / sims, use Nextflow's stub-mode against the ``--build-only``
artifacts::

    uv run runscripts/workflow.py --config <...> --build-only
    cd out/<exp>/nextflow
    nextflow run main.nf -profile standard -stub-run -c nextflow.config

On AWS Batch via atlantis
=========================

vEcoli simulators run on AWS Batch through sms-api's ``atlantis`` CLI
(see ``sms-api/ATLANTIS_TUTORIAL.md``). The campaign-specific additions::

    # 1. Generate the campaign artifacts (commit + push to a branch on the
    #    sms-api accepted list)
    uv run runscripts/run_sensitivity_campaign.py --spec <...>
    git add configs/campaigns/ ecoli/analysis/multivariant/sensitivity_overview.py
    git commit && git push origin <branch>

    # 2. Build a simulator from that branch
    uv run atlantis simulator latest \
        --repo-url https://github.com/CovertLab/vEcoli \
        --branch <branch>

    # 3. Sync data sources + submit the workflow
    uv run atlantis simulation run <exp> <SIMULATOR_ID> \
        --config-filename configs/campaigns/<name>.json \
        --sources ../ecoli-sources \
        --sources ../ecoli-sources-vegas \
        --run-parca --poll

The ``--sources`` flag (atlantis-side) does two things:

1. Runs ``aws s3 sync`` for each local directory to
   ``s3://{STORAGE_S3_BUCKET}/sources/<basename>/``.
2. Forwards the resulting URIs as ``ECOLI_SOURCES`` and
   ``ECOLI_SOURCES_OVERLAYS`` environment variables on the K8s Job that
   launches Nextflow — no manual config edits needed.

The first ``--sources`` directory backs ``ECOLI_SOURCES``; subsequent ones
become ``;``-joined overlay manifest URIs.

Build-time bake (optional)
==========================

Alternatively, primary ecoli-sources data can be baked into the simulator
image at build time (saves per-task S3 GETs, pins the data version to the
image). Controlled by build-args in ``runscripts/container/Dockerfile``::

    ARG ECOLI_SOURCES_REPO_URL=""
    ARG ECOLI_SOURCES_REF="main"

``build-and-push-ecr.sh`` reads the matching env vars and forwards them as
``--build-arg``. sms-api's ``_submit_batch_build`` sets these from the
``ecoli_sources_repo_url`` / ``ecoli_sources_ref`` settings on the Batch
build job. When set, the Dockerfile clones at build time and defaults
``ENV ECOLI_SOURCES=/ecoli-sources``; any runtime override (like the
atlantis-injected S3 URI) still takes precedence.

-------
Outputs
-------

Per-variant (one set per perturbed dataset)::

    s3://{bucket}/{experiment_id}/
      parca_{0..N-1}/kb/                  simData.cPickle + validation data
      variant_sim_data/{0..N-1}.cPickle + metadata.json
      history/                            Parquet (per sim)
      daughter_states/
      analyses/variant={0..N-1}/
        plots/analysis=mass_fraction_summary/
        lineage_seed=*/plots/analysis={ribosome_*,cd1_higher_order_properties}/

Cross-variant (one set for the whole campaign)::

      analyses/plots/analysis=sensitivity_overview/
        sensitivity_overview.html         4-panel scatter (axis vs. metric)
        sensitivity_overview.tsv          per-variant metric table

The sensitivity_overview TSV columns are the headline campaign deliverable:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Column
     - Meaning
   * - ``variant``
     - Global variant index
   * - ``axis_value``
     - The operator-param value for the x-axis (e.g. ``sigma``);
       0 for the source baseline
   * - ``seed``
     - RNG seed used by the operator (nullable)
   * - ``dataset_id``
     - ``source_id__operator__hash`` for the perturbed dataset
   * - ``operator``
     - Operator name, or ``"baseline"`` for the unperturbed source
   * - ``mean_doubling_time_min``
     - Mean over all sims in the variant
   * - ``n_sims``
     - Number of sims that produced history in this variant
   * - ``final_dry_mass_fg``
     - Mean dry mass at the last timestep of the final generation
   * - ``mass_drift_per_gen_fg``
     - Linear-regression slope of per-generation mean mass (headline
       "unhealthy sim" signal)
   * - ``frac_max_gen``
     - Fraction of the max observed generation reached. Variants whose
       parca failed are absent from the parquet data and won't appear.

For a post-hoc campaign summary (parca status, durations, failure reasons),
run the standalone report::

    uv run wholecell/io/multiparca_analysis.py \
        --out_dir out/<experiment_id> -o out/<exp>/reports/

----------------------
Adding new operators
----------------------

Operators live in ``ecoli-sources/processing/perturbations.py``:

1. Add the function. It takes a TPM DataFrame (``gene_id``, ``tpm_mean``,
   optional ``tpm_std``) and returns the same shape. Use the
   ``np.random.default_rng(seed)`` convention for stochastic ops.
2. Register it in ``UNARY_OPERATORS`` or ``BINARY_OPERATORS`` at the
   bottom of the module.
3. Reference it by name in a campaign spec's ``"operator"`` field.

The driver and manifest plumbing need no changes.

-----------------
Adding new analyses
-----------------

Analyses live under ``ecoli/analysis/<level>/``; see existing modules for
the ``plot(...)`` contract. For cross-variant comparisons use
``multivariant/``; reference it from a campaign spec's
``analysis_options.multivariant`` dict with any params the analysis
expects.

------------
Design refs
------------

* ``.claude/plans/dataset-sensitivity-exploration.md`` — scope, physiological
  groupings, ~10k-sim budget plan, follow-ups
* ``.claude/plans/rnaseq-dataset-failure-analysis.md`` — failure-mode taxonomy
  the campaign design responds to
* ``.claude/plans/multi-parca-workflow.md`` — earlier multi-parca design
  (pre-master content-addressing; historical context)
* ``doc/model_fragility_map.md`` — per-subsystem implicit dataset dependencies
  and the failure signatures to look for
