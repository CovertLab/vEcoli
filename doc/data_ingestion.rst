==========================
Data Ingestion (Bundles)
==========================

vEcoli's data ingestion layer reads experimental and reference inputs from
an external **data bundle** — a curated package of canonical *E. coli*
datasets shipped by `vivarium-collective/ecoli-sources <https://github.com/vivarium-collective/ecoli-sources>`_.
ParCa, the parameter calculator, resolves every flat-file input through
this bundle, so substituting a different dataset (different RNA-seq
condition, perturbed metabolite pool sizes, alternative kinetic
parameters, etc.) is done by pointing ParCa at a different bundle —
not by modifying files inside vEcoli.

This page describes the consumer side of that contract: how vEcoli
loads bundle data, what configuration controls the choice of bundle,
and how to author a custom bundle.

--------
Overview
--------

The system has two repositories with complementary roles:

============================  ==========================================================
Repository                    Role
============================  ==========================================================
``vEcoli`` (this repo)        **Consumer.** Declares which canonical inputs the model
                              needs; resolves them at ParCa time via ``SourceBundle``.
``ecoli-sources``             **Supplier.** Hosts the data files, the bundle manifest
                              that maps canonical keys to source paths, and the
                              Pandera schemas that validate each file's content.
============================  ==========================================================

A **bundle** is a TSV manifest of the form::

    canonical_key             source_path                                      description    schema_name
    rnaseq_basal_tpms         rnaseq_experimental/vecoli_m9_glucose_minus_aas.tsv  Basal TPMs   RnaseqTpmTableSchema
    rnaseq_experimental_tpms  rnaseq_experimental/vecoli_m9_glucose_minus_aas.tsv  Expt TPMs    RnaseqTpmTableSchema
    genes                     flat/genes.tsv                                       ParCa input  ...
    metabolic_reactions       flat/metabolic_reactions.tsv                         ParCa input  ...
    ...

Each row binds a **canonical key** (an addressable role in the model,
e.g. ``rnaseq_basal_tpms``, ``metabolic_reactions``, ``genes``) to a
specific file in the bundle's data tree. The set of required canonical
keys is enforced by ``ReferenceBundleSchema`` in ecoli-sources; any
loadable bundle must include all of them.

vEcoli ships with no per-file plumbing for these inputs. ``KnowledgeBaseEcoli``
asks the bundle for each canonical key it needs; the bundle resolves the
key to an absolute path on disk; the file is loaded and validated.

-----------------
Default behavior
-----------------

When ParCa is run with no bundle override, the default reference bundle
shipped with the installed ``ecoli-sources`` package is used. This
bundle reproduces the exact set of inputs used by the legacy
``reconstruction/ecoli/flat/`` directory tree, so the default behavior
is identical to the pre-migration model.

The default is resolved via:

.. code-block:: python

    from ecoli_sources import BUNDLE_PATH  # absolute path to reference_bundle.tsv

``ecoli-sources`` is pinned in this repo's ``pyproject.toml`` by commit
SHA; bumping the pin is a visible PR step.

------------------
ParCa configuration
------------------

To use a custom bundle, set ``bundle_manifest_path`` under
``parca_options`` in your config JSON:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Type
     - Description
   * - ``bundle_manifest_path``
     - string or null
     - Path to a bundle manifest TSV (canonical_key → source_path).
       If null, defaults to ``ecoli_sources.BUNDLE_PATH`` (the reference
       bundle shipped with the installed package).
   * - ``basal_expression_condition``
     - string
     - Modeled condition name for the baseline growth state.
       Default: ``"M9 Glucose minus AAs"``.
   * - ``rnaseq_fill_missing_genes_from_ref``
     - bool
     - If true, genes present in the basal RNA-seq table but missing
       from the experimental table are cross-filled. Default: true.

**Example:**

.. code-block:: json

    {
        "parca_options": {
            "cpus": 4,
            "outdir": "out/custom_bundle",
            "bundle_manifest_path": "/path/to/my_variant/reference_bundle.tsv",
            "basal_expression_condition": "M9 Glucose minus AAs"
        }
    }

The legacy ``rnaseq_manifest_path`` and ``rnaseq_basal_dataset_id``
config keys have been removed; the bundle's
``rnaseq_basal_tpms`` / ``rnaseq_experimental_tpms`` canonical keys
fully subsume what those flags used to address.

----------------
Validation
----------------

Bundles are validated at load time in three layers:

1. **Manifest schema** (``ReferenceBundleSchema``) — checks the TSV's
   columns and types, and asserts that every required canonical key is
   present.
2. **Path resolution** — every ``source_path`` must resolve to an
   existing file under the bundle's data root.
3. **Content schemas** — for rows with ``schema_name`` set, the
   referenced Pandera schema is applied to the file's contents.

Layers 1 and 2 are eager (run at ``SourceBundle.__init__``); layer 3
runs in ``ecoli-sources``' CI via ``scripts/validate_all.py`` and on
demand via ``scripts/validate_bundle.py`` against any bundle path.

A malformed or incomplete bundle fails at ParCa load time with an error
message naming the missing or invalid key, rather than at the first
deep ``bundle.get(...)`` call inside fitting.

--------------
Python API
--------------

The resolver is a small class in :py:mod:`wholecell.io.sources`:

.. code-block:: python

    from wholecell.io.sources import SourceBundle

    bundle = SourceBundle()  # defaults to ecoli_sources.BUNDLE_PATH
    # or: SourceBundle(manifest_path="/path/to/my_variant/reference_bundle.tsv")

    genes_path = bundle.get("genes")            # absolute Path
    has_rnaseq = bundle.has("rnaseq_basal_tpms")

The bundle is loaded once at the start of ParCa and queried many times.
``KnowledgeBaseEcoli`` excludes the live bundle handle from its pickled
state (``__getstate__``) so that ``rawData.cPickle`` does not bake
absolute machine paths into its contents.

--------------------------
Currently supported inputs
--------------------------

The default bundle ships 135 canonical keys covering the full ParCa
input surface. By data category:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Category
     - Notes
   * - Transcriptomics (RNA-seq)
     - ``rnaseq_basal_tpms``, ``rnaseq_experimental_tpms``. Per-condition
       TPM tables; cross-fill of missing genes from the basal table is
       optional.
   * - Translation efficiencies
     - Translation-rate parameters per gene (used as the ingestion path
       for protein abundance via the protein/RNA ratio).
   * - Metabolite pool sizes
     - ``metabolite_concentrations`` (absolute, multi-source consensus)
       and ``relative_metabolite_concentrations`` (per-condition fold
       changes); plus ``linked_metabolites`` for paired-ratio
       constraints.
   * - Amino-acid uptake fluxes
     - ``amino_acid_uptake_rates`` with LB/UB; literature-sourced.
   * - Secretion bounds
     - ``secretions`` — fixed-bound style for ATP/water/CO₂/etc.
   * - Media composition
     - ``condition__media__*`` — sets the *availability* side of
       exchange (concentration in mmol/L per molecule).
   * - Growth-rate-dependent parameters
     - ``growth_rate_dependent_parameters`` — RNAP/ribosome elongation,
       mass fractions, ppGpp concentration, etc., keyed by doubling
       time.
   * - Reaction networks
     - ``metabolic_reactions``, ``equilibrium_reactions``,
       ``complexation_reactions``, plus ``*_added`` / ``*_modified`` /
       ``*_removed`` deltas.
   * - Regulation
     - ``fold_changes``, ``ppgpp_regulation``, ``protein_half_lives_*``,
       ``rna_half_lives``, etc.

Macroscopic exchange fluxes (e.g. glucose qS, O₂ qO₂, organic-acid
qP) and intracellular flux distributions (13C MFA-style) do not yet
have a native canonical-key slot; introducing one is a future-work
item tracked on the ``ecoli-sources`` side.

-----------------
Authoring data
-----------------

Adding a new dataset, schema, or canonical key is done in
``ecoli-sources``, not vEcoli. See
`ecoli-sources/BUNDLES.md <https://github.com/vivarium-collective/ecoli-sources/blob/main/BUNDLES.md>`_
for the bundle authoring guide.

To use new data once it lives in ecoli-sources, either:

* **Bump the pinned commit** in ``pyproject.toml`` to a new
  ``ecoli-sources`` revision that includes the data, and rerun ParCa
  with the default bundle; or
* Create a **variant bundle** — a separate ``reference_bundle.tsv``
  with rows pointing at your custom files — and pass its path via
  ``--bundle-manifest-path``. The required-canonical-keys contract
  must still be satisfied; you can override individual rows or add
  new ones.

----------
References
----------

- Resolver module: :py:mod:`wholecell.io.sources`
- ParCa configuration overview: :doc:`workflows`
- Source-of-truth repo: `vivarium-collective/ecoli-sources <https://github.com/vivarium-collective/ecoli-sources>`_
- Bundle schema (consumer-side validation): ``ReferenceBundleSchema``
  in ``ecoli-sources/schemas/reference_bundle.py``
