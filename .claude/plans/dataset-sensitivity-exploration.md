# Plan: Dataset Sensitivity Exploration & Model–Data Interaction Study

## Purpose

Use the multi-parca workflow as a probe of the vEcoli model's response surface.
Rather than reproducing known conditions, systematically vary the *input* (starting
with RNAseq datasets) to map (a) where the model breaks, (b) why, and (c) which
parts of the model design are implicitly fine-tuned to the reference dataset vs.
genuinely robust.

Three observable regimes:
1. **Parca failure** — solver non-convergence (kcat, P-solve). Already partially
   understood (see `rnaseq-dataset-failure-analysis.md`).
2. **Sim failure** — parca succeeds, but a count goes to zero or a process blows up.
3. **Unhealthy sim** — no visible error, but growth degrades, especially across
   generations. This is the hardest regime and the most informative.

The sibling `omics-sources/` repo is the right home for dataset management;
vEcoli consumes via manifest.

---

## Part 1 — Analysis of the Model (what makes it fragile vs. robust)

Before designing perturbations, enumerate the parts of the model that have an
implicit dependence on the reference dataset. These are the hypotheses the
sensitivity study tests.

### 1.1 Known / suspected brittleness points

| Subsystem | Why suspicious | What to check |
|---|---|---|
| **ppGpp regulation** (`rna_synth_prob_factors_from_ppgpp`, condition-specific fits) | Parameters fit on 3 conditions (basal / +AA / anaerobic); extrapolation untested | Run off-reference datasets and log ppGpp and RNA synth-prob drift across gens |
| **Amino-acid kcat fitting** | Uses enzyme TPM as catalytic-capacity proxy; non-linear solve; already failing on CYS/TRP/ILE | Map per-pathway TPM statistics to failure mode (existing plan) |
| **P-solve (`fitPromoterBoundProbability`)** | LP with hard `pdiff` constraint, sensitive to TF-regulon expression structure | Diagnostic from `rnaseq-dataset-failure-analysis.md` Step 1 |
| **Expression adjustments** (`flat/adjustments/rna_expression_adjustments.tsv`) | ~dozen genes multiplied by 10× to rescue reference; fit choice, not physiology | Compare whether adjustments remain sufficient / excessive for alternate datasets |
| **Partitioning architecture** (Vivarium partition/pull) | Discrete count partitioning between processes; flagged for replacement | Log per-process request vs. allocation; identify processes starved on perturbed inputs |
| **Polycistronic TU → cistron fraction split** | Convention matters when one gene in a TU is rescaled | Audit during perturbation generation |
| **Fill-missing-genes-from-ref** | Hides the dataset's true coverage; stealth bias | Track fill rate per dataset and per-subsystem |
| **Translation efficiencies / ribosome allocation** | Fit once, assumed generalizable | Stress by uneven ribosomal gene expression |
| **Metabolism (FBA) initialization** | Targets from basal condition | Mass fraction drift across gens |

### 1.2 Outputs of the model analysis step

Produce `docs/model_fragility_map.md`: for each subsystem above, one paragraph on
(a) what it depends on in the input, (b) the failure mode expected, (c) the
observable that reveals it. This is the dashboard against which sensitivity runs
are read.

The `gene-metadata-extraction.md` plan's `gene_metadata.tsv` is the join table
that lets us tag genes by subsystem. Ship that first.

---

## Part 2 — Dataset Management (get data out of the repo)

### 2.1 Principles

- vEcoli should contain **no TPM TSVs**. Only the manifest path is a config knob.
- Datasets live in `omics-sources/` (already set up; schemas + processing exist).
- Derived / synthetic perturbation datasets live in the same place, versioned.
- vEcoli consumes via an opaque dataset-id → file-path resolver.

### 2.2 DVC recommendation

DVC fits cleanly:

- Dataset TSVs (tiny, ~10 KB each) and manifest are **source-controlled** in
  `omics-sources/data/`; no DVC needed for these small reference files.
- **Perturbation sets** (potentially thousands of TSVs, GB-scale) go under DVC:
  `omics-sources/perturbations/{family}/{params_hash}.tsv` with
  `perturbations.dvc` pointing to a GCS or local shared cache.
- `dvc.yaml` stages: `generate_perturbations` (input: seed datasets + perturbation
  spec → output: TSV set + extended manifest), `validate` (pandera schema).

### 2.3 Migration steps

1. Move `reconstruction/ecoli/experimental_data/rnaseq/*.tsv` (6.7 MB, 22 files)
   into `omics-sources/data/` (already largely mirrored there).
2. Delete the copies from vEcoli; add `reconstruction/ecoli/experimental_data/rnaseq/`
   to `.gitignore`.
3. Change `parca_options.rnaseq_manifest_path` to default to an env-var /
   config-resolved path (e.g. `$OMICS_SOURCES/data/manifest.tsv`).
4. Add a one-line `omics-sources` install doc pointing to the sibling-repo
   layout or to `dvc pull`.
5. Add `dvc init` in `omics-sources/` and a remote pointing at a shared bucket;
   small curated datasets stay git-tracked, perturbations go under DVC.

### 2.4 Cloud / HPC story

Nextflow already stages paths. For gcloud runs the manifest + TSVs must be
accessible — either bake into container or resolve via gcsfuse. DVC supports
GCS remotes natively, so the same bucket works for local + cloud.

---

## Part 2.5 — Data Grouping and Exploration Order

Model inputs span many tables with very different scales, units, and coherence
constraints. A flat concatenation into a single giant vector would (a) let
one high-variance feature swamp everything else, (b) violate hard physiological
constraints (mass fractions must sum to 1; kinetic constants must be positive),
and (c) make failure attribution ambiguous when a run breaks. Instead, group
inputs by physiological channel and let perturbations act within each group
(with interaction studies coming from factorial combinations across groups).

### 2.5.1 Groups

| # | Group | Tables | Why explore here |
|---|---|---|---|
| 1 | **Expression profile** | RNAseq TPM + `translation_efficiency` + `adjustments/rna_expression_adjustments` | Starting point. Direct handle on steady-state protein levels. |
| 2 | **Physiology / mass balance** | `growth_rate_dependent_parameters` + `dry_mass_composition` + `mass_parameters` + `metabolite_concentrations` + `relative_metabolite_concentrations` | Couples to ppGpp (fragility-map §1) and FBA init (§9). |
| 3 | **Metabolic kinetics** | `metabolism_kinetics` + `amino_acid_uptake_rates` + `amino_acid_export_kms` + `equilibrium_reaction_rates` | Drives the amino-acid kcat failure mode (§2). |
| 4 | **Turnover** | `rna_half_lives` + `protein_half_lives_{measured,pulsed_silac,n_end_rule}` + `adjustments/{rna,protein}_deg_rates_adjustments` | Couples to steady-state expression; orthogonal axis to group 1. |
| 5 | **Regulation magnitudes** | `fold_changes` + `ppgpp_regulation` + `ppgpp_fc` | Drives P-solve failure mode (§3). |
| 6 | **Structural (deferred)** | reactions (metabolic / complexation / equilibrium / trna-charging), transcription units, EcoCyc ontology (genes, rnas, proteins, metabolites) | Perturbing these changes the *model*, not the *data*. Out of scope for sensitivity. |

Groups 1-5 are data-sensitivity targets; group 6 is model-surgery territory.

### 2.5.2 Canonical object: `DatasetBundle`

Replace today's per-table config paths with a single `DatasetBundle` dict:

```python
DatasetBundle = {
    "expression":  {...},  # gene_id -> {tpm_mean, translation_efficiency, adjustment_factor}
    "physiology":  {...},  # doubling_time -> physiology params dict
    "kinetics":    {...},  # reaction_id -> kcat/km dict
    "turnover":    {...},  # id -> {rna_half_life, protein_half_life, adjust_factor}
    "regulation":  {...},  # (tf, target) -> {log2_fc, direction}
}
```

A sensitivity run specifies an operator tuple
`(op_expression, op_physiology, op_kinetics, op_turnover, op_regulation)`;
each operator is identity unless perturbing that group. The parca loads the
bundle and materializes it into the usual parca inputs.

Bundle reference sources come from `ecoli-sources/data/` (public bundle pieces)
with overlays from the private repo when available.

### 2.5.3 Exploration order

1. **Group 1 alone** (expression profile) — reuse existing RNAseq datasets as
   the reference bundle's group-1 slot; perturb with the operators in Part 3.
   This is the user's stated starting point; matches work already underway.
2. **Group 1 + 2** — factorial of expression × physiology. ppGpp lives at
   the coupling point, so this is where the "unhealthy sim" regime is most
   likely.
3. **Group 3 alone** — kinetics: direct test of the kcat failure mode with a
   cleaner attribution (no confounding expression changes).
4. **Group 1 + 4** — turnover perturbations alongside expression. Half-life
   and expression together determine steady-state count; isolating each
   reveals which parca solver branches are loading.
5. **Group 5** — regulation magnitudes alone; P-solve stress test.
6. **Pairwise interactions** — a subset of (i, j) pairs for i, j ∈ {1..5}
   beyond what steps 1-5 covered, using residual budget from Part 4.

This order front-loads the highest-leverage axis (expression, where we already
have datasets and the user's primary interest), then adds the axis where the
model's mechanistic coupling is strongest (physiology), then the well-defined
failure modes (kinetics, regulation).

### 2.5.4 Implications for perturbation operators

Each operator in Part 3 applies to a *single group*. Operator APIs are
group-specific because coherence constraints differ:

* Group 1 (expression) — operators on TPM vectors (log-normal noise, gene-set
  scaling, interpolation, quantile matching). See Part 3.1.
* Group 2 (physiology) — operators must preserve mass-fraction sum = 1 and
  monotone doubling-time ordering. Use Dirichlet noise for mass fractions;
  per-parameter log-normal noise for rates.
* Group 3 (kinetics) — log-normal perturbations around measured kcats / KMs
  with per-reaction independence or pathway-correlated structure.
* Group 4 (turnover) — log-normal perturbations on half-lives with a hard
  floor (0 is no-data marker, not a valid physiological value).
* Group 5 (regulation) — additive noise on log2 fold-changes; sign preservation
  is a choice to flag.

---

## Part 3 — Perturbation Operators

Build a library of operators that map a TPM vector → perturbed TPM vector.
Each operator is parameterized, deterministic given a seed, and produces a new
`dataset_id` with traceable provenance.

### 3.1 Operator catalog (start simple, add)

| Operator | Params | What it tests |
|---|---|---|
| `add_noise(sigma, seed)` | log-normal noise on TPM | Robustness to measurement noise |
| `scale_gene_set(gene_ids, factor)` | Multiply a named gene set (e.g. TCA cycle) | Does the model respond proportionately? Inverse of manual adjustments |
| `zero_gene(gene_id)` | Knockout expression | Identify load-bearing single genes |
| `interpolate(dataset_a, dataset_b, alpha)` | Convex mix of two datasets | Continuous path from working to failing |
| `quantile_match(target_dataset)` | Rank-preserving rescale to another dataset's distribution | Separates distribution-shape effects from identity effects |
| `shuffle_within_subsystem(subsystem)` | Permute TPM among genes in a class | Tests whether subsystem-level totals are what matters |
| `clip(p_low, p_high)` | Compress dynamic range | Tests P-solve sensitivity to outliers |
| `drop_and_fill(fraction)` | Synthesize coverage gaps | Isolates fill-from-ref bias |

Annabelle's operators (per user note — artifact not in tree) slot in here as
additional named operators once located.

### 3.2 Synthetic dataset storage

- `omics-sources/perturbations/<operator>/<seed_dataset>/<params_hash>.tsv`
- Extended manifest `perturbations_manifest.tsv` with columns from
  `RnaseqSamplesManifestSchema` plus `parent_dataset_id`, `operator`,
  `operator_params_json`, `seed`.
- Builder lives in `omics-sources/processing/perturbations.py` alongside
  existing `make_gene_exclusion_variant`.

---

## Part 4 — Experiment Design for 10,000 Sims

### 4.1 Budget allocation

Rough split (tunable):

- **2,000** — dense sweep around the reference (`vecoli_m9_glucose_minus_aas`)
  using `add_noise` at 4 noise levels × 50 seeds × 10 gens.
- **2,000** — interpolation paths between reference and each of the 21 other
  real datasets (≈100 sims per path, `alpha ∈ [0, 1]`, 5 seeds, 4 gens).
  This locates the break point on a continuous path.
- **1,500** — single-subsystem scaling: for each of {amino-acid biosynthesis,
  TCA, translation machinery, RNAP, TFs, ribosomes} scale by
  {0.1, 0.25, 0.5, 1, 2, 4, 10} × 5 seeds × 5 gens.
- **1,500** — single-gene knockouts across the adjustment-flagged gene set and
  a control set; isolates which adjustments are load-bearing.
- **1,000** — quantile-matched variants: rank-preserve successful datasets onto
  failing-dataset distributions; separates "what values" from "which genes".
- **1,000** — drop-and-fill sweep: 0–30% gene dropout at 10 levels × many seeds.
- **1,000** — reserved for targeted follow-ups based on early results.

### 4.2 Nextflow orchestration

The `multi-parca-workflow` already supports N parcas in one run. For 10k sims we
need to:

- **Chunk**: one `parca_variants` list per Nextflow run is fine up to ~200
  entries. Break the 10k design into ~50 workflow invocations by operator family
  and drive them from a meta-runner (`runscripts/run_sensitivity_campaign.py`)
  that emits one config per chunk and optionally launches them sequentially or on
  gcloud.
- **Metadata**: inject `parca_dataset_label`, `operator`, `operator_params`,
  `parent_dataset_id` into each variant's metadata (addresses the known
  limitation in `multi-parca-workflow.md`).
- **Fail isolation**: already handled — parca failures are ignored and the
  workflow continues. Record failure mode in the summary.
- **Trace collection**: rely on the Nextflow trace CSV; `multiparca_analysis.py`
  already consumes it.

### 4.3 Observable schema per run

One row per (parca_id, variant, seed, generation):

- Parca outcome: `status`, `error_class` (kcat / P-solve / other / ok), `duration`
- Sim outcome: `completed`, `crash_reason`, `final_gen_reached`
- Growth: `doubling_time`, `generation_time_std_across_gens`,
  `mass_drift_per_gen` (slope of dry mass over gens — the "unhealthy" signal)
- Composition: mean mass fractions (protein, RNA, DNA, small molecules)
- Starvation proxies: min count across critical species (ribosomes, RNAP, key
  aaRSs, tRNAs); which process requested-but-didn't-get allocations the most
  (partitioning diagnostic)
- ppGpp trajectory summary

Store as one Parquet per campaign chunk; concatenate for analysis.

---

## Part 5 — Analysis: Building the Response Surface

### 5.1 Primary questions

1. **Acceptance region**: which perturbations produce healthy sims? Characterize
   as a region in (operator, magnitude) space.
2. **Failure taxonomy**: cluster failures by error class + affected subsystem.
3. **Sensitivity ranking**: which genes / subsystems, when perturbed, move
   `doubling_time` or `mass_drift_per_gen` the most? Sobol / Morris indices on
   the subsystem-scaling sweep.
4. **Adjustment inversion**: for a failing dataset, what minimal scaling of the
   adjustment-flagged gene set (or its extension) restores a healthy sim? This
   is the user's "understand alternate inputs through the adjustments needed to
   make them work" — frame as an optimization over the adjustment vector.
5. **Predictive model**: train a simple regression (TPM features → doubling
   time) on the campaign; use it to (a) flag suspect datasets before running,
   (b) highlight which input coordinates dominate the prediction.

### 5.2 Reports

Extend `wholecell/io/multiparca_analysis.py` (already the campaign summary tool)
to consume a directory of chunks and emit:

- Campaign-level CSV, one row per run
- Failure taxonomy heatmap (operator × error class)
- Interpolation break-point plot per real-dataset target
- Sobol / Morris bar charts for subsystem-scaling sweep
- Acceptance-region projection plots (PCA over TPM + success/fail coloring)

---

## Part 6 — From Findings Back to the Model

Each finding should tag a subsystem from Part 1.1. Concretely:

- If P-solve fails on a structured subset of datasets, that's evidence the
  `pdiff` constraint or its parameterization is the culprit; candidate
  improvements: soft constraint, TF-specific threshold, or reformulation.
- If interpolation reveals a sharp cliff rather than a gradient, that indicates
  a non-smooth coupling (often a threshold inside the model) — locate and either
  smooth it or surface it as a config knob.
- If partitioning-starvation shows up as the dominant "unhealthy sim" mechanism,
  that corroborates the need to replace the current partition/pull architecture.
  Candidate replacement: *continuous* flux allocation (Vivarium's dynamic
  flux-balance-style topology) or a priority/quota scheme with explicit unmet
  demand reporting. Evaluate on the same campaign post-swap.
- ppGpp-regulated genes showing disproportionate drift across gens would
  motivate refitting the ppGpp factors on a broader condition set (or reducing
  the number of ppGpp-dependent couplings).

---

## Deliverables & Order

1. **Dataset migration + manifest resolver** (Part 2). Unblocks everything.
2. **Model fragility map doc** (Part 1.2). Can proceed in parallel.
3. **Gene metadata tables** (ship `gene-metadata-extraction.md`).
4. **Perturbation operators library + small synthetic set** (Part 3).
5. **Sensitivity campaign meta-runner** (Part 4).
6. **Pilot campaign (~500 sims)** before committing to 10k: validates pipeline
   + shrinks budget allocations based on observed variance.
7. **Full campaign + analysis reports** (Parts 4–5).
8. **Model-improvement write-up** grounded in campaign findings (Part 6).

## Open Questions

- Annabelle's perturbation toolkit — where does it live? Align operator API
  once located.
- Budget for gcloud vs. local HPC for 10k runs?
- Are we willing to move the existing curated RNAseq TSVs out of vEcoli *now*
  (breaking change for default configs) or stage the migration with a
  deprecation warning?
- DVC remote: existing bucket to reuse, or provision new?
