---
name: multi-parca project status
description: Current state of multi-parca workflow and analysis report work on branch multi-parca-workflow
type: project
---

**Why:** Compare vEcoli model behavior across different RNAseq datasets in a single
Nextflow workflow run, with automated summary reporting.

---

## Multi-ParCa Nextflow Workflow

**How it works:**

Two-level indexing: `global_variant = parca_idx * pickles_per_parca + variant_idx`.
This keeps all `.cPickle` filenames and simulation output directories non-colliding
while the existing `analysisMultiVariant` layer naturally compares across datasets.

Key implementation details:
- `configs/default.json` has a new `"parca_variants": []` top-level key. Each entry
  is a dict of `parca_options` overrides merged on top of baseline `parca_options`.
  Empty list → single-parca run (fully backward-compatible).
- `runscripts/workflow.py`: `generate_code()` emits a `Channel.of([0, file(...)], [1, file(...)], ...)`
  feeding a single `runParca` process (Nextflow fans out naturally). Per-parca config
  files (`parca_config_{i}.json`) are written to `local_outdir` and copied to `outdir`
  before Nextflow runs. `_count_pickles_per_parca()` (via importlib) computes
  non-overlapping offsets.
- `runscripts/create_variants.py`: new `--offset INT` arg (default 0) shifts baseline
  and variant pickle filenames so each parca's output occupies a non-overlapping range.
- `runscripts/nextflow/template.nf`:
  - `runParca` takes `tuple val(parca_id), path(config)` and publishes to `parca_{parca_id}/`
  - `createVariants` takes an offset arg and outputs `metadata_{parca_id}.json`
  - new `mergeVariantMetadata` process merges all per-parca metadata files into
    the single `metadata.json` that downstream analysis expects
  - all ParCa/variant setup is in the generated `RUN_PARCA` block (not hardcoded)
- Analysis uses parca_0's `kb/` (validationData); per-parca kb carry-through is a
  known future improvement.

---

## Analysis Report (`wholecell/io/multiparca_analysis.py`)

**What it produces:**
- `parca_summary.csv` — one row per parca: dataset_id, status, duration, per-generation
  sim counts, parca error, sim errors
- `parca_status.png` — bar chart of parca durations colored by status
- `cell_distributions.png` — 2×2 panel (violin + strip), all generations
- `cell_distributions_gen3plus.png` — same but filtered to generation ≥ 3 (steady-state)

**How it reads data:**
- Parca status/duration/workdir from Nextflow trace CSV (`trace--{expId}--*.csv` in repo root)
- Dataset labels from `nextflow/parca_config_{i}.json` → `parca_options.rnaseq_basal_dataset_id`
- Variant→parca mapping from `nextflow/workflow_config.json` + `_count_pickles_per_parca()`
- Per-cell doubling time and protein mass from Parquet history via DuckDB
- Cell mass/volume optionally from `analyses/variant={v}/plots/higher_order_properties.tsv`
  (written by `cd1_higher_order_properties` multiseed analysis; graceful fallback if absent)

**Plot layout:** 2×2 fixed grid (Doubling Time / Protein mass / Cell mass / Cell volume),
x-tick labels rotated 50°, ylim bottom=0. Missing metrics → panel hidden.

---

## Known Parca Failure Modes (observed on PRECISE/Ginkgo datasets)

From local test run `out/multiparca_rnaseq_datasets_20260324-135752`:

| Dataset | Failure |
|---|---|
| `gbw_vegas_wt_m9glc_34h` | `ValueError`: kcat not found for CYS (kinetics fitting) |
| `precise_ica/oxyR/minspan/ytf:wt_glc` | `RuntimeError`: Solver infeasible in `fitPromoterBoundProbability` (P-solve, not R-solve) — active/inactive TF binding probability separation constraint can't be satisfied given these datasets' expression targets |
| `precise_ytf5:wt_glc` | `ValueError`: kcat not found for ILE (kinetics fitting) |
| *(previously all precise)* | `IndexError` in `_apply_rnaseq_correction` — fixed: guard `if corrected_indexes:` before `np.array(corrected_indexes)` index assignment (empty list → float64 array) |

Datasets that succeeded locally: `vecoli_m9_glucose_minus_aas`, `vecoli_m9_glucose_plus_aas`,
`gbw_bermuda_ctrl`, `precise_control:wt_glc`, `precise_ytf3/ytf2` (varies by run).

---

## Production Config & Next Steps

`configs/multiparca_rnaseq_datasets.json` — 10 PRECISE/Ginkgo datasets, 6 generations,
3 seeds. Too large for local (360 sims); needs Sherlock or GCloud block added before
production run. Use `configs/cloud.json` or `configs/test_sherlock.json` as reference.

The `fitPromoterBoundProbability` infeasibility for 4+ PRECISE datasets is the main
open scientific question before a clean production run is possible.
