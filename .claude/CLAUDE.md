# vEcoli — Agent Orientation

Vivarium *E. coli* (vEcoli) is a whole-cell model of *E. coli* built on the
[Vivarium framework](https://github.com/vivarium-collective/vivarium-core). It
ports the Covert Lab's wcEcoli model to a modular, JSON-configured, Parquet-output
architecture runnable locally or on Google Cloud / HPC clusters via Nextflow.

---

## Key Concepts

### ParCa (Parameter Calculator)
Entry point: `runscripts/parca.py` → calls `reconstruction/ecoli/fit_sim_data_1.py`

ParCa ingests raw experimental data (`reconstruction/ecoli/flat/`) and RNAseq
expression data to fit model parameters. Outputs `simData.cPickle` and companion
pickles into a `kb/` directory. ParCa is expensive (minutes–hours); everything
downstream reads its output.

### Variants
Directory: `ecoli/variants/`
Script: `runscripts/create_variants.py`

Variants deep-copy `simData.cPickle` from ParCa and apply Python functions to
perturb it. They do **not** re-run ParCa. Each variant combination is saved as
`{i}.cPickle`; variant 0 is the unmodified baseline. Variant config lives in the
JSON config under `"variants": { "<variant_name>": { <param_spec> } }`.

### Simulations
Entry point: `ecoli/experiments/ecoli_master_sim.py`
Processes: `ecoli/processes/`

Simulations consume a variant `simData.cPickle` and emit Parquet output. Runs are
keyed by `(variant_index, seed, generation)`. Each cell division spawns daughter
sims. The `condition` variant is the standard way to run multiple growth conditions
from a single ParCa.

### Nextflow Workflow
Entry point: `runscripts/workflow.py` (generates Nextflow code) → invokes
`nextflow run` with `runscripts/nextflow/template.nf` (plus `sim.nf`,
`analysis.nf`).

Pipeline stages:
1. `runParca` — runs ParCa (skipped if `sim_data_path` is set in config)
2. `createVariants` — generates variant pickles from ParCa `kb/`
3. `simGen0` / `sim` — cell simulations across seeds and generations
4. `analysisSingle`, `analysisMultiGeneration`, `analysisMultiSeed`,
   `analysisMultiVariant` — analysis at increasing aggregation levels

### Config System
All configuration is JSON. `configs/default.json` is the baseline. User configs
specify only overrides. Key top-level sections:

| Key | Purpose |
|---|---|
| `parca_options` | Options passed to `parca.py` (including `rnaseq_basal_dataset_id`) |
| `variants` | Variant name + parameter spec |
| `n_init_sims`, `generations`, `lineage_seed` | Simulation topology |
| `analysis_options` | Which analysis scripts run and at which level |
| `sim_data_path` | Skip ParCa; use this pre-computed `simData.cPickle` |
| `gcloud` | Google Cloud Batch config (null for local) |

Configs support `inherit_from` for layered overrides.

---

## Directory Map

```
runscripts/
  workflow.py          # generates Nextflow .nf code and launches workflow
  parca.py             # ParCa entry point
  create_variants.py   # loads simData, applies variant functions, saves pickles
  nextflow/
    template.nf        # runParca, createVariants, hqWorker processes + workflow block
    sim.nf             # simGen0, sim processes
    analysis.nf        # all analysis aggregation processes
    config.template    # Nextflow profiles (local, gcloud, sherlock, sherlock_hq)

reconstruction/ecoli/
  fit_sim_data_1.py    # ParCa fitting pipeline
  simulation_data.py   # SimulationDataEcoli — the simData object
  flat/                # raw experimental data (TSVs)
  experimental_data/rnaseq/
    manifest.tsv       # available RNAseq datasets (ref_0001, gbw_0001_v2, ...)
    *.tsv              # TPM expression tables

ecoli/
  processes/           # individual simulation processes
  variants/            # variant functions (condition.py, etc.)
  experiments/
    ecoli_master_sim.py  # SimConfig + simulation runner

configs/
  default.json         # baseline config — read this before touching any config
  templates/           # environment-specific config templates
wholecell/io/
  multiparca_analysis.py  # standalone summary report for multi-parca runs (see plan)
  compare_sims.py         # Marimo notebook for manual two-experiment comparison
```

---

## RNAseq Data Ingestion

RNAseq datasets are listed in `reconstruction/ecoli/experimental_data/rnaseq/manifest.tsv`.

Available datasets:
- `ref_0001` — legacy reference, M9 Glucose minus AAs
- `ref_0002` — legacy reference, M9 Glucose plus AAs
- `gbw_0001` — Ginkgo Bioworks MG1655, Modified M9 Glucose
- `gbw_0001_v2` — same but with ssrA removed and renormalized
- `gbw_0002` — Ginkgo Bioworks MG1655, EZ Rich media

Config knobs (under `parca_options`):
- `rnaseq_manifest_path` — path to manifest TSV (null = use legacy flat tables)
- `rnaseq_basal_dataset_id` — which dataset ID to use as the basal condition
- `basal_expression_condition` — condition label within the dataset
- `rnaseq_fill_missing_genes_from_ref` — fill gaps from reference data

---

## Running Things

```bash
# Install dependencies
uv sync

# Run ParCa only
uv run runscripts/parca.py --config configs/my_config.json -o out/

# Run full workflow (ParCa + sims + analysis) via Nextflow
uv run runscripts/workflow.py --config configs/my_config.json

# Run a single simulation directly (no Nextflow)
uv run ecoli/experiments/ecoli_master_sim.py --config configs/my_config.json
```

---

## Plans

- **Multi-ParCa workflow** (`.claude/plans/multi-parca-workflow.md`): support for
  running multiple ParCa instances with different RNAseq datasets in a single
  Nextflow workflow. **Implemented and end-to-end validated** on branch
  `multi-parca-workflow`. Key files: `runscripts/workflow.py`,
  `runscripts/nextflow/template.nf`, `runscripts/create_variants.py`.
  Example config: `configs/test_multi_parca.json`.

- **Multi-parca analysis report** (`.claude/plans/experiment-summary-report.md`):
  standalone script (`wholecell/io/multiparca_analysis.py`) that produces a summary
  CSV + plots after a multi-parca run. **Implemented and validated** against
  `out/test_multi_parca_20260323-171234`. Companion analyses enabled in
  `configs/test_multi_parca.json`: `cd1_higher_order_properties` (multiseed),
  `mass_fraction_summary` (single), `ribosome_components/production/usage`
  (multigeneration). Next step: increase `generations` and `n_init_sims` for richer
  validation.
