# Plan: RNAseq Dataset Failure Analysis

## Goal

Understand what discriminates the 6 successful parca datasets from the 13 failing ones
in `configs/multiparca_rnaseq_datasets.json`. Two distinct failure modes require
separate analyses.

---

## Context

From runs `multiparca_rnaseq_datasets_20260324-135752` and `multiparca_rnaseq_datasets_20260325-070112`.

**Successful:**
- `vecoli_m9_glucose_minus_aas` (reference)
- `gbw_bermuda_ctrl`
- `precise_RPO:wt`
- `precise_SNPv1:D-Glucose_NH4Cl`
- `precise_SNPv2:Glucose_NH4Cl`
- `precise_control:wt_glc`

**Failed — kcat (amino acid kinetics):**
- `gbw_vegas_wt_m9glc_{3h,34h,5h}` → CYS
- `precise_abx_media:m9_ctrl` → TRP
- `precise_ytf5:wt_glc` → ILE

**Failed — `fitPromoterBoundProbability` P-solve:**
- `precise_eep:BOP27`, `precise_ica:wt_glc`, `precise_oxyR:wt_glc`,
  `precise_minspan:wt_glc`, `precise_misc:wt_no_te`, `precise_ytf:wt_glc`,
  `precise_ytf2:wt_glc`, `precise_ytf3:wt_glc2`

---

## Failure Mode 1: kcat (CYS / TRP / ILE)

### Mechanism

Parca fits forward/reverse kcats for metabolic enzymes using enzyme expression
levels as a proxy for catalytic capacity. If a dataset has anomalously high or
low expression of the relevant biosynthesis genes, the optimizer can't find
physically valid kcats. Error: `ValueError: Could not find positive forward and
reverse kcat for {AA}[c]`.

### Analysis

Purely data-side; no parca changes needed. All raw data in
`reconstruction/ecoli/experimental_data/rnaseq/*.tsv` (column: `gene_id`, `tpm_mean`).

Gene lists extracted from `simData.cPickle` via `aa_synthesis_pathways` →
`complexation.get_monomers` → `cistron_to_monomer_mapping`:

**CYS[c]** — forward: `CYSSYNMULTI-CPLX`, `CPLX0-237`; reverse: `G7622-MONOMER`, `TRYPTOPHAN-CPLX`, `ACSERLYB-CPLX`
| gene_id  | symbol |
|----------|--------|
| EG10187  | cysE   |
| EG10192  | cysK   |
| EG10193  | cysM   |
| EG11005  | tnaA   |
| G7622    | cyuA   |

**TRP[c]** — forward: `ANTHRANSYN-CPLX`; reverse: `TRYPTOPHAN-CPLX`
| gene_id  | symbol |
|----------|--------|
| EG11005  | tnaA   |
| EG11027  | trpD   |
| EG11028  | trpE   |

**ILE[c]** — forward: `THREDEHYDSYN-CPLX`; reverse: `BRANCHED-CHAINAMINOTRANSFER-CPLX`
| gene_id  | symbol |
|----------|--------|
| EG10493  | ilvA   |
| EG10497  | ilvE   |

Note: `tnaA` (tryptophanase) appears in both CYS and TRP pathways as a reverse enzyme.

For each dataset TSV, extract `tpm_mean` for these gene_ids and plot distributions
across all datasets, with successful/failed color-coded.

**Expected finding:** Failing datasets have outlier TPM for the relevant pathway —
either near-zero (implying implausible low flux) or very high.

---

## Failure Mode 2: `fitPromoterBoundProbability` P-solve infeasibility

### Mechanism

The P-solve fits TF promoter-binding probabilities `P ∈ [0,1]` such that predicted
RNA synthesis rates `H @ P` match RNAseq-derived targets `k`. A hard constraint
(`pdiff @ P >= PROMOTER_PDIFF_THRESHOLD`) requires a minimum separation between
each TF's active and inactive binding probability. Infeasibility means the expression
targets can't be satisfied while maintaining that separation for at least one TF.

Note: the R-solve (fitting transcription factor influence parameters) succeeds for
these datasets — the failure is specifically in the subsequent P-solve.

### Step 1 — Identify the bottleneck TF(s)

Add a diagnostic to `fitPromoterBoundProbability` in `fit_sim_data_1.py` that, on
solver infeasibility, logs which pdiff constraints are most violated. Options:

- Run the problem without the pdiff constraint and compute `pdiff @ P_unconstrained`
  to see which TFs fall below threshold.
- Or inspect constraint dual values / residuals from the failed solve.

Run this diagnostic on one representative failing dataset (e.g. `precise_ica:wt_glc`)
to identify the specific TF(s). Likely the same TF(s) across all 8 failures.

### Step 2 — TF-regulated gene expression comparison

Once the bottleneck TF(s) are known:

1. For each TF, identify the genes it regulates (from `sim_data.relation.rna_id_to_regulating_tfs`
   or the flat files).
2. Extract TPM for those genes across all datasets.
3. Compare distributions between successful and failing datasets — expect more extreme
   or compressed dynamic range in failing ones.

Also worth doing genome-wide: compare expression of TF-regulated vs non-TF-regulated
genes across all datasets to see if the effect is TF-specific or global.

### Step 3 — Fill rate analysis

Each dataset has genes with missing TPM values that are filled from the reference
(`rnaseq_fill_missing_genes_from_ref`). The warning logged during parca reports the
count (e.g. "403 genes were missing..."). If fill rate is high, or if the filled
genes are disproportionately TF-regulated, the filled values may create
active/inactive inconsistencies that make the pdiff constraint unsatisfiable.

1. For each dataset TSV, count missing genes (zero or absent TPM).
2. Cross-reference against TF-regulated gene list from Step 2.
3. Compare fill rate and TF-regulated fill rate between successful and failing datasets.

### Step 4 — Global expression distance from reference

Quick sanity check: PCA or pairwise Pearson correlation of full TPM profiles across
all datasets. If failing datasets are more distant from the reference, that
contextualizes the problem. Notable: `precise_oxyR:wt_glc` is labeled WT but fails,
so distance from reference is probably not the whole story.

---

## Recommended Order

1. **kcat analysis** (Step 1 of Failure Mode 1) — fast, purely data-side, likely
   confirms the outlier-expression hypothesis in one plot.
2. **Parca diagnostic** (Failure Mode 2, Step 1) — small code change, immediately
   identifies the bottleneck TF(s) and makes Step 2 targeted.
3. **TF-regulated gene expression** (Failure Mode 2, Step 2) — once TFs are known.
4. **Fill rate analysis** (Failure Mode 2, Step 3) — may explain residual cases.
5. **Global PCA** (Failure Mode 2, Step 4) — contextual, lower priority.

---

## Data Sources

- Dataset TSVs: `reconstruction/ecoli/experimental_data/rnaseq/*.tsv`
- Manifest: `reconstruction/ecoli/experimental_data/rnaseq/manifest.tsv`
- Gene/pathway info: `reconstruction/ecoli/flat/`
- TF→gene mappings: `sim_data.relation.rna_id_to_regulating_tfs` (or flat files)
- Parca fitting code: `reconstruction/ecoli/fit_sim_data_1.py:fitPromoterBoundProbability`
