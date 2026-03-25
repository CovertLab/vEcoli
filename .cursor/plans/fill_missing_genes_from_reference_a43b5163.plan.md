---
name: Fill missing genes from reference
overview: When using the new RNA-seq ingestion path, genes missing from the experimental dataset can be filled from the basal-expression-condition column of the reference table (rnaseq_rsem_tpm_mean.tsv), with a one-time warning. This behavior is controlled by a config option (default True).
todos: []
isProject: false
---

# Fill missing genes from reference RNA-seq when using new ingestion

## Current behavior

In [reconstruction/ecoli/dataclasses/process/transcription.py](reconstruction/ecoli/dataclasses/process/transcription.py), when `sim_data.rnaseq_manifest_path` is set, `seq_data` is built only from the ingested TPM table. For any gene in the model that is missing from that table, `seq_data.get(gene_id, 0.0)` returns **0.0** (lines 531–551). No warning is issued.

## Goal

- **Warn** once that genes are missing and are being filled from the basal-expression-condition reference (when the option is enabled).
- **Fill** those missing genes from the reference table: `raw_data.rna_seq_data.rnaseq_rsem_tpm_mean`, using the column `sim_data.basal_expression_condition` (same source/column as the legacy path).
- **Config option:** `rnaseq_fill_missing_genes_from_ref` (default `true`). When `true`, fill missing genes from reference and warn; when `false`, keep current behavior (missing → 0.0, no fill).

## Data available in context

- `raw_data` is already loaded by `KnowledgeBaseEcoli` from the flat files, so `raw_data.rna_seq_data.rnaseq_rsem_tpm_mean` is a list-of-dicts with keys `"Gene"` and condition names (e.g. `"M9 Glucose minus AAs"`). No extra I/O.
- `RNA_SEQ_ANALYSIS = "rsem_tpm"` in the same file (line 28), so the reference table name is `rnaseq_rsem_tpm_mean`.

## Implementation steps

### 0. Add config option and thread it through

- **Config:** Add `rnaseq_fill_missing_genes_from_ref` to `parca_options` in [configs/default.json](configs/default.json) (e.g. `true`) and [configs/templates/parca_standalone.json](configs/templates/parca_standalone.json) with a short comment.
- **runscripts/parca.py:** Pass `rnaseq_fill_missing_genes_from_ref=config["rnaseq_fill_missing_genes_from_ref"]` into `fitSimData_1(...)`.
- **fit_sim_data_1.py:** No signature change; `initialize(...)` already receives `**kwargs`. Ensure the `initialize()` wrapper passes `rnaseq_fill_missing_genes_from_ref=kwargs.get("rnaseq_fill_missing_genes_from_ref", True)` into `sim_data.initialize(...)`.
- **SimulationDataEcoli.initialize()** in [reconstruction/ecoli/simulation_data.py](reconstruction/ecoli/simulation_data.py): Add parameter `rnaseq_fill_missing_genes_from_ref=True`, set `self.rnaseq_fill_missing_genes_from_ref = rnaseq_fill_missing_genes_from_ref`.

Optional: add a CLI flag (e.g. `--rnaseq-fill-missing-genes-from-ref` / `--no-rnaseq-fill-missing-genes-from-ref`) in parca.py if command-line override is desired; otherwise config-only is sufficient.

### 1. In `_build_cistron_data()` (new-ingestion branch only)

**Location:** [reconstruction/ecoli/dataclasses/process/transcription.py](reconstruction/ecoli/dataclasses/process/transcription.py), inside the `if sim_data.rnaseq_manifest_path is not None:` block, after building `seq_data` from `ingest_transcriptome()`.

- **Keep an “experimental-only” mapping:** Use e.g. `seq_data_exp = dict(zip(tpm_table["gene_id"], tpm_table["tpm_mean"]))`. Let `seq_data` start as a copy of `seq_data_exp` (or the same dict); when fill-from-ref is enabled, `seq_data` will be updated with reference-filled values for missing genes.
- **When `sim_data.rnaseq_fill_missing_genes_from_ref` is true:** Build reference mapping from the same table/column as the legacy path: `ref_data = { x["Gene"]: x[sim_data.basal_expression_condition] for x in getattr(raw_data.rna_seq_data, f"rnaseq_{RNA_SEQ_ANALYSIS}_mean") }`.
- **Gate on config:** Only run the fill-from-reference logic when `sim_data.rnaseq_fill_missing_genes_from_ref` is true. When false, keep `seq_data` as experimental-only (current behavior: missing → 0.0) and use `seq_data` for both expression and coverage (so no change to existing behavior).
- **When true:**  
  - **Identify missing genes:** All model genes come from `cistron_id_to_gene_id.values()`. Compute the set of genes that are in the model but not in `seq_data_exp`.
  - **Fill and warn:** For each such `gene_id`, set `seq_data[gene_id] = ref_data.get(gene_id, 0.0)` (still 0.0 if missing from reference too). If the number of filled genes is greater than zero, issue a single warning (e.g. `warnings.warn`) stating that N genes were missing from the experimental dataset and were filled from the basal expression condition reference (`sim_data.basal_expression_condition`). Optionally include a short list of filled gene IDs (e.g. first 10).
  - **Use combined `seq_data` for expression, experimental-only for coverage:** When building `cistron_expression`, use `seq_data.get(gene_id, 0.0)`. When building `cistron_rnaseq_coverage`, use `gene_id in seq_data_exp` so that “covered” means “present in the experimental dataset” and genes filled from reference remain “not covered” for QC.

### 2. Warning style

- Use Python’s `warnings.warn()` (with a clear message and optional `UserWarning` category) so that callers can filter or promote to errors if desired. The codebase also uses `print("Warning: ...")` in a couple of places; `warnings.warn` is preferable for this case so that the message is consistent and filterable.

### 3. Edge cases

- **Gene in model but in neither experimental nor reference:** Keep 0.0; no need to warn again beyond the single “filled from reference” warning (the warning can state that some genes may still be 0 if missing from both).
- **Legacy path:** No change; behavior remains “use raw_data only” with no fill-from-reference step.
- **Reference table or column missing:** If `sim_data.basal_expression_condition` is not a key in the reference rows, the existing dict comprehension would raise or produce NaNs. If the codebase already assumes this column exists for the legacy path, the same assumption holds; otherwise a one-line check or try/except could be added and documented.

## Files to touch


| File                                                                                                                   | Change                                                                                                                                                                                                                                                                                   |
| ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [configs/default.json](configs/default.json)                                                                           | Add `"rnaseq_fill_missing_genes_from_ref": true` under `parca_options`.                                                                                                                                                                                                                  |
| [configs/templates/parca_standalone.json](configs/templates/parca_standalone.json)                                     | Add the key and a short comment.                                                                                                                                                                                                                                                         |
| [runscripts/parca.py](runscripts/parca.py)                                                                             | Pass `rnaseq_fill_missing_genes_from_ref` into `fitSimData_1(...)`.                                                                                                                                                                                                                      |
| [reconstruction/ecoli/fit_sim_data_1.py](reconstruction/ecoli/fit_sim_data_1.py)                                       | In `initialize()` wrapper, pass `rnaseq_fill_missing_genes_from_ref` into `sim_data.initialize(...)`.                                                                                                                                                                                    |
| [reconstruction/ecoli/simulation_data.py](reconstruction/ecoli/simulation_data.py)                                     | In `initialize()`, add parameter and set `self.rnaseq_fill_missing_genes_from_ref`.                                                                                                                                                                                                      |
| [reconstruction/ecoli/dataclasses/process/transcription.py](reconstruction/ecoli/dataclasses/process/transcription.py) | In `_build_cistron_data()`, new-ingestion branch: add `seq_data_exp`; when `sim_data.rnaseq_fill_missing_genes_from_ref` is true, build `ref_data`, fill missing from ref, warn once; set coverage from `seq_data_exp` when fill-from-ref is true. Add `import warnings` if not present. |


## Optional follow-up

- Unit test: with a small fixture where the experimental TPM table is missing one or two genes that exist in the reference, assert that (1) their TPMs come from the reference, (2) a warning is raised, and (3) `_cistron_is_rnaseq_covered` is False for those genes.

