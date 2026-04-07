---
name: sms_omics reference
description: Upstream data repo for RNAseq datasets ingested into vEcoli; feedback loop for parca failure analysis
type: reference
---

Companion repository: `/Users/chris/projects/sms_omics`

**Role:** Upstream data pipeline. Owns `data_formatted/` (per-dataset TSVs + manifest.tsv),
which is manually copied into `vEcoli/reconstruction/ecoli/experimental_data/rnaseq/` for ingestion.

**Feedback interface:** After parca runs, write summaries to `sms_omics/analysis/model_results/`
as CSV files with date-prefix names (e.g. `06apr2026_01_parca_summary.csv`). Schema columns:
`parca_id`, `dataset_id`, `parca_status`, `parca_error`, `parca_duration_min`, generation counts.

**Data pipeline:** Raw data → notebook 1 (normalizes, builds `data/data_all.h5ad`) → notebook 2
(filters by condition, writes per-dataset TSVs with schema validation).

**Key policy:** Never edit `data_formatted/` directly — all changes go through notebook 2 or
`processing/post_processing.py` (which validates against `RnaseqTpmTableSchema`).

**Active research questions (per CLAUDE.md):**
- Why do some datasets fail parca (kcat or P-solve)?
- Can pre-screening flag likely failures before ingestion?
- Can gene-level exclusions rescue failing datasets?
