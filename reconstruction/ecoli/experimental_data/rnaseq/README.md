# Empty: RNA-seq data has moved

This directory previously held RNA-seq TPM TSVs + a manifest. All datasets
have been migrated to sibling repos:

- **Public datasets** (`vecoli_*`, `precise_*`) →
  [`ecoli-sources`](https://github.com/…/ecoli-sources) at `$ECOLI_SOURCES/data/`.
- **Private overlays** (e.g. `gbw_vegas_*`) → separate private repos
  (e.g. `ecoli-sources-vegas/data/`); loaded via the
  `$ECOLI_SOURCES_OVERLAYS` environment variable.

vEcoli configs reference them as:

```json
"rnaseq_manifest_path": "$ECOLI_SOURCES/data/manifest.tsv"
```

Overlays are loaded automatically by `wholecell.io.ingestion.ingest_transcriptome`
whenever `ECOLI_SOURCES_OVERLAYS` is set (colon-separated list of overlay
manifest paths).

This directory is kept as a signpost; do not add data here.
