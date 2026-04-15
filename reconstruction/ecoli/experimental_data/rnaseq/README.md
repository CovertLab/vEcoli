# Deprecated: RNA-seq TSVs now live in `ecoli-sources`

**This directory is being phased out.**

Public RNA-seq datasets have moved to the sibling repo
[`ecoli-sources`](https://github.com/…/ecoli-sources) at
`ecoli-sources/data/`. Configs reference it via
`"rnaseq_manifest_path": "$ECOLI_SOURCES/data/manifest.tsv"` (resolved by
`wholecell.io.ingestion.resolve_ecoli_sources_path`).

What remains here is **private** Ginkgo Bioworks (`gbw_*`) data that cannot
go into `ecoli-sources`. Once a private-overlay repo is set up
(`$ECOLI_SOURCES_PRIVATE`), these files will also be removed and this
directory will be deleted entirely.

Do not add new datasets here. New TSVs go in `ecoli-sources/data/` (public)
or the private overlay (private).
