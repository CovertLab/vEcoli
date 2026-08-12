# yggX homolog expression across pathogens

This folder finds *yggX* (Fe-S cluster oxidative-damage-protection protein, Pfam **PF04362**) and its
homologs across the 32 pathogens in the GSE152295 RNA-seq atlas (Avican et al. 2021,
*Nature Communications*) and produces two supplementary figures showing that yggX is **not an
*E. coli*-only protein** — coordinate-confirmed PF04362 homologs are present, and highly expressed,
across diverse Gram-negative pathogens.

The two figures live in the notebook `../supp_yggX.ipynb`:
- `ygghomologrank` → `figures/yggX_rank_plot.{svg,png}` — expression rank of each homolog within its
  own genome under the control condition.
- `ygghomologoxs` → `figures/yggX_oxidative_plot.{svg,png}` — expression rank under oxidative stress,
  dot colour = fold change vs control.

---

## Quickstart — replicate the figures

The committed inputs (`output/yggX_expression.csv`, `cache/frozen_hits.json`) already contain
everything the figures need, **except** the raw GEO expression files (`data/gse152295/`, ~24 MB), which
are downloaded on first run. From inside this folder:

```bash
# 1. Environment (from the vEcoli repo — packages are already in pyproject.toml)
uv sync --frozen
#    (standalone checkout instead: `pip install -r requirements.txt`, then drop the `uv run` prefix)

# 2. Download the 32 GEO source files into data/gse152295/  (only needed once; resume-safe)
uv run python download_geo_gse152295.py

# 3. Rebuild output/yggX_expression.csv from the committed frozen artifact — no network, pandas+numpy only
uv run python search_gene_across_pathogens.py --use-frozen

# 4. Generate the figures
#    Open ../supp_yggX.ipynb and run cells `ygghomologrank` and `ygghomologoxs`.
#    They read output/yggX_expression.csv + data/gse152295/ and write to ../figures/.
```

Step 3 (`--use-frozen`) is the recommended path for reviewers: it re-selects each matched gene by
locus in the downloaded GEO files and recomputes TPM/fold-change from the frozen coordinates, producing a
CSV identical to the original online run **without any NCBI calls**. This is because if there are any updates to NCBI calls from future updates, reproducibility becomes tricky. If `output/yggX_expression.csv`
is already present and you trust it, you can skip straight from step 2 to step 4.

---

**yggX** (*E. coli* K-12 locus; also "Fe(2+)-trafficking protein" / "oxidative damage protection
protein") Its family is Pfam **PF04362**.

- Reference protein: **NP_417437.1** (*E. coli* K-12, yggX) → WP_000091700.1
- Pfam family: **PF04362**

---

## Files

```
yggX_homologs/
  ├── README.md                        # this file
  ├── requirements.txt                 # pip deps (biopython, pandas, numpy, requests, matplotlib, seaborn)
  ├── download_geo_gse152295.py        # step 2: downloads the 32 GEO source files
  ├── search_gene_across_pathogens.py  # step 3: finds the homolog in each of 32 GEO strains
  ├── cache/
  │   └── frozen_hits.json             # committed resolved coordinates → offline, no-network reproduction
  ├── data/
  │   └── gse152295/                   # GEO source files (created by download step; ~24 MB, 32 .txt.gz)
  └── output/
      └── yggX_expression.csv          
```
---

## Full (online) regeneration

To regenerate `output/yggX_expression.csv` from live NCBI instead of the frozen artifact — this is how
the committed CSV and `cache/frozen_hits.json` were originally produced:

```bash
uv run python search_gene_across_pathogens.py \
    --protein NP_417437.1 \
    --pfam    PF04362 \
    --gene    yggX \
    --secondary-proteins WP_000091706.1 WP_005192570.1 \
    --overrides YPSTB:YPK_0819 \
    --no-scan
```

An online run rewrites `cache/frozen_hits.json` automatically (via `--frozen-out`). `--no-scan` skips
the slow genome-scan strategy, which never contributes a hit for this set, so the result is identical
to a full run but far faster.

**Why the extra arguments:**
- `--secondary-proteins WP_000091706.1` — *Salmonella* SL1344 (SALMT) uses a slightly diverged protein
  absent from the *E. coli* reference IPG; adding its WP_ accession enables coordinate-based matching.
- `--secondary-proteins WP_005192570.1` — covers additional *Yersinia*-lineage entries (minor, kept
  for completeness).
- `--overrides YPSTB:YPK_0819` — NCBI retired the old NC_010465 assembly from the IPG (replaced by
  NZ_CP032566.1), so automated coordinate matching fails for this strain; the locus tag is supplied
  directly.

---

## How `search_gene_across_pathogens.py` resolves a homolog

Three strategies are tried in order per strain of the 32 downlaoded, stopping at the first match:

1. **IPG** — Fetch the Identical Protein Group XML for the reference WP_ (and any secondary WP_s). The
   IPG lists every genome encoding the exact same protein with 1-based chromosomal coordinates; match
   by coordinate against the GEO file's `Region` column.
2. **Family** — If IPG misses a strain, search NCBI protein by Pfam domain or product description
   within that taxon. *Every* candidate a sub-query returns is coordinate-confirmed via its own IPG,
   and the confirmed candidate with the highest sequence identity to the reference is kept.
3. **Genome scan** — Download the primary chromosome in 400 kb windows and scan every CDS by gene name
   or product keywords. Slow but catches poorly indexed annotations (disabled with `--no-scan`; never
   contributes a hit for this set).

After a match, `fetch_gene_annotation()` fetches the exact genome slice from NCBI nuccore and parses
the CDS by locus tag to recover `gene_name` and `product`.

### Robustness & reproducibility notes
- **Coordinate matching** is exact-string-first; only on failure does a tolerant fallback run (±10 bp,
  handles `join(...)` spans), labelled `IPG(approx)` so approximate hits never masquerade as exact.
  None were needed for the current 15-species set.
- **NCBI calls retry** transient failures (3 attempts, exponential backoff).
- **Sequence-identity column.** IPG hits are sequence-identical by definition (`pct_identity` = 100).
  Family/Scan hits are aligned to the reference (global BLOSUM62 via `Bio.Align.PairwiseAligner`) and
  their `pct_identity` / `aln_len` recorded. This is an auditable column, **not a filter** — hits below
  `--min-identity` (default 30%) are kept with a stderr warning. Override hits leave `pct_identity`
  blank.
- **Condition parsing** matches known condition tokens (`Ctrl`, `Oxs`, …) rather than underscore
  position, so codes containing underscores (e.g. HPG27's `HP_G27_Ctrl_1`) parse correctly.
- **Provenance columns** in `yggX_expression.csv` (`match_wp`, `match_accession`, `match_region`,
  `pct_identity`, `aln_len`, `ref_protein`, `ref_wp`, `query_date`) record which protein/coordinate
  resolved each species and when. Fold-change columns are raw mean-TPM ratios (n≤3, no statistical
  test) — descriptive only.

---

