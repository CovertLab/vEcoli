#!/usr/bin/env python3
"""
search_gene_across_pathogens.py

Given a reference protein accession, find its homolog across the 32 pathogens
in the GSE152295 RNA-atlas dataset (Avican et al. 2021, Nature Communications)
and extract its stress-condition expression profile from the GEO files.

Three strategies are tried in order for each species:

  1. IPG   — Fetch the Identical Protein Group for the reference WP_ accession.
             The IPG lists every genome encoding the same (or sequence-identical)
             protein, with exact chromosomal coordinates.  Fastest and most precise.

  2. Family — If IPG misses a strain, search NCBI's protein database for the
             given Pfam family ID OR the gene/product name within that taxon.
             Every candidate returned by a sub-query (not just the first) is
             coordinate-confirmed via its own IPG, and the confirmed candidate with
             the highest sequence identity to the reference protein is kept.  Catches
             diverged homologs not in the first IPG.

  3. Scan  — Last resort.  Download the primary chromosome in 400 kb windows
             and scan every CDS feature for the gene name or product keywords.
             Slow (~20 s per genome) but works for poorly indexed annotations.

Coordinate matching is exact-string-first (start..stop / complement(...)), so the
default behaviour is unchanged.  Only when exact matching fails does a tolerant
fallback run: it parses the Region column numerically (handling join(...) spans)
and matches by coordinate proximity (+/- a few bp), rescuing genes lost to minor
assembly/annotation drift.  Tolerant IPG matches are labelled "IPG(approx)" so they
are distinguishable in the output.

Sequence-identity control: IPG hits are sequence-identical to the reference by
definition (pct_identity = 100).  Family/Scan hits, which resolve a candidate by
family/keyword + coordinate confirmation, are additionally aligned (global BLOSUM62)
to the reference and their % identity is recorded (pct_identity / aln_len).  A hit
below --min-identity is KEPT but flagged with a stderr warning -- identity is an
auditable control, never a silent filter.

Provenance: every found row records the WP accession that resolved it (match_wp),
the genome accession and Region it matched (match_accession / match_region), the
sequence identity to the reference (pct_identity / aln_len), and the run date
(query_date), so a reviewer can reproduce or audit each call.

Caveats for interpretation:
  * Fold changes (<cond>_fc) are raw ratios of mean replicate TPM (n<=3) to mean
    control TPM.  They are point estimates with NO dispersion, significance test, or
    multiple-testing correction -- treat them as descriptive, not as DE calls.
  * Percentile rank (computed downstream in ../supp_yggX.ipynb) is taken over every row
    of each GEO file; the gene-set denominator is not identical across species.
  * pct_identity is blank for Override hits (manual locus, not sequence-verified).

Usage:
    uv run python search_gene_across_pathogens.py \\
        --protein NP_417437.1 \\
        --pfam    PF04362     \\
        --gene    yggX        \\
        --out     output/yggX_expression.csv

    # Skip the slow genome scan:
    ... --no-scan

    # Show this help:
    ... --help
"""

import argparse
import datetime
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from Bio import Entrez, SeqIO
except ImportError:  # pragma: no cover
    Entrez = SeqIO = None  # type: ignore

# 32 study strains from Avican et al. 2021 / GSE152295
# taxid values are for the specific strain used, matching the GEO deposit.
STRAINS = [
    # (dataset_code,  display_name,                                    taxid)
    ("ACHX", "Achromobacter xylosoxidans SOLR10", "762376"),
    ("ACIB", "Acinetobacter baumannii AB5075-UW", "509173"),
    ("AGGA", "Aggregatibacter actinomycetemcomitans D7S-1", "557685"),
    ("BBURG", "Borrelia burgdorferi B31", "224326"),
    ("BURK", "Burkholderia pseudomallei K96243", "272560"),
    ("Campy", "Campylobacter jejuni 81-176", "407148"),
    ("ENTFA", "Enterococcus faecalis OG1RF", "474186"),
    ("EPEC", "Escherichia coli O127:H6 E2348/69", "155864"),
    ("ETEC", "Escherichia coli H10407", "1045013"),
    ("FRAT", "Francisella tularensis FSC200", "393011"),
    ("HINF", "Haemophilus influenzae 86-028NP", "281310"),
    ("HPG27", "Helicobacter pylori G27", "563041"),
    ("HPJ99", "Helicobacter pylori J99", "85963"),
    ("KLEBS", "Klebsiella pneumoniae MGH 78578", "272620"),
    ("LEGIP", "Legionella pneumophila Philadelphia 1", "272624"),
    ("Listeria", "Listeria monocytogenes EGD-e", "169963"),
    ("MRSA", "Staphylococcus aureus MRSA252", "176279"),
    ("MSSA", "Staphylococcus aureus MSSA476", "176280"),
    ("MTB", "Mycobacterium tuberculosis H37Ra", "83331"),
    ("NGON", "Neisseria gonorrhoeae FA 1090", "242231"),
    ("NMEN", "Neisseria meningitidis FAM18", "272831"),
    ("PSEUDO", "Pseudomonas aeruginosa PAO1", "208964"),
    ("SALMT", "Salmonella enterica SL1344", "216597"),
    ("SEPI", "Staphylococcus epidermidis 1457", "638300"),
    ("SHIF", "Shigella flexneri 5a M90T", "198215"),
    ("SPYO", "Streptococcus pyogenes 5448", "319939"),
    ("SSUIS", "Streptococcus suis P1/7", "372430"),
    ("STAGA", "Streptococcus agalactiae NEM316", "197061"),
    ("STRPN", "Streptococcus pneumoniae D39", "373153"),
    ("UPEC", "Escherichia coli 536", "362663"),
    ("Vibrio", "Vibrio cholerae O1", "345073"),
    ("YPSTB", "Yersinia pseudotuberculosis YPIII", "273123"),
]

# Conditions as they appear in GEO column names and their human-readable names
CONDITION_LABELS = {
    "As": "Acidic stress",
    "Bs": "Bile stress",
    "Li": "Low iron",
    "Mig": "Hypoxia",
    "Nd": "Nutritional downshift",
    "Ns": "Nitrosative stress",
    "Oss": "Osmotic stress",
    "Oxs": "Oxidative stress",
    "Sp": "Stationary phase",
    "Tm": "Temperature stress",
    "Vic": "Virulence-inducing condition",
}

# Rate-limited NCBI calls
# 3 req/s without API key, 10 req/s with key
_last_ncbi_call: float = 0.0
NCBI_DELAY: float = 0.4
NCBI_RETRIES: int = 3


def ncbi(fn, *args, **kwargs):
    """
    Call an Entrez function with global rate limiting and bounded retries.

    Retrying is what lets callers treat an exception as a *genuine* negative
    rather than a transient network/NCBI hiccup: a real 'not found' only surfaces
    after the retries are exhausted, so a momentary blip cannot silently drop a
    species from the results.
    """
    global _last_ncbi_call
    last_exc: Exception | None = None
    for attempt in range(NCBI_RETRIES):
        wait = NCBI_DELAY - (time.time() - _last_ncbi_call)
        if wait > 0:
            time.sleep(wait)
        try:
            result = fn(*args, **kwargs)
            _last_ncbi_call = time.time()
            return result
        except Exception as exc:
            last_exc = exc
            _last_ncbi_call = time.time()
            time.sleep(0.5 * (2**attempt))
    raise last_exc if last_exc else RuntimeError("ncbi() failed with no exception")


# Reference protein helpers
def get_wp_accession(protein_id: str) -> str | None:
    """
    Return the WP_ (non-redundant RefSeq) accession for any protein ID.
    If the input is already a WP_ accession, return it directly.
    """
    if re.match(r"WP_\d+", protein_id):
        return protein_id
    handle = ncbi(
        Entrez.efetch, db="protein", id=protein_id, rettype="gb", retmode="text"
    )
    text = handle.read()
    handle.close()
    m = re.search(r"WP_\d+\.\d+", text)
    return m.group() if m else None


def fetch_protein_seq(protein_id: str) -> str:
    """
    Return the amino-acid sequence (no FASTA header, no whitespace) for a protein
    accession, or "" on failure.  Used to compute the sequence-identity control for
    non-IPG hits (Strategies 2 & 3)
    """
    try:
        handle = ncbi(
            Entrez.efetch, db="protein", id=protein_id, rettype="fasta", retmode="text"
        )
        text = handle.read()
        handle.close()
    except Exception:
        return ""
    lines = [ln.strip() for ln in text.splitlines() if ln and not ln.startswith(">")]
    return "".join(lines)


_ALIGNER = None


def _get_aligner():
    global _ALIGNER
    if _ALIGNER is None:
        from Bio.Align import PairwiseAligner, substitution_matrices

        aln = PairwiseAligner()
        aln.mode = "global"
        aln.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aln.open_gap_score = -11
        aln.extend_gap_score = -1
        _ALIGNER = aln
    return _ALIGNER


def protein_identity(seq_a: str, seq_b: str) -> tuple[float, int]:
    """
    Return (pct_identity, aln_len) for a global BLOSUM62 alignment of two protein
    sequences.  aln_len is the number of alignment columns (residues + gaps) and
    pct_identity = 100 * matches / aln_len -- identity over the full alignment
    length, so length differences (gaps) count against identity.  This is the
    conservative, standard reading for whole-protein homology.

    Returns (0.0, 0) if either sequence is empty or alignment fails.
    """
    if not seq_a or not seq_b:
        return 0.0, 0
    try:
        aligner = _get_aligner()
        aln = aligner.align(seq_a, seq_b)[0]
    except Exception:
        return 0.0, 0

    # Count identical positions by walking the ungapped aligned index blocks:
    # aln.aligned is ((a_start, a_stop)...), ((b_start, b_stop)...) of matched spans.
    matches = 0
    a_blocks, b_blocks = aln.aligned
    for (a0, a1), (b0, b1) in zip(a_blocks, b_blocks):
        matches += sum(1 for x, y in zip(seq_a[a0:a1], seq_b[b0:b1]) if x == y)

    # Denominator: total alignment columns (including gap columns).  aln.length is
    # available in biopython >= 1.80; fall back to the longer sequence otherwise.
    aln_len = int(getattr(aln, "length", 0)) or max(len(seq_a), len(seq_b))
    if aln_len == 0:
        return 0.0, 0
    return round(100.0 * matches / aln_len, 1), aln_len


def get_product_info(protein_id: str) -> tuple[str, list[str]]:
    """
    Return (full_product_name, [significant_search_terms]) for a protein.
    Terms are used in Strategies 2 and 3 for matching product descriptions.
    """
    handle = ncbi(
        Entrez.efetch, db="protein", id=protein_id, rettype="gb", retmode="text"
    )
    text = handle.read()
    handle.close()

    product = ""
    for line in text.splitlines():
        if "/product=" in line:
            product = line.strip().replace("/product=", "").strip('"')
            break

    # Build search terms from the product description.
    # Priority: most-specific first (full product phrase, then normalized variants,
    # then individual meaningful words).  Avoid partial/malformed strings — they
    # cause false-positive matches in NCBI protein searches.
    terms: list[str] = []

    # 1. The full product string stripped of generic prefixes
    skip_words = {"putative", "probable", "hypothetical"}
    clean = " ".join(w for w in product.split() if w.lower() not in skip_words).strip()
    if clean and clean.lower() != "protein":
        terms.append(clean.lower())

    # 2. A normalized variant: collapse nested parentheses like Fe(2(+)) → Fe(2+)
    normalized = re.sub(r"\(([^()]+)\(([^()]+)\)\)", r"(\1\2)", clean)
    if normalized.lower() != clean.lower():
        terms.append(normalized.lower())

    # 3. The most descriptive hyphenated compound (e.g. "Fe(2+)-trafficking protein")
    # Take only the largest hyphenated word that doesn't have unbalanced parens
    for w in clean.split():
        if "-" in w and w.count("(") == w.count(")"):
            terms.append(w.lower())

    return product, terms


# Gene annotation lookup
def fetch_gene_annotation(geo_row: "pd.Series") -> tuple[str, str]:
    """
    Return (gene_name, product) by fetching the exact genome slice for a GEO
    row using its Chromosome and Region columns.  This avoids locus-tag indexing
    differences between old-style (lpg1927) and RS-style (ABUW_RS17615) tags.
    gene_name is empty string when the CDS has no /gene= qualifier.
    """
    import urllib.request as _urlreq

    chrom_raw = str(geo_row["Chromosome"])
    region_str = str(geo_row["Region"])

    # Strip author-added suffixes like _mix_r_dep to get a bare NCBI accession
    m = re.match(r"([A-Za-z0-9_]+?\d+)", chrom_raw)
    if not m:
        return "", ""
    chrom = m.group(1)

    # Parse start/stop from "start..stop", "complement(start..stop)", or
    # "join(a..b,c..d)".  Use the outer bounds (min/max of all coordinates) so a
    # join-format / origin-spanning gene fetches a valid (start <= stop) slice
    # instead of producing an inverted range.
    nums = re.findall(r"\d+", region_str)
    if len(nums) < 2:
        return "", ""
    coords = [int(n) for n in nums]
    start, stop = min(coords), max(coords)

    # Rate limit then fetch via raw URL (Biopython efetch doesn't support from/to)
    global _last_ncbi_call
    wait = NCBI_DELAY - (time.time() - _last_ncbi_call)
    if wait > 0:
        time.sleep(wait)
    api_param = f"&api_key={Entrez.api_key}" if Entrez.api_key else ""
    url = (
        f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        f"?db=nuccore&id={chrom}&rettype=gb&retmode=text"
        f"&from={start}&to={stop}{api_param}"
    )
    try:
        with _urlreq.urlopen(url, timeout=30) as r:
            text = r.read().decode()
    except Exception:
        return "", ""
    _last_ncbi_call = time.time()

    # Parse with BioPython and find the CDS matching the target locus tag.
    # A fetched slice may include neighboring genes, so we must not just take
    # the first /gene= and /product= — they could belong to an overlapping neighbor.
    from io import StringIO as _StringIO

    locus_tag = str(geo_row["Name"])
    try:
        record = SeqIO.read(_StringIO(text), "genbank")
    except Exception:
        return "", ""
    for feat in record.features:
        if feat.type != "CDS":
            continue
        lt = feat.qualifiers.get("locus_tag", [""])[0]
        old_lt = feat.qualifiers.get("old_locus_tag", [""])[0]
        if lt == locus_tag or old_lt == locus_tag:
            gene = feat.qualifiers.get("gene", [""])[0]
            product = feat.qualifiers.get("product", [""])[0]
            return gene, product
    return "", ""


# IPG (Identical Protein Group) helpers
def fetch_ipg_coords(wp_accession: str) -> dict:
    """
    Fetch the IPG XML for a WP_ accession.
    Returns {accession: [(start_1based, stop_1based, strand), ...]} for every CDS.
    NCBI's IPG XML reports 1-based inclusive coordinates, matching the GEO
    Region column format ("start..stop").
    Indexes by both versioned (NC_000913.3) and bare (NC_000913) accession.

    A genome accession may carry more than one identical-protein CDS (duplicated
    genes), so coordinates are accumulated in a list rather than overwritten --
    every candidate copy is then tried during matching.
    """
    handle = ncbi(
        Entrez.efetch, db="protein", id=wp_accession, rettype="ipg", retmode="xml"
    )
    content = handle.read()
    handle.close()
    if isinstance(content, bytes):
        content = content.decode("utf-8")

    coords: dict[str, list] = {}
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return coords

    for cds in root.iter("CDS"):
        accver = cds.get("accver", "")
        if not accver:
            continue
        start = int(cds.get("start", 0))  # 1-based
        stop = int(cds.get("stop", 0))  # 1-based
        strand = cds.get("strand", "+")
        entry = (start, stop, strand)
        coords.setdefault(accver, []).append(entry)
        coords.setdefault(accver.split(".")[0], []).append(entry)  # bare accession

    return coords


# GEO coordinate matching
def _region_bounds(region_str: str) -> tuple[int, int] | None:
    """Return (min_coord, max_coord) parsed from any Region string, or None."""
    nums = re.findall(r"\d+", str(region_str))
    if not nums:
        return None
    ints = [int(n) for n in nums]
    return min(ints), max(ints)


def match_to_geo(
    geo_df: pd.DataFrame,
    start: int,
    stop: int,
    strand: str,
    tolerant: bool = False,
    tol: int = 10,
) -> pd.Series | None:
    """
    Match a 1-based (start, stop) coordinate pair to the GEO file's Region column.
    The Region column format is '3121811..3122086' or 'complement(3121811..3122086)'.
    Both orientations are tried regardless of the strand argument.

    Exact string matching is always attempted first
    When ``tolerant`` is set and exact matching fails, a numeric
    fallback parses the outer bounds of every Region (handling join(...) spans)
    and matches the closest gene whose start AND stop are within ``tol`` bp.  If
    two rows tie ambiguously the closest wins; if none is within tolerance the
    result is None (no wandering to an unrelated neighbour).

    Returns the matching DataFrame row, or None if not found.
    """
    for region in (f"{start}..{stop}", f"complement({start}..{stop})"):
        hit = geo_df[geo_df["Region"] == region]
        if not hit.empty:
            return hit.iloc[0]

    if not tolerant:
        return None

    q_lo, q_hi = min(start, stop), max(start, stop)
    best_idx, best_dist = None, None
    for idx, region in geo_df["Region"].items():
        bounds = _region_bounds(region)
        if bounds is None:
            continue
        r_lo, r_hi = bounds
        if abs(r_lo - q_lo) <= tol and abs(r_hi - q_hi) <= tol:
            dist = abs(r_lo - q_lo) + abs(r_hi - q_hi)
            if best_dist is None or dist < best_dist:
                best_idx, best_dist = idx, dist
    if best_idx is not None:
        return geo_df.loc[best_idx]
    return None


# Strategy 1: match reference protein IPG against GEO chromosomes


def strategy_ipg(
    geo_df: pd.DataFrame, ipg_coords: dict
) -> tuple[pd.Series | None, str, dict]:
    """
    Try every chromosome accession present in the GEO file against the
    pre-built IPG coordinate map.  Both versioned and bare accessions are tried,
    and every candidate CDS coordinate per accession is tried.

    Two passes are made: exact string matching first over ALL accessions, then --
    only if nothing matched exactly -- a tolerant pass.  This guarantees an exact
    match anywhere wins before any approximate match, so results never drift to a
    neighbouring gene when an exact hit exists.  Approximate matches are labelled
    "IPG(approx)".

    Returns (row, label, provenance) where provenance records match_accession and
    match_region for auditing.
    """
    chroms = list(geo_df["Chromosome"].unique())
    for tolerant in (False, True):
        for chrom in chroms:
            for key in (chrom, chrom.split(".")[0]):
                for start, stop, strand in ipg_coords.get(key, []):
                    row = match_to_geo(geo_df, start, stop, strand, tolerant=tolerant)
                    if row is not None:
                        label = "IPG(approx)" if tolerant else "IPG"
                        # IPG membership means a sequence-identical protein, so the
                        # identity control is 100% by definition (no alignment needed).
                        prov = {
                            "match_accession": key,
                            "match_region": str(row["Region"]),
                            "pct_identity": 100.0,
                            "aln_len": None,
                        }
                        return row, label, prov
    return None, "", {}


# Strategy 2: search by family (Pfam) or product/gene name, then IPG
FAMILY_RETMAX: int = 3


def strategy_family(
    taxid: str,
    pfam_id: str | None,
    gene_name: str,
    product_terms: list[str],
    geo_df: pd.DataFrame,
    ref_seq: str = "",
    min_identity: float = 30.0,
) -> tuple[pd.Series | None, str, dict]:
    """
    Search NCBI protein DB for a homolog in this taxon using Pfam family,
    gene name, or product description.  For each hit, fetch a fresh IPG
    and attempt coordinate matching against the GEO file.

    BEST-HIT SELECTION: each sub-query returns up to FAMILY_RETMAX candidates.  The
    old code requested 3 but blindly used IdList[0] -- arbitrary w.r.t. orthology when
    a taxon has several family members.  We now resolve *every* candidate that
    coordinate-confirms and keep the one with the highest sequence identity to the
    reference protein (ties -> first).  Sub-queries are tried in priority order
    (Pfam, then gene name, then product title) and the first sub-query that yields
    any confirmed candidate wins.

    IDENTITY CONTROL: because the GeneName and Product-title sub-queries are keyword
    matches that can return a paralog or an unrelated protein sharing a term, the
    resolved candidate's sequence is aligned to ``ref_seq`` and the % identity is
    recorded in the provenance (pct_identity / aln_len).  A hit below ``min_identity``
    is kept but flagged with a stderr WARNING -- identity is an auditable control,
    not a silent filter.  (If ``ref_seq`` is empty, identity is skipped/recorded 0.)

    Returns (row, label, provenance).  Organism scoping is uniform
    ([Organism:exp], exact strain, no subtree) across all sub-queries
    """
    queries: list[tuple[str, str]] = []

    if pfam_id:
        queries.append((f"txid{taxid}[Organism:exp] AND {pfam_id}[Pfam]", "Pfam"))

    if gene_name:
        queries.append(
            (f"txid{taxid}[Organism:exp] AND {gene_name}[Gene Name]", "GeneName")
        )

    # Product description: try the most distinctive phrases (multi-word first)
    multi = [t for t in product_terms if " " in t]
    single = [t for t in product_terms if " " not in t]
    for term in (multi + single)[:3]:
        queries.append((f'txid{taxid}[Organism:exp] AND "{term}"[Title]', "Product"))

    seq_cache: dict[str, str] = {}

    for query, label in queries:
        handle = ncbi(Entrez.esearch, db="protein", term=query, retmax=FAMILY_RETMAX)
        try:
            rec = Entrez.read(handle)
        except Exception:
            continue
        finally:
            handle.close()
        if not rec.get("IdList"):
            continue

        # Evaluate every candidate this sub-query returned; keep the coordinate-
        # confirmed one with the highest identity to the reference.
        best = None  # (pct_identity, row, ipg_label, prov)
        for cand_id in rec["IdList"]:
            wp = get_wp_accession(cand_id)
            if not wp:
                continue

            ipg = fetch_ipg_coords(wp)
            row, ipg_label, prov = strategy_ipg(geo_df, ipg)
            if row is None:
                continue

            if wp not in seq_cache:
                seq_cache[wp] = fetch_protein_seq(wp)
            pct, aln_len = protein_identity(ref_seq, seq_cache[wp])

            prov = {**prov, "match_wp": wp, "pct_identity": pct, "aln_len": aln_len}
            if best is None or pct > best[0]:
                best = (pct, row, ipg_label, prov)

        if best is not None:
            pct, row, ipg_label, prov = best
            if ref_seq and pct < min_identity:
                print(
                    f"    WARNING: Family({label}) hit {prov.get('match_wp')} has "
                    f"{pct:.1f}% identity to reference (< {min_identity:.0f}% floor) "
                    f"-- kept, verify orthology",
                    file=sys.stderr,
                )
            approx = "(approx)" if ipg_label == "IPG(approx)" else ""
            return row, f"Family({label}){approx}", prov

    return None, "", {}


# Strategy 3: genome scan (last resort)


def strategy_scan(
    geo_df: pd.DataFrame,
    gene_name: str,
    product_terms: list[str],
    window: int = 400_000,
    ref_seq: str = "",
    min_identity: float = 30.0,
) -> tuple[pd.Series | None, str, dict]:
    """
    Download the genome in 400 kb windows and scan every CDS feature for a
    match on gene name or product description keywords.

    The primary chromosome is the one with the most rows in the GEO file.
    Author-added suffixes (e.g. '_mix_r_dep') are stripped before fetching.

    Because ncbi() now retries transient failures internally, an exception raised
    out of the fetch here is treated as a genuine end-of-sequence signal rather
    than a network blip.  A gene straddling a 400 kb window boundary can still be
    missed (windows do not overlap) -- this is the least-precise strategy and was
    not needed for any species in the published set.

    Returns (row, label, provenance).
    """
    chromosomes = geo_df["Chromosome"].value_counts().index.tolist()
    # Use all product terms as substring checks against the GenBank product qualifier.
    # Multi-word terms like "fe(2+)-trafficking protein" are more specific than single
    # words and reduce false positives in the genome scan context.
    scan_terms = list(product_terms)

    # Estimate genome size from GEO Region column to avoid over-scanning.
    # Handles both "start..stop" and "complement(start..stop)" formats.
    def _max_coord(region_str: str) -> int:
        nums = re.findall(r"\d+", str(region_str))
        return max(int(n) for n in nums) if nums else 0

    for raw_chrom in chromosomes:
        chrom = re.sub(r"_mix.*$", "", raw_chrom)  # strip study-specific suffixes
        chrom_rows = geo_df[geo_df["Chromosome"] == raw_chrom]
        max_coord = chrom_rows["Region"].apply(_max_coord).max()
        scan_limit = min(int(max_coord) + window, 12_000_000)

        for start in range(0, scan_limit, window):
            try:
                handle = ncbi(
                    Entrez.efetch,
                    db="nucleotide",
                    id=chrom,
                    rettype="gb",
                    retmode="text",
                    seq_start=str(start + 1),
                    seq_stop=str(start + window),
                )
                record = SeqIO.read(handle, "genbank")
                handle.close()
            except Exception:
                break  # past end of sequence or network error

            cds_feats = [f for f in record.features if f.type == "CDS"]
            if not cds_feats:
                break  # empty window means we've gone past the chromosome end

            for feat in cds_feats:
                feat_gene = feat.qualifiers.get("gene", [""])[0].lower()
                feat_product = feat.qualifiers.get("product", [""])[0].lower()

                matches_gene = bool(gene_name) and gene_name.lower() in feat_gene
                matches_product = any(t in feat_product for t in scan_terms)

                if matches_gene or matches_product:
                    # BioPython uses 0-based start; convert to 1-based for match_to_geo
                    abs_start_1b = int(feat.location.start) + start + 1
                    abs_end_1b = int(feat.location.end) + start
                    row = match_to_geo(
                        geo_df, abs_start_1b, abs_end_1b, str(feat.location.strand)
                    )
                    if row is not None:
                        # Identity control: score the CDS translation against the
                        # reference (keyword/name matches can catch a paralog).
                        cand_seq = feat.qualifiers.get("translation", [""])[0]
                        pct, aln_len = protein_identity(ref_seq, cand_seq)
                        if ref_seq and pct < min_identity:
                            print(
                                f"    WARNING: Scan hit at {row['Name']} has "
                                f"{pct:.1f}% identity to reference "
                                f"(< {min_identity:.0f}% floor) -- kept, verify "
                                f"orthology",
                                file=sys.stderr,
                            )
                        prov = {
                            "match_accession": chrom,
                            "match_region": str(row["Region"]),
                            "pct_identity": pct,
                            "aln_len": aln_len,
                        }
                        return row, "Scan", prov

    return None, "", {}


# TPM extraction

# Recognised condition tokens: control + the 11 stress conditions.
KNOWN_CONDITIONS = ["Ctrl", *CONDITION_LABELS.keys()]
# Anchored '_<COND>_<replicate>' matcher.  Anchoring on a known token (rather than
# taking the 2nd underscore field) is robust to dataset codes that themselves
# contain underscores -- e.g. HPG27's columns are named 'HP_G27_Ctrl_1 (GE)', which
# the old positional parser mis-read as condition 'G27' with no 'Ctrl' at all.
_COND_RE = re.compile(r"_(" + "|".join(KNOWN_CONDITIONS) + r")_\d+")


def extract_mean_tpm(geo_df: pd.DataFrame, gene_row: pd.Series) -> dict:
    """
    Parse the GEO column names to recover condition labels and replicate numbers,
    then return the mean TPM across replicates for each condition.

    Column format: '1 - {CODE}_{CONDITION}_{REP} (GE) - TPM'
    e.g.          '1 - MTB_As_1 (GE) - TPM'  ->  condition 'As'
                  '1 - HP_G27_Ctrl_1 (GE) - TPM'  ->  condition 'Ctrl'

    Conditions are identified by matching a known condition token (see
    KNOWN_CONDITIONS), NOT by underscore position, so codes containing underscores
    are handled correctly.  Any TPM column that does not contain a recognised
    condition token is skipped with a warning rather than silently mislabelled.
    """
    tpm_cols = [c for c in geo_df.columns if c.endswith("- TPM")]
    by_cond: dict[str, list] = {}
    for col in tpm_cols:
        m = _COND_RE.search(col)
        if not m:
            print(
                f"    WARNING: could not parse condition from TPM column {col!r} "
                f"-- skipping",
                file=sys.stderr,
            )
            continue
        cond = m.group(1)
        by_cond.setdefault(cond, []).append(float(gene_row[col]))

    return {cond: round(float(np.mean(vals)), 2) for cond, vals in by_cond.items()}


# Record building + output (shared by the online and frozen paths)
def build_record(
    code: str,
    display_name: str,
    geo_df: pd.DataFrame,
    row: pd.Series,
    strategy: str,
    prov: dict,
    gene_name: str,
    product: str,
    query_date: str,
    ref_protein: str,
    ref_wp: str,
    cond_cols_seen: set,
) -> dict:
    """
    Assemble one found-species CSV record from a resolved GEO row.  Shared by the
    online resolver and the offline (frozen) reproducer so both emit identical
    rows.  Mutates ``cond_cols_seen`` with every stress condition encountered.
    """
    locus_tag = row["Name"]
    tpm = extract_mean_tpm(geo_df, row)
    ctrl_tpm = tpm.get("Ctrl", tpm.get("ctrl", np.nan))

    entry: dict = {
        "code": code,
        "name": display_name,
        "locus": locus_tag,
        "gene_name": gene_name,
        "product": product,
        "found": True,
        "strategy": strategy,
        "ctrl_tpm": ctrl_tpm,
        #  provenance for reproducibility / audit
        "match_wp": prov.get("match_wp", ""),
        "match_accession": prov.get("match_accession", ""),
        "match_region": prov.get("match_region", ""),
        # sequence-identity control: 100 for IPG (identical by definition),
        # aligned % for Family/Scan hits, blank for Override (manual/unverified).
        "pct_identity": prov.get("pct_identity", ""),
        "aln_len": prov.get("aln_len", ""),
        "ref_protein": ref_protein,
        "ref_wp": ref_wp or "",
        "query_date": query_date,
    }
    for cond, val in sorted(tpm.items()):
        if cond.lower() == "ctrl":
            continue
        entry[f"{cond}_tpm"] = val
        fc = round(val / ctrl_tpm, 3) if (ctrl_tpm and ctrl_tpm > 0) else None
        entry[f"{cond}_fc"] = fc
        cond_cols_seen.add(cond)
    return entry


def write_output_csv(
    records: list, cond_cols_seen: set, out_path: Path
) -> tuple[int, int]:
    """Write the expression CSV with a stable column order.  Returns (n_found, n_total)."""
    df_out = pd.DataFrame(records)

    # Reorder columns:
    meta_cols = [
        "code",
        "name",
        "locus",
        "gene_name",
        "product",
        "found",
        "strategy",
        "ctrl_tpm",
    ]
    stress_conds = sorted(cond_cols_seen - {"Ctrl", "ctrl"})
    value_cols = []
    for cond in stress_conds:
        if f"{cond}_tpm" in df_out.columns:
            value_cols.append(f"{cond}_tpm")
        if f"{cond}_fc" in df_out.columns:
            value_cols.append(f"{cond}_fc")
    prov_cols = [
        "match_wp",
        "match_accession",
        "match_region",
        "pct_identity",
        "aln_len",
        "ref_protein",
        "ref_wp",
        "query_date",
    ]
    prov_cols = [c for c in prov_cols if c in df_out.columns]
    all_cols = meta_cols + value_cols + prov_cols
    df_out = df_out.reindex(columns=all_cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    return int(df_out["found"].sum()), len(df_out)


def write_frozen(path: Path, meta: dict, hits: dict) -> None:
    """Write the small, committed frozen_hits.json reproducibility artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"meta": meta, "hits": hits}, f, indent=2)
        f.write("\n")


def run_frozen(frozen_path: Path, geo_dir: Path) -> tuple[list, set]:
    """
    Offline reproducer: rebuild the expression records purely from a committed
    frozen_hits.json plus the in-repo GEO files.  Makes NO network/Entrez calls,
    so it needs only pandas + numpy and reproduces a prior online run exactly.
    """
    data = json.loads(Path(frozen_path).read_text())
    meta = data.get("meta", {})
    hits = data.get("hits", {})
    query_date = meta.get("query_date", "")
    ref_protein = meta.get("ref_protein", "")
    ref_wp = meta.get("ref_wp", "")

    records: list = []
    cond_cols_seen: set = set()

    for code, display_name, _taxid in STRAINS:
        entry = hits.get(code)
        if entry is None or not entry.get("found"):
            print(f"  [{code:<10}] Not found (frozen)")
            records.append(
                {
                    "code": code,
                    "name": display_name,
                    "locus": None,
                    "found": False,
                    "strategy": None,
                }
            )
            continue

        geo_file = geo_dir / f"GSE152295_{code}_processed.txt.gz"
        if not geo_file.exists():
            print(f"  [{code:<10}] GEO file missing — skipping")
            records.append(
                {
                    "code": code,
                    "name": display_name,
                    "locus": None,
                    "found": False,
                    "strategy": None,
                }
            )
            continue

        geo_df = pd.read_csv(geo_file, sep="\t")
        locus = entry["locus"]

        # Re-select the exact matched row offline: by locus (Name), then by the
        # frozen Region string as a fallback.
        sel = geo_df[geo_df["Name"] == locus]
        if sel.empty and entry.get("match_region"):
            sel = geo_df[geo_df["Region"] == entry["match_region"]]
        if sel.empty:
            print(
                f"  [{code:<10}] frozen locus {locus!r} not found in GEO file "
                f"— skipping",
                file=sys.stderr,
            )
            records.append(
                {
                    "code": code,
                    "name": display_name,
                    "locus": None,
                    "found": False,
                    "strategy": None,
                }
            )
            continue
        if len(sel) > 1:
            print(
                f"  [{code:<10}] WARNING: frozen locus {locus!r} is ambiguous "
                f"({len(sel)} rows) — using the first",
                file=sys.stderr,
            )
        row = sel.iloc[0]

        prov = {
            "match_wp": entry.get("match_wp", ""),
            "match_accession": entry.get("match_accession", ""),
            "match_region": entry.get("match_region", ""),
            "pct_identity": entry.get("pct_identity", ""),
            "aln_len": entry.get("aln_len", ""),
        }
        rec = build_record(
            code,
            display_name,
            geo_df,
            row,
            entry["strategy"],
            prov,
            entry.get("gene_name", ""),
            entry.get("product", ""),
            query_date,
            ref_protein,
            ref_wp,
            cond_cols_seen,
        )
        print(
            f"  [{code:<10}] {rec['locus']:<25} via {rec['strategy']:<20} "
            f"ctrl={rec['ctrl_tpm']:.1f} TPM  (frozen)"
        )
        records.append(rec)

    return records, cond_cols_seen


# Main


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _here = Path(__file__).parent
    ap.add_argument(
        "--protein",
        default=None,
        help="Reference protein accession (e.g. NP_417437.1). "
        "Required unless --use-frozen is given.",
    )
    ap.add_argument(
        "--pfam", default=None, help="Pfam family ID for Strategy 2 (e.g. PF04362)"
    )
    ap.add_argument(
        "--gene", default="", help="Gene name for Strategies 2 & 3 (e.g. yggX)"
    )
    ap.add_argument(
        "--geo-dir",
        default=str(_here / "data" / "gse152295"),
        help="Directory containing GSE152295_*_processed.txt.gz",
    )
    ap.add_argument(
        "--out",
        default=str(_here / "output" / "yggX_expression.csv"),
        help="Output CSV path",
    )
    ap.add_argument(
        "--use-frozen",
        nargs="?",
        const=str(_here / "cache" / "frozen_hits.json"),
        default=None,
        help="Offline reproduce mode: rebuild results from a committed "
        "frozen_hits.json (NO NCBI, needs only pandas+numpy). "
        "Optional path; defaults to cache/frozen_hits.json.",
    )
    ap.add_argument(
        "--frozen-out",
        default=str(_here / "cache" / "frozen_hits.json"),
        help="Where an online run writes the frozen_hits.json artifact.",
    )
    ap.add_argument(
        "--email",
        default="aj0204@stanford.edu",
        help="Email for NCBI Entrez (required by NCBI)",
    )
    ap.add_argument(
        "--no-scan", action="store_true", help="Skip the slow genome scan (Strategy 3)"
    )
    ap.add_argument(
        "--min-identity",
        type=float,
        default=30.0,
        help="Sequence-identity floor (%%) for Family/Scan hits. Hits "
        "below it are KEPT but flagged with a stderr warning "
        "(identity is a recorded control, not a filter). Default 30.",
    )
    ap.add_argument(
        "--alt-products",
        nargs="*",
        default=[],
        help="Additional product description substrings to search in genome scan",
    )
    ap.add_argument(
        "--secondary-proteins",
        nargs="*",
        default=[],
        help="Additional WP_ protein accessions whose IPGs are merged with the"
        " reference for Strategy 1 (e.g. WP_000091706.1 WP_005192570.1)",
    )
    ap.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Direct locus-tag overrides bypassing all strategies, "
        "for cases where NCBI database drift breaks automated search. "
        "Format: CODE:LOCUS_TAG (e.g. YPSTB:YPK_0819)",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    geo_dir = Path(args.geo_dir)

    # Offline reproduce mode: rebuild from the frozen artifact, no NCBI
    if args.use_frozen:
        frozen_path = Path(args.use_frozen)
        if not frozen_path.exists():
            ap.error(f"--use-frozen file not found: {frozen_path}")
        print(f"Reproducing offline from frozen artifact: {frozen_path}")
        print("(no NCBI calls; requires only pandas + numpy)\n")
        records, cond_cols_seen = run_frozen(frozen_path, geo_dir)
        n_found, n_total = write_output_csv(records, cond_cols_seen, out_path)
        print()
        print(f"Found in {n_found}/{n_total} species (frozen).")
        print(f"Results written to: {out_path}")
        return

    if not args.protein:
        ap.error("--protein is required unless --use-frozen is given")

    # Parse overrides into {CODE: locus_tag}
    locus_overrides: dict[str, str] = {}
    for item in args.overrides or []:
        if ":" not in item:
            ap.error(f"--overrides item must be CODE:LOCUS_TAG, got: {item!r}")
        code_key, locus_val = item.split(":", 1)
        locus_overrides[code_key.strip()] = locus_val.strip()

    # Configure Entrez
    Entrez.email = args.email
    global NCBI_DELAY
    if os.environ.get("NCBI_API_KEY"):
        Entrez.api_key = os.environ["NCBI_API_KEY"]
        NCBI_DELAY = 0.12

    #  Step 1: Build global IPG from the reference protein
    print(f"Reference protein : {args.protein}")
    print(f"Pfam family       : {args.pfam or '(not provided)'}")
    print(f"Gene name         : {args.gene or '(not provided)'}")
    print(f"GEO directory     : {geo_dir}")
    print()

    print("Fetching reference protein info...", flush=True)
    wp_ref = get_wp_accession(args.protein)
    product_name, product_terms = get_product_info(args.protein)
    # Reference amino-acid sequence for the identity control on Family/Scan hits.
    ref_seq = fetch_protein_seq(args.protein)
    # Append any user-supplied alternative product descriptions (used in genome scan)
    for alt in args.alt_products:
        if alt.lower() not in product_terms:
            product_terms.append(alt.lower())
    print(f"  WP_ accession   : {wp_ref}")
    print(f"  Product         : {product_name}")
    print(f"  Search terms    : {product_terms}")
    print(
        f"  Reference length: {len(ref_seq)} aa"
        + ("" if ref_seq else "  (identity control disabled -- no ref sequence)")
    )

    ipg_coords: dict = {}
    if wp_ref:
        print(f"  Fetching IPG for {wp_ref}...", flush=True)
        ipg_coords = fetch_ipg_coords(wp_ref)
        print(f"  IPG covers {len(ipg_coords) // 2} unique genome accessions")
    for sec_wp in args.secondary_proteins:
        print(f"  Fetching secondary IPG for {sec_wp}...", flush=True)
        sec_coords = fetch_ipg_coords(sec_wp)
        ipg_coords.update(sec_coords)
        print(f"  IPG now covers {len(ipg_coords) // 2} unique genome accessions")
    print()

    #  Step 2: Process each strain
    records = []
    cond_cols_seen: set = set()
    frozen_hits: dict = {}
    query_date = datetime.date.today().isoformat()

    for code, display_name, taxid in STRAINS:
        geo_file = geo_dir / f"GSE152295_{code}_processed.txt.gz"
        if not geo_file.exists():
            print(f"  [{code:<10}] GEO file missing — skipping")
            continue

        geo_df = pd.read_csv(geo_file, sep="\t")

        prov: dict = {}
        # Check for manual override (bypasses automated strategies)
        if code in locus_overrides:
            override_locus = locus_overrides[code]
            hits = geo_df[geo_df["Name"] == override_locus]
            row = hits.iloc[0] if not hits.empty else None
            strategy = "Override"
            if row is not None:
                prov = {
                    "match_accession": str(row["Chromosome"]),
                    "match_region": str(row["Region"]),
                }
        else:
            row, strategy = None, ""

        # Try strategies in order (skipped if override matched)
        if row is None and strategy != "Override":
            row, strategy, prov = strategy_ipg(geo_df, ipg_coords)

        if row is None and strategy != "Override":
            row, strategy, prov = strategy_family(
                taxid,
                args.pfam,
                args.gene,
                product_terms,
                geo_df,
                ref_seq=ref_seq,
                min_identity=args.min_identity,
            )

        if row is None and strategy != "Override" and not args.no_scan:
            print(
                f"  [{code:<10}] Strategies 1 & 2 failed — running genome scan...",
                flush=True,
            )
            row, strategy, prov = strategy_scan(
                geo_df,
                args.gene,
                product_terms,
                ref_seq=ref_seq,
                min_identity=args.min_identity,
            )

        #  Build output record
        if row is None:
            print(f"  [{code:<10}] Not found")
            records.append(
                {
                    "code": code,
                    "name": display_name,
                    "locus": None,
                    "found": False,
                    "strategy": None,
                }
            )
            frozen_hits[code] = {"found": False}
            continue

        gene_name, product = fetch_gene_annotation(row)

        entry = build_record(
            code,
            display_name,
            geo_df,
            row,
            strategy,
            prov,
            gene_name,
            product,
            query_date,
            args.protein,
            wp_ref or "",
            cond_cols_seen,
        )

        # Record the resolved coordinates for the frozen (offline) artifact.
        frozen_hits[code] = {
            "found": True,
            "locus": entry["locus"],
            "strategy": strategy,
            "gene_name": gene_name,
            "product": product,
            "match_wp": prov.get("match_wp", ""),
            "match_accession": prov.get("match_accession", ""),
            "match_region": prov.get("match_region", ""),
            "pct_identity": prov.get("pct_identity", ""),
            "aln_len": prov.get("aln_len", ""),
        }

        max_fc = max(
            (v for k, v in entry.items() if k.endswith("_fc") and isinstance(v, float)),
            default=None,
        )
        fc_str = f"{max_fc:.2f}" if max_fc is not None else "N/A"
        pid = entry.get("pct_identity", "")
        id_str = f"{pid:.0f}%" if isinstance(pid, (int, float)) else "n/a"
        print(
            f"  [{code:<10}] {entry['locus']:<25} via {strategy:<20} "
            f"ctrl={entry['ctrl_tpm']:.1f} TPM  id={id_str:<5} max_FC={fc_str}"
        )
        records.append(entry)

    #  Step 3: Write the frozen reproducibility artifact
    frozen_meta = {
        "ref_protein": args.protein,
        "ref_wp": wp_ref or "",
        "secondary_proteins": list(args.secondary_proteins),
        "overrides": locus_overrides,
        "no_scan": bool(args.no_scan),
        "query_date": query_date,
        "geo_dataset": "GSE152295",
        "n_found": int(sum(1 for h in frozen_hits.values() if h.get("found"))),
        "n_total": len(frozen_hits),
    }
    write_frozen(Path(args.frozen_out), frozen_meta, frozen_hits)
    print()
    print(f"Frozen artifact written to: {args.frozen_out}")

    #  Step 4: Write CSV
    n_found, n_total = write_output_csv(records, cond_cols_seen, out_path)
    print(f"Found in {n_found}/{n_total} species.")
    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
