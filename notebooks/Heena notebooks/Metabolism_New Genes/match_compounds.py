"""Map EcoCyc phenotype-microarray (PM) well compounds to this model's
internal exchange-species IDs.

Reads `phenotypic_array_wells.json` (383 EcoCyc PM wells) and
`reconstruction/ecoli/flat/metabolites.tsv` (this model's metabolite
universe), normalizes compound names on both sides, and looks each well's
compound up against `metabolism.metabolite_names` (all species with a row in
the model's stoichiometric matrix, ~6100 entries) from a loaded simulation
checkpoint. Writes one row per well to `compound_mapping.csv` for human
review; that file is consumed by the sibling script `run_phenotypic_arrays.py`
(see its `build_conditions`).

Deliberately NOT validated against `metabolism.exchange_molecules` /
`allowed_exchange_uptake`: those two attributes only reflect which exchanges
happened to be active during whichever specific media history the loaded
checkpoint's simulation ran (52 and 19 entries respectively for the basal
glucose-minimal-media checkpoint used here) -- not what the model is capable
of. The whole point of a PM well is to introduce an exchange that ISN'T
currently active, so checking against the currently-active set would reject
almost everything (verified: this was the original bug in this script, which
found only 14/383 matches; `metabolite_names` is the correct universe and
matches the old hand-curated notebooks' ~85-90% usable rate). Actually
enabling a newly-matched species' exchange reaction at FBA-run time is
`run_phenotypic_arrays.py`'s job (via `new_exchange_molecules`), not this
script's.

`load_sim` below is a local copy of the identically named function in
`run_phenotypic_arrays.py` (itself ported from `Standalone_FBA.ipynb`), kept
duplicated here rather than imported to avoid dealing with importing a
module from a path containing spaces.
"""

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO_ROOT = Path(__file__).resolve().parents[3]
NB_DIR = Path(__file__).resolve().parent

WELLS_JSON = NB_DIR / "phenotypic_array_wells.json"
METABOLITES_TSV = REPO_ROOT / "reconstruction/ecoli/flat/metabolites.tsv"
OUT_CSV = NB_DIR / "compound_mapping.csv"
SIM_FOLDER = REPO_ROOT / "out/objective_weight/basal/baseline_10_2026-05-18/"

# Exact (post-normalization) compound_name strings that mark a well as a
# negative control rather than an actual compound to match. Always well A1
# except PM4's sulfur negative control, which is at F1 (handled naturally
# here since we match on compound_name text, not well position).
NEGATIVE_CONTROLS = {
    "carbon negative control",
    "nitrogen negative control",
    "phosphorus negative control",
    "sulfur negative control",
}

# Stereo/prefix tokens EcoCyc and this model's metabolites.tsv don't always
# agree on formatting-wise; stripped as a fallback pass only if the exact
# normalized name doesn't match anything.
STEREO_PREFIXES = {"d", "l", "dl", "alpha", "beta", "gamma", "n"}

# Compartment tags to try, in the order used for reporting.
COMPARTMENT_TAGS = ("c", "e", "p")

TAG_RE = re.compile(r"<[^>]+>")
NONALNUM_RE = re.compile(r"[^a-z0-9]+")

# metabolites.tsv spells Greek letters as HTML entities (e.g. "&alpha;"),
# which html.unescape() turns into the literal Unicode Greek character (e.g.
# "α") -- that character then gets silently stripped out by NONALNUM_RE,
# losing the alpha/beta/gamma distinction entirely (e.g. both "&alpha;-D-
# glucose" and "&beta;-D-glucose" would otherwise collapse to the same
# normalized "d glucose"). The well JSON's compound_name field instead
# spells these out as plain English words ("alpha-Ketoglutaric acid"), so
# metabolites.tsv's entities are rewritten to match that convention *before*
# falling through to html.unescape for everything else (arrows, primes,
# etc., which carry no matching signal and are fine to just strip).
GREEK_ENTITY_RE = re.compile(r"&([A-Za-z]+);")
GREEK_ENTITY_WORDS = {
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "omega",
    "tau",
    "pi",
    "kappa",
    "theta",
    "psi",
    "nu",
    "lambda",
    "iota",
    "xi",
    "phi",
    "mu",
    "chi",
    "sigma",
    "rho",
    "eta",
    "zeta",
    "upsilon",
}


def _spell_out_greek_entities(text):
    def repl(match):
        name = match.group(1).lower()
        if name in GREEK_ENTITY_WORDS:
            return f" {name} "
        return match.group(0)

    return GREEK_ENTITY_RE.sub(repl, text)


def normalize(text):
    """Decode HTML entities, strip tags, lowercase, and collapse punctuation
    and whitespace to single spaces."""
    if text is None:
        return ""
    text = _spell_out_greek_entities(str(text))
    text = html.unescape(text)
    text = TAG_RE.sub("", text)
    text = text.lower()
    text = NONALNUM_RE.sub(" ", text)
    return " ".join(text.split())


def strip_stereo_prefix(normalized):
    """Drop a single leading stereo/prefix token (e.g. "l arabinose" ->
    "arabinose"), used only as a fallback when the exact normalized name has
    no match."""
    tokens = normalized.split(" ")
    while len(tokens) > 1 and tokens[0] in STEREO_PREFIXES:
        tokens = tokens[1:]
    return " ".join(tokens)


def load_metabolite_lookup(tsv_path):
    """Build normalized-alias -> {metabolite ids} from metabolites.tsv.

    Each row's `id`, `common_name`, and every entry in `synonyms` are
    normalized and used as aliases pointing back to that row's `id`. If two
    different ids produce the same normalized alias, both are kept in the
    alias's id set (resolved as ambiguous later) rather than picking one.
    """
    lookup = {}
    with open(tsv_path) as f:
        # First 5 lines are '#'-prefixed metadata comments; header is line 6.
        for _ in range(5):
            next(f)
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            mid = row["id"]
            aliases = {normalize(mid), normalize(row["common_name"])}
            try:
                synonyms = json.loads(row["synonyms"])
            except (json.JSONDecodeError, TypeError):
                synonyms = []
            for syn in synonyms:
                aliases.add(normalize(syn))
            aliases.discard("")
            for alias in aliases:
                lookup.setdefault(alias, set()).add(mid)
    return lookup


def find_candidates(compound_name, lookup):
    """Return the set of metabolites.tsv ids matching a well's compound_name,
    trying the exact normalized name first and falling back to a
    stereo-prefix-stripped version only if the exact pass finds nothing."""
    norm = normalize(compound_name)
    candidates = lookup.get(norm)
    if candidates:
        return set(candidates)

    stripped = strip_stereo_prefix(norm)
    if stripped != norm:
        candidates = lookup.get(stripped)
        if candidates:
            return set(candidates)

    return set()


# Old hand-curated notebooks with per-well Add-species choices, used as a
# cross-reference for wells the automated matcher leaves ambiguous/unmatched
# (see `load_old_notebook_lookup`). Cell indices point at the raw
# `conditions = {...}` dict for each plate (verified by inspection, not
# guessed -- other cells in the same notebooks contain unrelated dicts).
OLD_NOTEBOOKS = {
    "PM1": (NB_DIR / "20250616_test_carbon_source_cp3.ipynb", 13),
    "PM2": (
        NB_DIR.parent / "6-27-25 Implementing MicroArray 2 (carbon source).ipynb",
        12,
    ),
    "PM3": (NB_DIR.parent / "7-24-25 EasyGenes34micros.ipynb", 12),
    "PM4": (NB_DIR.parent / "7-24-25 EasyGenes34micros.ipynb", 15),
}

WELL_KEY_RE = re.compile(r"^\s*([A-H])\s*(\d{1,2})\b")
CONDITIONS_ENTRY_RE = re.compile(
    r"'([^']*)':\s*(None|\{\s*'Add':\s*set\(\[(.*?)\]\))", re.DOTALL
)


def load_old_notebook_lookup():
    """Parse the 4 old hand-curated notebooks' `conditions` dicts into
    {(plate, "A2"): set-of-Add-species-or-None}, for cross-referencing wells
    the automated matcher can't resolve on its own. `None` means the old
    notebook explicitly excluded that well ("not in model"); a present-but-
    empty set means an intentional no-addition well (negative controls,
    handled separately upstream of this lookup)."""
    lookup = {}
    for plate, (nb_path, cell_idx) in OLD_NOTEBOOKS.items():
        with open(nb_path) as f:
            nb = json.load(f)
        src = "".join(nb["cells"][cell_idx]["source"])
        for m in CONDITIONS_ENTRY_RE.finditer(src):
            key, whole, inner = m.groups()
            well_match = WELL_KEY_RE.match(key)
            if not well_match:
                continue
            well = f"{well_match.group(1)}{int(well_match.group(2))}"
            if whole == "None":
                lookup[(plate, well)] = None
            else:
                lookup[(plate, well)] = set(re.findall(r"'([^']+)'", inner or ""))
    return lookup


def cross_reference_old_notebook(row, old_lookup, valid_species_set):
    """For a row still ambiguous/unmatched after automated matching, check
    whether the old hand-curated notebooks already resolved this exact well,
    and validate that choice still exists in the current model before
    trusting it (species can be renamed/removed between model versions).

    Mutates `row` in place (matched_species_id/compartment/match_status/notes)
    when a validated old-notebook answer is found; otherwise only appends an
    explanatory note.
    """
    key = (row["plate"], row["well"])
    if key not in old_lookup:
        return
    old_value = old_lookup[key]

    if old_value is None:
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + (
            "old notebook also excluded this well (not in model)"
        )
        return

    if not old_value:
        # Old notebook had an empty Add set for a non-control well; nothing
        # to cross-reference.
        return

    missing = [sp for sp in old_value if sp not in valid_species_set]
    if missing:
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + (
            f"old notebook proposed {sorted(old_value)} but "
            f"{sorted(missing)} not found in current model metabolite_names "
            "-- needs re-mapping"
        )
        return

    ordered = sorted(old_value)
    if len(ordered) == 1:
        species_full = ordered[0]
        cm = re.match(r"^(.*)\[(\w+)\]$", species_full)
        row["matched_species_id"] = cm.group(1) if cm else species_full
        row["compartment"] = cm.group(2) if cm else ""
    else:
        # Rare (1/383): old notebook added more than one species for this
        # well (e.g. both D- and L- forms of a racemic compound). Store all
        # of them pipe-joined, already compartment-tagged; build_conditions
        # in run_phenotypic_arrays.py splits on "|" for this case.
        row["matched_species_id"] = "|".join(ordered)
        row["compartment"] = ""
    row["match_status"] = "matched_from_notebook"
    row["notes"] = (row["notes"] + "; " if row["notes"] else "") + (
        "sourced from old hand-curated notebook, validated against current "
        "model -- spot-check recommended"
    )


def load_sim(folder_path):
    """Load an output of a simulation in timeseries form.

    Note: This is not designed for parquet output format. Local copy of
    `run_phenotypic_arrays.load_sim` (ported from Standalone_FBA.ipynb).
    """
    output = np.load(folder_path + "0_output.npy", allow_pickle="TRUE").item()
    output = output["agents"]["0"]
    fba = output["listeners"]["fba_results"]
    bulk = pd.DataFrame(output["bulk"])
    with open(folder_path + "agent_steps.pkl", "rb") as f:
        agent = dill.load(f)

    metabolism = agent["ecoli-metabolism-redux-classic"]

    return fba, bulk, metabolism, output


def well_label(well_row, well_col):
    """Combine well_row/well_col into the label format run_phenotypic_arrays'
    parse_well/build_conditions expect, e.g. ("A", 2) -> "A2"."""
    return f"{well_row}{int(well_col)}"


def build_reaction_index(metabolism):
    """Map species id (e.g. 'SUC[e]') -> list of reaction names touching it,
    from the model's stoichiometric matrix. Used only to build a human-
    readable hint for genuinely ambiguous [e]-vs-[p] calls (see
    `resolve_species`) -- NOT used to auto-resolve them, since a 5-compound
    spot check against the old hand-curated notebooks showed a "prefer the
    compartment with fewer/transport-only reactions" heuristic gets 4/5
    right and silently wrong on the 5th (ammonia) -- not good enough odds to
    guess a nutrient's exchange compartment automatically.
    """
    name_idx = {name: i for i, name in enumerate(metabolism.metabolite_names)}
    S = metabolism.stoichiometry
    if sp.issparse(S):
        S = S.tocsr()
    reaction_names = metabolism.reaction_names

    def reactions_for(species_id):
        idx = name_idx.get(species_id)
        if idx is None:
            return []
        row = S[idx]
        nz = row.nonzero()[1] if sp.issparse(row) else np.nonzero(row)[0]
        return [reaction_names[j] for j in nz]

    return reactions_for


def resolve_species(candidate_ids, valid_species_set, reactions_for=None):
    """Resolve a set of candidate metabolites.tsv ids to a single species,
    given the model's full metabolite universe (already compartment-tagged,
    e.g. 'GLC[p]').

    Returns (matched_species_id, compartment, match_status, note).
    """
    if not candidate_ids:
        return "", "", "unmatched", ""

    per_id_hits = {}
    for cid in candidate_ids:
        hits = [tag for tag in COMPARTMENT_TAGS if f"{cid}[{tag}]" in valid_species_set]
        if hits:
            per_id_hits[cid] = hits

    if not per_id_hits:
        # Candidate(s) matched a metabolites.tsv id, but that id has no row
        # in this model's stoichiometric matrix under any compartment tag.
        return "", "", "unmatched", ""

    if len(per_id_hits) > 1:
        # More than one distinct metabolite id resolves to something valid
        # (a genuine synonym collision) -- can't pick between different
        # compounds without guessing.
        ids = ", ".join(sorted(per_id_hits))
        return "", "", "ambiguous", f"multiple candidate ids: {ids}"

    ((cid, hits),) = per_id_hits.items()

    # Nearly every metabolite has [c]/[e]/[p] rows in metabolite_names just
    # from general compartmentalization bookkeeping, so a single candidate id
    # usually comes back with 2-3 "valid" compartments. Exchange/boundary
    # species are extracellular or periplasmic in this model, never
    # cytoplasmic -- verified every Add species in the old hand-curated PM
    # notebooks uses [e] or [p], never [c] -- so prefer those over a
    # cytoplasm-only hit rather than treating this as ambiguous.
    boundary_hits = [tag for tag in hits if tag != "c"]
    if len(boundary_hits) == 1:
        return cid, boundary_hits[0], "matched", ""
    if len(boundary_hits) > 1:
        # Both [e] and [p] rows exist; telling which one this model's
        # reaction network actually treats as the exchange boundary for this
        # specific compound needs a human to look at its reactions -- surface
        # a reaction-count hint per compartment rather than guessing.
        note = ""
        if reactions_for is not None:
            parts = []
            for tag in boundary_hits:
                rxns = reactions_for(f"{cid}[{tag}]")
                sample = ", ".join(rxns[:2])
                parts.append(f"{tag}:{len(rxns)} rxns ({sample})")
            note = "; ".join(parts)
        return "", "", "ambiguous", note
    # Only a [c] (cytoplasm-only) row exists -- no plausible exchange form.
    return "", "", "unmatched", ""


def main():
    with open(WELLS_JSON) as f:
        wells = json.load(f)

    print(f"Loaded {len(wells)} wells from {WELLS_JSON}")

    lookup = load_metabolite_lookup(METABOLITES_TSV)
    print(f"Built alias lookup with {len(lookup)} normalized aliases")

    sim_folder = str(SIM_FOLDER)
    if not sim_folder.endswith("/"):
        sim_folder += "/"
    print(f"Loading simulation checkpoint from {sim_folder} (this takes a while)...")
    _, _, metabolism, _ = load_sim(sim_folder)
    exchange_set = set(metabolism.metabolite_names)
    print(f"Loaded {len(exchange_set)} species ids in the model's metabolite universe")
    reactions_for = build_reaction_index(metabolism)

    rows = []
    is_control = []
    for well in wells:
        plate = well["plate"]
        well_row = well["well_row"]
        well_col = well["well_col"]
        compound_name = well["compound_name"]
        mix0_id = well.get("mix0_id", "")
        label = well_label(well_row, well_col)

        if normalize(compound_name) in NEGATIVE_CONTROLS:
            rows.append(
                {
                    "plate": plate,
                    "well": label,
                    "compound_name": compound_name,
                    "mix0_id": mix0_id,
                    "matched_species_id": "",
                    "compartment": "",
                    "match_status": "matched",
                    "notes": "",
                }
            )
            is_control.append(True)
            continue

        candidates = find_candidates(compound_name, lookup)
        species_id, compartment, status, note = resolve_species(
            candidates, exchange_set, reactions_for
        )

        rows.append(
            {
                "plate": plate,
                "well": label,
                "compound_name": compound_name,
                "mix0_id": mix0_id,
                "matched_species_id": species_id,
                "compartment": compartment,
                "match_status": status,
                "notes": note,
            }
        )
        is_control.append(False)

    # --- Cross-reference remaining ambiguous/unmatched rows against the old
    # hand-curated notebooks (see cross_reference_old_notebook) ---
    old_lookup = load_old_notebook_lookup()
    n_from_notebook = 0
    for row in rows:
        if row["match_status"] not in ("ambiguous", "unmatched"):
            continue
        before = row["match_status"]
        cross_reference_old_notebook(row, old_lookup, exchange_set)
        if row["match_status"] != before:
            n_from_notebook += 1
    print(f"Cross-referenced old notebooks: {n_from_notebook} wells resolved")

    fieldnames = [
        "plate",
        "well",
        "compound_name",
        "mix0_id",
        "matched_species_id",
        "compartment",
        "match_status",
        "notes",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_CSV}")

    # --- Summary ---
    n_control = sum(is_control)
    n_matched_noncontrol = sum(
        1 for r, c in zip(rows, is_control) if r["match_status"] == "matched" and not c
    )
    status_counts = Counter(r["match_status"] for r in rows)

    print("\n--- Summary ---")
    print(f"Total wells: {len(rows)}")
    print(f"  matched (non-control, automated): {n_matched_noncontrol}")
    print(
        f"  matched_from_notebook (cross-referenced): {status_counts.get('matched_from_notebook', 0)}"
    )
    print(f"  unmatched: {status_counts.get('unmatched', 0)}")
    print(f"  ambiguous: {status_counts.get('ambiguous', 0)}")
    print(f"  control: {n_control}")
    assert n_matched_noncontrol + status_counts.get(
        "matched_from_notebook", 0
    ) + status_counts.get("unmatched", 0) + status_counts.get(
        "ambiguous", 0
    ) + n_control == len(rows)

    # --- Flag wells whose matched compartment species coincides with that
    # plate's default Remove set, since Add/Remove would cancel out. ---
    default_removals = {
        "PM1": {"GLC[p]", "CA+2[p]"},
        "PM2": {"GLC[p]", "CA+2[p]"},
        "PM3": {"AMMONIUM[c]", "CA+2[p]"},
    }

    def pm4_removal(well_label_str):
        row_letter = well_label_str[0]
        if row_letter in "ABCDE":
            return {"Pi[p]", "CA+2[p]"}
        return {"SULFATE[p]", "CA+2[p]"}

    print("\n--- Wells whose match coincides with the default Remove set ---")
    n_collisions = 0
    for r in rows:
        if r["match_status"] != "matched" or not r["matched_species_id"]:
            continue
        species_tag = f"{r['matched_species_id']}[{r['compartment']}]"
        plate = r["plate"]
        removals = (
            pm4_removal(r["well"])
            if plate == "PM4"
            else default_removals.get(plate, set())
        )
        if species_tag in removals:
            n_collisions += 1
            print(
                f"  {plate} {r['well']} {r['compound_name']!r} -> {species_tag} "
                f"(collides with {plate} Remove set)"
            )
    if n_collisions == 0:
        print("  none found")


if __name__ == "__main__":
    main()
