---
name: ""
overview: ""
todos: []
isProject: false
---

# ParCa CYS mechanistic supply failure (new RNaseq ingestion)

## Context

- **Data change:** New RNaseq data ingestion (manifest + `gbw_0001_v2`, basal condition `"M9 Glucose minus AAs"`) is used for ParCa.
- **Symptom:** ParCa fails in **stage 9 (final_adjustments)** at `set_mechanistic_supply_constants` with:
  - `ValueError: Could not find positive forward and reverse kcat for CYS[c]. Run with VERBOSE to check input parameters like KM and KI or check concentrations.`
- **Reference behaviour:** ParCa **completes** when run with the **default/reference** config (no `rnaseq_manifest_path`), so the failure is tied to the new RNaseq path.

## How mechanistic supply fitting works (relevant part)

- **Two conditions:** The code fits one set of AA supply parameters to satisfy **both**:
  - **Minimal:** `sim_data.translation_supply_rate["minimal"]` (from ParCa’s basal fit, which **is** driven by the new RNaseq for “M9 Glucose minus AAs”).
  - **With AAs:** `sim_data.translation_supply_rate["minimal_plus_amino_acids"]` (from the “with_aa” cell spec). Its **definition** (media, TF sets, doubling time) is fixed in `condition_defs.tsv`, but its **expression and translation_aa_supply** are still derived from the same basal expression that was fit (reference vs new RNaseq).
- **Fixed inputs (unchanged by RNaseq):**
  - `amino_acid_uptake_rates.tsv` (e.g. CYS uptake 0.0017 mmol/g/h from Zampieri et al.).
  - `amino_acid_pathways.tsv` (KMs, KI, reverse/degradation).
  - Metabolite concentrations for minimal vs minimal_plus_amino_acids.
- **Acceptance rule:** The solver only accepts a solution when **both** `kcat_fwd >= 0` and `kcat_rev >= 0`.

## What was checked

1. **VERBOSE run** with the new RNaseq config produced a full CYS log (e.g. in `parca_verbose.log`).
2. **Bug fix (during investigation):** When VERBOSE was on and no valid kcats were found, a line `print(f"*** {amino_acid}: {kcat_fwd:5.1f} ...")` was still executed with `kcat_fwd is None`, causing a `TypeError`. That was fixed by only printing that line when `kcat_fwd is not None`. (Status of this fix in current codebase not confirmed.)
3. **Per-AA header:** A VERBOSE line `--- CYS[c] ---` was added at the start of each AA block so CYS (and others) are easy to find in large logs. (Status of this in current codebase not confirmed.)

## What the CYS VERBOSE log showed

- **Uptake scan (0.1×–10×):** For every factor, `kcat_fwd ≈ 1829.9`, `kcat_rev ≈ -3456.4`. Result does not depend on uptake scaling.
- **KMs, km_degradation, ki:** Either same (1829.9 / -3456.4) or kcat_fwd varies while kcat_rev stays around -3450 to -3457. **No** trial had `kcat_rev >= 0`.
- **km_reverse scan:** For larger `km_reverse` (e.g. factor ≥ ~0.83), kcat_fwd becomes positive (e.g. 15.4 → 1829.9 at 1.0) but kcat_rev remains negative (e.g. -2979.7 → -3456.4). So **no** (kcat_fwd, kcat_rev) pair with both ≥ 0 exists in any scan.

**Conclusion from the log:** The two-condition balance (basal vs with_AA supply + uptake + kinetics) **always** implies a **negative reverse kcat** for CYS. The forward kcat can be positive; the hard constraint is the reverse. So the failure is a **consistency** issue between:

- New RNaseq-driven **minimal** translation supply (CYS),
- Legacy **minimal_plus_amino_acids** supply (CYS),
- Fixed CYS **uptake** and **pathway kinetics**,

not a simple bug or a single wrong number.

## Why “M9 glucose minus AAs” can still cause this

- The new dataset describes **only** the minimal (no-AAs) condition; there is no Cys (or other AA) uptake in that condition.
- `set_mechanistic_supply_constants` still fits **both** minimal and with_AA and uses the **same** `amino_acid_uptake_rates.tsv` (including CYS 0.0017) for the with_AA side.
- So: **new** minimal CYS demand (from new RNaseq) is combined with **unchanged** with_AA demand and **unchanged** CYS uptake/kinetics. If that combination is inconsistent (e.g. the implied reverse flux is negative), the solver never finds both kcats ≥ 0. Hence the hypothesis: **mismatch between new minimal (RNaseq) and legacy with_AA (reference) is what makes CYS fail**, even though the new data itself has no AA uptake.

## What was proposed but rejected (not in codebase)

During investigation, the following were **implemented in a prior session** but the user **rejected** them; they are **not** currently in the codebase (or have been reverted):

- **Fallback in metabolism:** When no (kcat_fwd, kcat_rev) with both ≥ 0 is found, use the nominal (factor=1) `calc_kcats` result: keep kcat_fwd and set **kcat_rev = 0**, **rev_rate = 0**, and emit a **UserWarning**, so ParCa can complete with a forward-only mechanistic supply for that AA (e.g. CYS).
- **VERBOSE** was set back to `False` after debugging (may or may not still be False).

The user chose not to adopt this fallback and wants to **understand why** the inconsistency arises before choosing a course of action.

## Possible next steps (for understanding or fixing)

1. **Compare CYS supply numbers (DONE, via notebook):** We now have `wholecell/io/comparative_simdata_demo.py`, a marimo notebook that:
  - Loads `simData.cPickle` from both reference and RNaseq ParCa runs (or the RNaseq `sim_data_fit_condition.cPickle` intermediate when final ParCa fails).
  - Extracts `sim_data.translation_supply_rate["minimal"]` and `["minimal_plus_amino_acids"]` for all AAs and plots tables/ratios; for CYS, the RNaseq run shows only ~7–8% higher supply than reference.
2. **Trace where with_AA comes from (DONE):** The notebook and code review confirm:
  - `condition_defs.tsv` defines `"with_aa"` (nutrients `minimal_plus_amino_acids`, TF sets, doubling time 25 min).
  - `buildCombinedConditionCellSpecifications` builds the `"with_aa"` cell spec by applying TF fold changes on **basal** expression and running `expressionConverge`.
  - `fit_condition` then sets `translation_supply_rate["minimal_plus_amino_acids"]` from that `"with_aa"` spec’s `translation_aa_supply`.
  - Thus when we switch to the new RNaseq, **both** `["minimal"]` and `["minimal_plus_amino_acids"]` are ultimately driven by the new basal expression; the “legacy vs new” mismatch is in condition definition + fixed uptake/kinetics, not in mixing two entirely different expression sources.
3. **Data-side options:** Adjust CYS in `amino_acid_uptake_rates.tsv` or in the CYS pathway (e.g. `amino_acid_pathways.tsv`) so the two-condition balance is consistent; or constrain/cap CYS translation demand when using the new RNaseq.
4. **Model-side options:** Revisit a fallback (accept kcat_rev=0 and warn) for AAs where the solver never finds both kcats ≥ 0; or relax the “both kcats ≥ 0” rule for specific AAs (e.g. CYS) or pathway types (e.g. when reverse is effectively absent).
5. **QC for new datasets:** Add a ParCa “diagnostic” or QC step that runs the mechanistic supply solver and reports which AAs fail or need extreme parameters, to catch similar mismatches for other datasets.

## New diagnostic capabilities (current status)

- **Side-by-side sim_data comparison:** `wholecell/io/comparative_simdata_demo.py` now:
  - Loads reference and new-RNaseq `sim_data` objects.
  - Shows RNaseq config metadata (rnaseq manifest path, dataset id).
  - Compares condition-to-doubling-time mappings.
  - Compares translation AA supply (`minimal` and `minimal_plus_amino_acids`) across all AAs, with ratios and CYS highlighted, in convention units mmol/g/h.
- **CYS-pathway transcript inspection:**
  - For CYS, we use `metabolism.aa_synthesis_pathways["CYS[c]"]` to get forward/reverse enzymes, then `process.complexation.get_monomers` to expand complexes into monomer subunits.
  - We map those monomers to cistrons and RNAs using `relation.cistron_to_monomer_mapping` and `cistron_id_to_rna_indexes`.
  - The notebook builds a `cys_rna_expr` table of all RNAs encoding CYS-pathway enzymes, with:
    - Reference vs RNaseq expression in basal and with_aa.
    - Log2 fold-changes.
    - Annotation of which enzyme(s) and whether each RNA contributes to forward and/or reverse CYS flux.
- **CYS-pathway enzyme abundance (bulk counts):**
  - Using ParCa intermediates (`cell_specs_fit_condition.cPickle`) from both runs, we can load `cell_specs["basal"]["bulkContainer"]` and `["with_aa"]["bulkContainer"]`.
  - For each CYS enzyme (forward and reverse), we sum counts of:
    - The complex species itself (if present in bulk),
    - All monomer subunits associated with that enzyme/direction.
  - This yields a `cys_enzyme_counts` table comparing basal vs with_aa, reference vs RNaseq, at the level that actually enters the kcat balance (`fwd_capacity` / `rev_capacity`).

These diagnostics let us look, side by side, at:

- CYS supply (minimal and minimal_plus_amino_acids) in mmol/g/h.
- Expression shifts for all CYS-pathway RNAs.
- Changes in effective forward and reverse enzyme capacities (bulk counts) in basal and with_aa.

The next conceptual step is to interpret these together to see exactly why the 7–8% higher CYS demand plus the specific enzyme and uptake configuration forces `kcat_rev < 0` in the two-condition fit.

## Key files

- **Error and solver:** `reconstruction/ecoli/dataclasses/process/metabolism.py` — `set_mechanistic_supply_constants`, `calc_kcats`, VERBOSE scans; any fallback would go here.
- **ParCa flow:** `reconstruction/ecoli/fit_sim_data_1.py` — `final_adjustments` (calls `set_mechanistic_supply_constants`), `fit_condition` (fills `translation_supply_rate`).
- **Fixed inputs:** `reconstruction/ecoli/flat/amino_acid_uptake_rates.tsv`, `reconstruction/ecoli/flat/amino_acid_pathways.tsv`.
- **Configs:** `configs/test_rnaseq_ingestion.json` (new RNaseq, fails), `configs/test_rnaseq_ingestion_defaults.json` (reference, completes).

