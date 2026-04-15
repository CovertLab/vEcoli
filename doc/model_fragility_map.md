# Model Fragility Map

## Purpose and usage

This doc enumerates the parts of the vEcoli model whose correctness depends on
properties of the RNA-seq input dataset. Its target reader is a research
engineer running sensitivity experiments who wants to know which subsystem to
investigate when a perturbed dataset produces a specific failure signature.
Deliverable #2 from `.claude/plans/dataset-sensitivity-exploration.md`; pairs
with `.claude/plans/rnaseq-dataset-failure-analysis.md`.

Each entry has three parts:
- **(a) Implicit dependence** — which statistical property of the TPM vector,
  or which named gene subset, this subsystem assumes.
- **(b) Failure mode** — named exception, silent drift, assertion, or
  convergence failure that surfaces when (a) is violated.
- **(c) Observable signature** — the thing you grep for in logs, parca output,
  or simulated trajectories to confirm this subsystem is the culprit.

Reading guide: if your sensitivity run produced `ValueError: Could not find
positive forward and reverse kcat for ...`, start at subsystem 2. If ParCa
reached the P-solve and then died with "Solver could not find optimal value",
jump to subsystem 3. If ParCa succeeded but the sim dies with
`NegativeCountsError`, go to subsystem 5. If everything runs but mass drifts
across generations, subsystems 1, 8, and 9 are the prime suspects.

All file:line references verified against the current tree; paths are relative
to the repo root unless absolute.

---

## 1. ppGpp regulation

Anchors: `reconstruction/ecoli/dataclasses/process/transcription.py:80`
(`_build_ppgpp_regulation`), `:1803` (`_solve_ppgpp_km`), `:1945`
(`set_ppgpp_expression`), `:1999` (`adjust_polymerizing_ppgpp_expression`),
`:2087` (`adjust_ppgpp_expression_for_tfs`). Invoked from
`reconstruction/ecoli/fit_sim_data_1.py:253, :495, :496`.

(a) The ppGpp subsystem splits TU expression into a "RNAP-bound-to-ppGpp"
component and a "free-RNAP" component using a per-cistron fold change vector
`ppgpp_fold_changes` (transcription.py:180). The fold changes are sourced from
a curated table (`raw_data.ppgpp_fc`, transcription.py:137) — they are not
recomputed from the input RNA-seq — but the *expression baselines* to which
those fold changes are applied come directly from the input TPM through
`fit_cistron_expression["basal"]` (transcription.py:1971). The least-squares
split between free and bound expression (`adjust_polymerizing_ppgpp_expression`,
transcription.py:2062-2082) is fit on exactly **three** conditions — hard-coded
`["with_aa", "basal", "no_oxygen"]` — and only applied to the
`rna_data["includes_RNAP"] | rna_data["includes_ribosomal_protein"] |
rna_data["is_rRNA"]` mask (transcription.py:2042-2046). The fit also assumes
`0 < fraction_active_rnap_bound < 1` and `0 < fraction_active_rnap_free < 1`
(transcription.py:229-230).

(b) Two expected modes. (1) **Silent drift across generations**: the basal
expression comes from the input TPM but the fold-change curation is fixed,
so on off-reference datasets the ratio of RNAP-bound to free expression on
polymerizing genes will land somewhere the 3-condition LS fit wasn't designed
to handle, producing a valid but subtly biased `exp_ppgpp`/`exp_free` split.
The downstream effect is growth-rate-dependent RNAP/ribosome scaling that
drifts from target mass fractions over generations. (2) **Assertion failure**:
if the input dataset shifts ribosomal/RNAP expression enough that the
basal/with_aa/anaerobic total ribosome+RNAP mass fractions move outside the
bound-vs-free interpolation, the `assert 0 < self.fraction_active_rnap_bound <
1` / `< self.fraction_active_rnap_free < 1` at transcription.py:229-230 fires
during ParCa.

(c) Grep ParCa stderr for `AssertionError` near `transcription.py:229`. In sim
trajectories, look at the ppGpp concentration listener and at
`transcription.exp_ppgpp`/`exp_free` diff vs. the reference baseline; divergence
in stable-RNA mass fraction across generations is the silent-drift signature.
The supplement log lines printed when `PRINT_VALUES=True` (transcription.py:206-217)
report KM, FC bounds, FC- and FC+ — anomalous values there are an early warning.

---

## 2. Amino-acid kcat fitting

Anchors: `reconstruction/ecoli/dataclasses/process/metabolism.py:1415-1779`,
specifically the `calc_kcats` closure at `:1516`, the logspace search at
`:1664` (500 factors from 0.1 to 10), and the terminal
`raise ValueError("Could not find positive forward and reverse kcat for ...")`
at `:1761-1766`. Pathway data from `flat/amino_acid_pathways.tsv` loaded at
`metabolism.py:801-835`; pathway-specific parameter overrides in
`flat/adjustments/amino_acid_pathways.tsv` loaded at `:837-847`.

(a) For each amino acid, the solver needs `fwd_enzymes_{basal,with_aa}` and
`rev_enzymes_{basal,with_aa}` expressed at levels that let a 2x2 linear
solve for `(kcat_fwd, kcat_rev)` yield both non-negative. The enzyme counts
are computed in `buildBasalCellSpecifications` from `rna_expression["basal"]`
via ribosome-rate-weighted protein expression (fit_sim_data_1.py:578+), so
low-TPM biosynthesis gene sets propagate directly into near-zero fwd/rev
capacities. The solver scans uptake factors from 0.1x to 10x measured, and
accepts the minimum-objective point where both kcats are strictly non-negative
(metabolism.py:1664-1692). Pathway-specific fragility is highest where the
forward/reverse split is near-degenerate (e.g. single reverse enzyme `tnaA`
appears in both CYS and TRP, cf. `.claude/plans/rnaseq-dataset-failure-analysis.md:54,62`).

(b) **Named exception**: `ValueError: Could not find positive forward and
reverse kcat for {AA}[c]. Run with VERBOSE to check input parameters like KM
and KI or check concentrations.` (metabolism.py:1761-1766). Confirmed failing
AAs in the existing campaign: CYS (gbw_vegas_wt_m9glc_*), TRP
(precise_abx_media:m9_ctrl), ILE (precise_ytf5:wt_glc). Silent-drift mode is
possible if the uptake factor lands at the edge of 0.1 or 10 but both kcats
are still non-negative — then `measured_uptake_rates` gets multiplied by that
extremum and supply/demand balance is off.

(c) Grep ParCa stderr for `Could not find positive forward and reverse kcat`;
the line immediately identifies the failing AA. With `VERBOSE=1` (default,
fit_sim_data_1.py:45), the iteration table of `factor / kcat_fwd / kcat_rev`
prints to stdout (metabolism.py:1678-1679) — if all rows show one kcat
negative, the enzyme ratio is the problem. Cross-reference with the gene
lists in `.claude/plans/rnaseq-dataset-failure-analysis.md` for which TPM
values to inspect in the input dataset.

---

## 3. P-solve (`fitPromoterBoundProbability`)

Anchors: `reconstruction/ecoli/fit_sim_data_1.py:2912` (function entry),
`:3624` (pdiff matrix build at each iteration), `:3648-3667` (objective +
constraints), `:3678-3679` (`raise RuntimeError("Solver could not find optimal
value")`). Matrix constructor at `:3425-3466`. Hyperparameters at lines 37-43:
`PROMOTER_PDIFF_THRESHOLD = 0.06`, `PROMOTER_REG_COEFF = 1e-3`,
`PROMOTER_SCALING = 10`, `PROMOTER_NORM_TYPE = 1`, `PROMOTER_MAX_ITERATIONS = 100`.

(a) Fits a TF-promoter-binding probability vector `P` such that `H @ P ≈ k`,
where `k` is the per-gene-copy RNA synthesis probability derived from the
input TPM (fit_sim_data_1.py:3004-3011), under the hard linear constraints
`0 ≤ P ≤ 1`, `D @ P == Drhs` (fixed TFs/alphas), and
`pdiff @ P >= PROMOTER_PDIFF_THRESHOLD`
(fit_sim_data_1.py:3661-3667). `pdiff` is a `n_TF x n_columns` matrix with +1
for `TF__active` and -1 for `TF__inactive`; the constraint says every TF must
show at least a 0.06 probability gap between its active and inactive regulated
conditions. The feasibility of this set depends on: (i) whether the RNA
synthesis targets `k` for the active/inactive regulated genes of each TF are
separable at all; (ii) whether alpha's and fixed-TF probabilities (`Drhs`)
leave headroom in the remaining budget for the required gap. Sensitive to
TF-regulon expression *structure*, not magnitude.

(b) **Named exception**: `RuntimeError("Solver could not find optimal value")`
(`prob_p.status != "optimal"`, fit_sim_data_1.py:3678-3679) or
`RuntimeError("Solver found an optimum that is inaccurate...")` at :3673-3677.
ECOS returns `infeasible` when the pdiff gap cannot be met for at least one
TF given the fit `k`. Confirmed failing datasets: `precise_eep:BOP27`,
`precise_ica:wt_glc`, `precise_oxyR:wt_glc`, `precise_minspan:wt_glc`,
`precise_misc:wt_no_te`, `precise_ytf{,2,3}:wt_glc{,2}` (per
`rnaseq-dataset-failure-analysis.md`). Note R-solve succeeds on these; only
P-solve is brittle.

(c) ParCa stderr shows `RuntimeError: Solver could not find optimal value`
or the ECOS-inaccurate variant. The step prior is reliably `Fitting promoter
binding` (printed at fit_sim_data_1.py:354 with `VERBOSE > 0`). To identify
the bottleneck TF, re-solve without the pdiff constraint and compute
`pdiff @ P_unconstrained` — any row below 0.06 is the culprit (see
`rnaseq-dataset-failure-analysis.md` Step 1).

---

## 4. Expression adjustments

Anchors: data in `reconstruction/ecoli/flat/adjustments/rna_expression_adjustments.tsv`
(10 RNA entries, all multiplied by 10x except metA and asnA which are also 10x);
applied in `reconstruction/ecoli/fit_sim_data_1.py:1230-1284` (`setRNAExpression`),
invoked from `:227`. Companion protein-level adjustments in
`flat/adjustments/translation_efficiencies_adjustments.tsv` (24 entries, factors
ranging 0.01-5x). Loaded via `reconstruction/ecoli/dataclasses/adjustments.py:14-27`.

The 10 RNA adjustments (all factor 10x, all with `_source: fit_sim_data_1.py`):

| Gene (RNA ID)         | Symbol | Stated purpose                                           |
|-----------------------|--------|----------------------------------------------------------|
| EG11493_RNA           | pabC   | aminodeoxychorismate lyase                               |
| EG12438_RNA           | menH   | menaquinone biosynthesis                                 |
| EG12298_RNA           | yibQ   | anaerobic viability                                      |
| EG11672_RNA           | atoB   | anaerobic viability (acetyl-CoA acetyltransferase)       |
| EG10236_RNA           | dnaB   | DNA helicase, anaerobic replication timing               |
| EG10238_RNA           | dnaE   | DNA polymerase III alpha, replication timing             |
| EG11673_RNA           | folB   | folate, acetate condition viability                      |
| EG10808_RNA           | pyrE   | UTP synthesis (UTP regulation not in model)              |
| EG10581_RNA           | metA   | Met synthesis, low protein expression                    |
| EG10091_RNA           | asnA   | Asn synthetase A, low protein expression                 |

(a) These adjustments compensate for systematic under-expression of specific
enzymes in the reference RNA-seq baseline — they are fit choices calibrated to
`vecoli_m9_glucose_minus_aas`, not physiology. The implicit assumption is
that the reference TPM for these 10 genes is "too low" by roughly 10x relative
to what the downstream simulation needs. When the input dataset already
reports these genes at elevated TPM (e.g. a dataset with higher met/asn
biosynthesis capacity), applying a 10x multiplier on top produces 100x
overshoot.

(b) Silent drift. The adjustment code (fit_sim_data_1.py:1258-1284) maps the
adjustment factor through cistron→TU indexes, multiplies, and renormalizes —
no validation that the result stays physiological. Two failure families:
(1) under-adjusted: alternate dataset still too low, enzyme remains limiting,
downstream kcat fitting (subsystem 2) fails on the relevant AA or sim exhibits
shortages; (2) over-adjusted: alternate dataset already high, expression
explodes, mass fraction drifts, ribosomes/RNAP starved by the redistribution
(since `rna_expression["basal"]` is re-normalized to sum to 1 at :1282-1283,
over-inflating one gene shrinks everything else).

(c) No exception fires from this subsystem directly; signatures appear
downstream. For pattern (1): the subsystem-2 kcat `ValueError` for an AA whose
biosynthesis gene is in the list (metA → MET, asnA → ASN). For pattern (2):
ribosome-protein count drift, RNAP count drift, mass-fraction imbalance. The
multiparca analysis output (`wholecell/io/multiparca_analysis.py`) surfaces
these as doubling-time shifts and mass-fraction regressions across parcas.
Direct inspection: compare `sim_data.process.transcription.rna_expression["basal"]`
on the 10 adjusted genes vs. the raw TPM to see if the 10x factor was
appropriate.

---

## 5. Partitioning architecture

Anchors: `ecoli/processes/allocator.py:40` (`Allocator` class),
`:217` (`calculatePartition` helper), `:36` (`NegativeCountsError`),
`:33` (`ASSERT_POSITIVE_COUNTS = True`). Base classes in
`ecoli/processes/partition.py:26` (`Requester`), `:119` (`Evolver`), `:198`
(`PartitionedProcess`). Integer proportional allocation with stochastic
remainder distribution at allocator.py:234-256.

(a) Every `PartitionedProcess` subclass (polypeptide elongation, transcript
elongation, RNA degradation, chromosome replication, etc. — 21 files under
`ecoli/processes/` reference partitioning) computes a `calculate_request`
from current bulk counts, submits it, and then runs `evolve_state` with
whatever the Allocator grants. The allocator does **priority-tiered integer
proportional allocation**: within a priority level, if total requests exceed
available counts, each process gets
`requests * total_counts / total_requested` (allocator.py:234-238), with
fractional remainders distributed stochastically (:242-252). The assumption
is that requests are calibrated such that the aggregate demand rarely exceeds
supply, and when it does, proportional division is a sensible approximation
of biology. Both assumptions are calibrated to the reference dataset through
the cascade (a) TPM → expression → counts → request. A perturbed dataset can
(i) systematically elevate some process's request without matching counts
(starvation) or (ii) produce a negative intermediate via the
`-self.charging_stoich_matrix @ ...` computations in elongation
(polypeptide_elongation.py:1081-1087).

(b) **Named exception**: `NegativeCountsError` (allocator.py:36) raised
from any of three spots: counts_requested (:141-151), partitioned_counts
(:163-174), counts_unallocated (:179-188). Shows up when a process requests a
negative count (from a signed stoichiometry calc applied to an unexpected
counts vector) or when per-process allocations sum to more than the total
(shouldn't happen with integer proportional, but rounding at layer boundaries
can). The softer failure is **silent starvation**: a process receives fewer
molecules than requested, `evolve_state` runs with reduced activity, and growth
slows without any exception. This is the user-flagged "fragile and needing
replacement" mechanism.

(c) For the exception: grep sim stderr for `NegativeCountsError`; the error
string itself names the molecule and process (allocator.py:143-151). For
starvation: no log line fires by default. Diagnostic is instrumenting
`counts_requested` vs. `partitioned_counts` per process — the sensitivity plan
(dataset-sensitivity-exploration.md:180) specifies this as "min count across
critical species" and "which process requested-but-didn't-get allocations the
most". ATP listener at allocator.py:104-117 already captures ATP request vs
allocation per process; extend that pattern to other critical molecules for
diagnostics.

---

## 6. Polycistronic TU-to-cistron split

Anchors: `reconstruction/ecoli/dataclasses/process/transcription.py:598`
(`_build_rna_data` — builds the `cistron_tu_mapping_matrix`), `:672`
(sparse matrix construction), `:1066`
(`fit_rna_expression` — NNLS inverse), `:737` (where NNLS is first applied:
`expression, _ = self.fit_rna_expression(self.cistron_expression["basal"])`).
Operon grouping at `:680-725`; correction for zero-expression short genes at
`:1101-1171` (`_apply_rnaseq_correction`).

(a) RNA-seq reports per-cistron/per-gene TPM. The model runs on transcription
units (TUs), some of which contain multiple cistrons (operons). The mapping
`cistron_tu_mapping_matrix` is 0/1 per (cistron, TU) pair (transcription.py:672).
To invert — given per-cistron measurements, recover per-TU expression — the
model uses non-negative least squares: `fit_rna_expression` calls `fast_nnls`
on the mapping matrix (transcription.py:1071). This assumes the measured
cistron expression vector is approximately in the image of the mapping matrix;
when a dataset reports anomalously different TPM for two cistrons in the same
TU (biologically they must be equal), NNLS returns the best-fit projection
but the residual is discarded. There's an additional correction
(`_apply_rnaseq_correction`, :1101) that zeroes out expression for mRNA
cistrons shorter than the minimum-nonzero-length cistron when they appear in
multicistronic operons and the operon model is on (:734).

(b) Silent drift. The NNLS residual is computed but not inspected or thresholded
(returned as `_` at transcription.py:737, :1071). Datasets with strong
within-operon TPM variation get projected onto the TU basis and the
inconsistency is absorbed without warning. Second, `_apply_rnaseq_correction`
(transcription.py:1115-1171) silently zeroes expression of short-gene cistrons
whose TPM is already zero — a dataset with genuine zero TPM on a short mRNA
in a polycistronic operon triggers the correction even when the zero is
biologically correct.

(c) No exception. Diagnostic is inspecting the NNLS residual at the
`fit_rna_expression` call sites — the second return value `res` is a residual
norm. Post-hoc: compare `cistron_expression["basal"]` (per-cistron input) to
`cistron_tu_mapping_matrix @ rna_expression["basal"]` (per-cistron reconstruction
from fit) — large element-wise differences identify operons where the split
lost information. The corrected-gene indexes are tracked in
`cistron_data["uses_corrected_seq_counts"]` (transcription.py:497, :1169) —
count of True entries per dataset quantifies how much silent correction the
dataset triggered.

---

## 7. Fill-missing-genes-from-ref

Anchors: `reconstruction/ecoli/dataclasses/process/transcription.py:541-569`
(only consumer); config knob `rnaseq_fill_missing_genes_from_ref` threaded
through `runscripts/parca.py:59`, `reconstruction/ecoli/fit_sim_data_1.py:204-206`,
`reconstruction/ecoli/simulation_data.py:52, :80`. Default `true` in
`configs/default.json:56`.

(a) When true (default), genes present in the model but absent from the input
TPM table have their TPM filled from the legacy reference table
(`raw_data.rna_seq_data.rnaseq_{RNA_SEQ_ANALYSIS}_mean`) indexed by
`sim_data.basal_expression_condition` (transcription.py:543-548). A single
`UserWarning` is emitted with the fill count (`"{filled} genes were missing
from experimental RNA-seq dataset ..."`, transcription.py:563-569). The
implicit assumption is that the reference TPM is an acceptable proxy for
missing genes — which silently couples the input dataset's effective coverage
to the reference even when the user specified a completely different dataset.

(b) Silent bias: downstream subsystems (1, 3, 6, 8) see a hybrid TPM vector
whose "coverage" column (`_cistron_is_rnaseq_covered`, transcription.py:587)
flags only the experimental genes. The dataset-quality illusion: a dataset
with 20% coverage looks similar to one with 100% coverage after fill, but the
subsystem-2 kcat fits and subsystem-3 P-solve are being driven by reference
values on the filled genes. If those filled genes happen to be TF-regulated
or in an AA biosynthesis pathway that the real experiment would have measured
differently, the model silently uses the wrong values.

(c) Warning at ParCa stderr: grep for `"genes were missing from experimental
RNA-seq dataset"`. The preceding integer is the fill count. Diagnostic: set
`rnaseq_fill_missing_genes_from_ref: false` in config and re-run — if
subsystem 2 or 3 now fails where it previously succeeded, the fill was
masking a real coverage gap. Per-dataset fill rate and overlap with
TF-regulated gene sets is the Step 3 analysis in
`.claude/plans/rnaseq-dataset-failure-analysis.md:123-134`.

---

## 8. Translation efficiencies / ribosome allocation

Anchors: base translation efficiency loaded at
`reconstruction/ecoli/dataclasses/process/translation.py:260-280` (NaN filled
with `np.nanmean`); adjustments at
`reconstruction/ecoli/fit_sim_data_1.py:1178-1199` (`setTranslationEfficiencies`
reads `flat/adjustments/translation_efficiencies_adjustments.tsv` — 24 entries)
and `:1202-1227` (`set_balanced_translation_efficiencies` reads
`flat/adjustments/balanced_translation_efficiencies.tsv` — 4 groups of
ribosomal proteins, sets them to their group mean). Ribosome capacity fit at
`:1755-1836+` (`setRibosomeCountsConstrainedByPhysiology`).

(a) Translation efficiency per monomer comes from a curated `raw_data.translation_efficiency`
table, NaN-filled with the non-NaN mean (translation.py:277-280). Two adjustment
layers then mutate the vector: (i) per-protein factor from
`translation_efficiencies_adjustments.tsv` (24 proteins, factors 0.01-5.0 — most
are 3-5x for electron-transport and NADH:quinone oxidoreductase subunits;
fit_sim_data_1.py:1195-1199); (ii) per-operon balancing from
`balanced_translation_efficiencies.tsv` which sets all proteins in the listed
operons to the mean of the group (4 groups, all ribosomal proteins;
fit_sim_data_1.py:1220-1227). `setRibosomeCountsConstrainedByPhysiology`
(:1755-1836+) then solves for ribosomal subunit counts under three constraints:
(1) protein distribution must double per cell cycle, (2) measured rRNA mass
fractions match, (3) ribosomal protein counts match RNA expression. The
implicit assumption: the reference TPM has approximately uniform expression
across each of the 4 balanced ribosomal-protein operons, so setting them to
the mean is a small correction, not a large override. For perturbed datasets
that deliberately skew ribosomal-protein expression, this mean-substitution
*erases* the perturbation.

(b) Silent drift for unperturbed sensitivity runs (the adjustments nudge
downstream fitting by a calibrated amount). **Stronger failure mode for
ribosomal-protein targeted perturbations**: the `set_balanced_translation_efficiencies`
step overwrites any dataset-level variation within the listed operons
(28 ribosomal proteins total across the 4 groups) with the group mean —
meaning sensitivity runs that scale a subset of ribosomal-protein TPM expect
a downstream response that silently fails to materialize. For the ribosome
capacity fit itself, if input TPM yields a ribosomal-protein count profile
incompatible with the 3 constraints, the fit converges to an internally
inconsistent point rather than raising.

(c) No exception at the adjustment step. Detection: diff
`sim_data.process.translation.translation_efficiencies_by_monomer` against
the un-adjusted baseline to see which entries were mutated; for the balancing
step, check whether any of the 28 listed monomer IDs have identical values
(they will, by construction, whether the input dataset agreed or not). For
the ribosome fit, `setRibosomeCountsConstrainedByPhysiology` has three
internal convergence paths (:1810-constraint blocks); instrument by comparing
fit ribosomal-protein counts vs. constraint-3 expected counts — a gap
indicates the fit silently compromised. In sim trajectories, watch
`active_ribosome` count and elongation rate per generation — monotone decline
across generations is the ribosome-allocation failure signature.

---

## 9. Metabolism (FBA) homeostatic target initialization

Anchors: `ecoli/processes/metabolism_redux.py:207-224`
(homeostatic objective construction), `:966` (runtime update via
`update_homeostatic_targets`). Same pattern at
`ecoli/processes/metabolism_redux_classic.py:215-225` and
`ecoli/processes/metabolism.py:785-793`. Targets come from
`concentrations_based_on_nutrients(media_id=current_timeline[0][1], ...)`
followed by `getBiomassAsConcentrations(doubling_time)`
(metabolism_redux.py:209-213). The biomass concentration function lives in
`reconstruction/ecoli/dataclasses/growth_rate_dependent_parameters.py:311`.

(a) The homeostatic objective (the FBA target that pulls the cell toward a
specific internal metabolite concentration profile) is built by taking nutrient
concentrations from `concentration_updates` plus a biomass contribution scaled
by `doubling_time`. `doubling_time` is pulled from `condition_to_doubling_time`
via `sim_data.condition` (simulation_data.py:72) — for the basal condition,
this resolves to the doubling time implied by the reference mass-fraction
calibration. The biomass composition itself (`getBiomassAsConcentrations`) is
a function of total cell dry mass and doubling time (growth_rate_dependent_parameters.py:311-364);
total cell dry mass was set in `buildBasalCellSpecifications` from the
basal TPM profile via `avgCellDryMassInit` (fit_sim_data_1.py:646, :671). So
the homeostatic targets that drive the FBA are indirectly derived from the
input TPM's induced mass.

(b) Silent drift. If the input dataset shifts average cell mass or protein/RNA
mass ratio at the ParCa stage, the homeostatic targets move with it. FBA then
solves against targets that no longer match the actual cellular composition,
producing a steady-state mass that diverges from the target over generations.
No exception — the FBA step will find *a* solution, just one calibrated to
the wrong setpoint. Specific assertion paths exist when the fit becomes
degenerate: `fitPromoterBoundProbability` checks at `:3805, :3825`
("Check results from fitPromoterBoundProbability and Kd values.") can fire
when downstream transcription probabilities are out of bounds, but those are
P-solve-rooted (subsystem 3), not directly metabolism-rooted.

(c) No exception in the baseline case; signature is in sim trajectories.
Listeners to watch: `target_homeostatic_dmdt` vs. `estimated_homeostatic_dmdt`
(metabolism_redux.py:383-392) — persistent gap means targets are unattainable.
Growth rate relative to `sim_data.doubling_time` — systematic shortfall means
the target corresponds to a faster doubling time than the cell can support.
Mass fraction drift across generations (plan-flagged "unhealthy sim" regime)
is the canonical signal. The `mass_fraction_summary` analysis
(configured in `configs/test_multi_parca.json`) surfaces exactly this.

---

## Open gaps

Subsystems partially analyzed or with weak source anchors:

- **ppGpp runtime coupling in simulation**: the ParCa-side
  `adjust_polymerizing_ppgpp_expression` is well-anchored, but the runtime
  ppGpp dynamics in `ecoli/processes/polypeptide_elongation.py:1103-1134`
  (calling `ppgpp_metabolite_changes`) live in the charging path and were
  not traced here for dataset-dependence. Likely also sensitive to tRNA
  expression distribution.
- **Kinetic FBA constraints**: `metabolism_redux.py:247-253`
  (`active_constraints_mask`, `constraints_to_disable`) references a
  per-reaction mask whose origin I didn't trace. Could be an additional
  TPM-dependent layer; not obvious from the consuming code.
- **Per-TF pdiff tolerance**: the global `PROMOTER_PDIFF_THRESHOLD = 0.06`
  applies uniformly to all TFs (subsystem 3). Whether some TFs are
  systematically harder to satisfy (e.g. TFs with few regulon members) is a
  structural question the source alone doesn't answer; identifying the
  typical-bottleneck TFs requires the instrumented infeasibility-analysis
  run specified in `rnaseq-dataset-failure-analysis.md:98-108`.
- **R-solve (precursor to P-solve)**: the user's plan (subsystem 3 context)
  mentions that R-solve succeeds on all observed failing datasets, so
  the R-fitting step inside `fitPromoterBoundProbability` was not enumerated
  separately. If future datasets make R-solve fail, that's a distinct
  fragility dimension not covered here.
- **Coverage of equilibrium/complexation processes**: `ecoli/processes/equilibrium.py`
  and `complexation.py` are PartitionedProcesses (subsystem 5) but their
  internal dataset sensitivity was not separately analyzed; assumption is
  failures route through subsystem 5's allocator exceptions.
- **Dataset-dependent condition labels**: `basal_expression_condition` is a
  free-form string config knob (CLAUDE.md). If a custom dataset uses a label
  not present in `raw_data.rna_seq_data.rnaseq_*_mean`, the fill-from-ref
  path (subsystem 7) will silently hit a `KeyError` rather than warning —
  did not verify this code path.
