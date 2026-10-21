"""
Multiexperiment FBA flux comparison (plotly scatters)

Produces two HTML plots (mirrors cistron_count_comparison_plotly.py /
protein_count_comparison_plotly.py), and shares its richer hover/marker
machinery with fba_flux_lineage_comparisons_plotly.py (that file restricts
the same kind of scatter to the BFS "lineage" of one center reaction; this
one has no lineage and instead plots every shared reaction):
  1. fba_flux_comparison_plotly_highlighted_<exp1>_vs_<exp2>.html: every
     reaction in one background color (lightseagreen) except a
     user-specified highlight list of reaction ids, where each highlighted
     reaction gets its own color and its own legend entry.
  2. fba_flux_comparison_plotly_molecules_<exp1>_vs_<exp2>.html: reactions
     colored by which compartment-tagged form of a params-driven
     "molecule of interest" (base id) they touch as a reactant, product, or
     catalyst.

In both figures, kinetically constrained reactions are drawn as squares and
everything else as circles (see _kinetic_symbols_for()).

# TODOs:
# TODO 1: write a long and descriptive docstring
# TODO: make config options
# TODO: if a user specifies a reaction to be plotted in a config that has zero
 counts in both sims, make sure its still plotted
# TODO: remove default variables
# TODO: decide if the reaction list cap should be expanded based on how many this affects
# TODO: determine what is causing the odd warning to emit
# TODO: resolve all other TODOs in the script
# TODO: figure out how to plot negative fluxes in a way that makes sense (maybe
 triangles for normally negative, circles for positive, square for negative in one and positive in another)
 # TODO: decide if the reactant and product avg counts should be the total (from listener) or free
 # todo: decide if the catalyst avg counts should be the total (from listener for monomers) or free
"""

import os
import pickle
import re
from typing import Any, cast

import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import plotly.graph_objects as go

from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
    read_stacked_columns,
)
from reconstruction.ecoli.dataclasses.process.metabolism import REVERSE_TAG

# USER-DEFINED REACTIONS OF INTEREST (each gets its own color/trace in the
# highlighted plot):
PLOT_REACTIONS_OF_INTEREST = ["RXN0-5405"]

# Default molecules of interest for the by-molecule highlight scheme
# (overridable via params["molecules_of_interest"], with
# params["metabolites_of_interest"] accepted as a backward-compat alias).
# Give BASE (compartment-stripped) ids here -- every compartment-tagged form
# actually present among the plotted reactions is auto-discovered and given
# its own color/legend entry (see discover_highlight_molecule_matches()):
DEFAULT_METABOLITES_OF_INTEREST = ["MET"]

FLUX_COLUMN = "listeners__fba_results__reaction_fluxes"

# Cap for long id lists shown in hover text (show first N, then "(+M more)"):
MAX_HOVER_IDS = 5

# Distinct high-contrast qualitative palette used to give each highlighted
# reaction/molecule its own legend color, from
# https://sashamaps.net/docs/resources/20-colors/ (cycles via % len(...) if
# there are more highlighted items than colors):
SASHA_PALETTE = [
    "#e6194B",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#f032e6",
    "#469990",
    "#9A6324",
    "#800000",
    "#808000",
    "#000075",
    "#e6beff",
    "#aaffc3",
    "#ffd8b1",
    "#42d4f4",
]

# This |net| / total (|positive| or |negative|) ratio a forward/reverse pair is
# flagged in the hover data as a "near-futile cycle" when both directions of a
# reversible reaction carry real flux, but they nearly cancel, so the pair
# moves almost no net mass. This is needed because each direction of a
# reversible reaction is plotted as its own point (never netted together, or
# most of the interesting structure would collapse near zero), so this flag
# points out when the opposite direction is actually canceling out most of a
# given direction's flux:
FUTILE_NET_FRACTION = 0.1

# Floor for the log offset (this ensures the value added to the raw fluxes is
# not below 10**(-MAX_LOG_ORDERS) times the largest |avg flux| (i.e. a
# millionth of it when MAX_LOG_ORDERS = 6)). This ensures that a single near-zero
# flux will not drag the added value down so much that a huge empty low region
# appears on the graph.
MAX_LOG_ORDERS = 6.0


def flux_units(flux_column: str) -> str:
    """Return the flux unit string for a given FBA flux listener column.

    reaction_fluxes / solution_fluxes are the FBA solver velocities on a
    concentration basis (mM/s, i.e. mmol/L/s); estimated_fluxes /
    base_reaction_fluxes are the same converted to molecule counts per
    timestep. # Todo: double check this again
    """
    leaf = flux_column.rsplit("__", 1)[-1]
    if leaf in ("estimated_fluxes", "base_reaction_fluxes"):
        return "counts/timestep"
    return "mM/s"


def _strip_compartment(mol_id: str) -> str:
    """Removes a trailing compartment tag from a molecule id."""
    if mol_id.endswith("]") and len(mol_id) >= 3 and mol_id[-3] == "[":
        return mol_id[:-3]
    return mol_id


def get_reaction_categories(sim_data, flux_column):
    """Builds the per-reaction category sets and hover-detail maps from sim_data.

    Returns a dict with:
      - kinetic_set: reaction ids carrying a kinetic (kcat) constraint
        (metabolism.kinetic_constraint_reactions).
      - catalyst_map: reaction id -> list of catalyst ids
        (metabolism.reaction_catalysts; only non-empty entries).
      - stoich: reaction id -> {metabolite id: coeff}
        (metabolism.reaction_stoich).
    """
    metabolism = sim_data.process.metabolism
    kinetic_set = set(metabolism.kinetic_constraint_reactions)
    catalyst_map = {
        rxn: list(cats) for rxn, cats in metabolism.reaction_catalysts.items() if cats
    }
    stoich = metabolism.reaction_stoich
    return {
        "kinetic_set": kinetic_set,
        "catalyst_map": catalyst_map,
        "stoich": stoich,
    }


def reaction_reactants(rxn_id, stoich):
    """Return the list of reactant (consumed, i.e. negative-coefficient)
    participant ids for a reaction, in stoich dict order."""
    return [met_id for met_id, coeff in stoich.get(rxn_id, {}).items() if coeff < 0]


def reaction_products(rxn_id, stoich):
    """Return the list of product (produced, i.e. positive-coefficient)
    participant ids for a reaction, in stoich dict order."""
    return [met_id for met_id, coeff in stoich.get(rxn_id, {}).items() if coeff > 0]


def reaction_highlight_matches(rxn_id, stoich, catalyst_map, base_set):
    """Maps each molecule-of-interest base id a reaction contains to the set
    of roles it plays in that reaction (reactant, product, or catalyst).

    Used for the hover text's "Molecule(s) of interest" block (which reports
    by base id, regardless of compartment); see
    discover_highlight_molecule_matches() for the compartment-TAGGED version
    used to color/group the molecule-highlight figure.
    """
    matches: dict[str, set[str]] = {}
    for met_id, coeff in stoich.get(rxn_id, {}).items():
        b = _strip_compartment(met_id)
        if b in base_set:
            role = "product" if coeff > 0 else "reactant"
            matches.setdefault(b, set()).add(role)
    for cat_id in catalyst_map.get(rxn_id, []):
        b = _strip_compartment(cat_id)
        if b in base_set:
            matches.setdefault(b, set()).add("catalyst")
    return matches


def discover_highlight_molecule_matches(rxn_ids, stoich, catalyst_map, molecule_bases):
    """Discovers, for the reactions actually being plotted, every
    compartment-tagged form of the requested base molecule ids that is
    present as a reactant, product, or catalyst, and which reactions (by
    position in rxn_ids) match each one.

    Config gives BASE ids (e.g. "MET"), but this plot colors/groups by the
    compartment-TAGGED id (e.g. MET[c] and MET[p] each get their own color and
    legend entry, labeled with how many of the plotted reactions contain that
    exact tagged form) since a molecule's compartment often matters for which
    reactions can actually touch it. A reaction containing two matching
    tagged molecules appears in both lists (drawn once per molecule, as
    overlapping points at the same coordinates).

    Returns {tagged_id: [point_indices]}, sorted by tagged id so legend order
    is deterministic run to run.
    """
    matches: dict[str, list[int]] = {}
    for i, rxn_id in enumerate(rxn_ids):
        tagged_here: set[str] = set()
        for met_id in stoich.get(rxn_id, {}):
            if _strip_compartment(met_id) in molecule_bases:
                tagged_here.add(met_id)
        for cat_id in catalyst_map.get(rxn_id, []):
            if _strip_compartment(cat_id) in molecule_bases:
                tagged_here.add(cat_id)
        for tagged_id in tagged_here:
            matches.setdefault(tagged_id, []).append(i)
    return dict(sorted(matches.items()))


def _format_conc(count, ctm):
    """Render a bulk count as a concentration string like ' (30.2 µM)' for the
    hover, using a sim's average counts_to_molar scalar (ctm, in mM per count,
    i.e. the model's own listeners__enzyme_kinetics__counts_to_molar =
    1/(nAvogadro * cell_volume); conc_mM = count * ctm).

    Returns '' when the count or ctm is missing (so the hover just shows the
    bare count) and auto-scales the unit to mM / µM / nM so the number stays
    readable. Only the magnitude is shown (concentrations are always >= 0).
    """
    if count is None or ctm is None:
        return ""
    conc_mM = count * ctm
    if conc_mM <= 0:
        return ""
    if conc_mM >= 1.0:
        return f" ({conc_mM:.3g} mM)"
    conc_uM = conc_mM * 1e3
    if conc_uM >= 1.0:
        return f" ({conc_uM:.3g} µM)"
    return f" ({conc_mM * 1e6:.3g} nM)"


def _format_id_list_with_counts(ids, counts_1, counts_2, ctm_1=None, ctm_2=None):
    """Format a list of ids for hover, each annotated with its per-sim average
    bulk count (when resolvable) and, when a counts_to_molar scalar is given
    for that sim, the corresponding concentration in parentheses after the
    count (see _format_conc). One id per line (via '<br>') on its own
    indented '-' bullet, capping at MAX_HOVER_IDS. An id missing from a sim's
    count map (unresolved bulk index, or absent from that sim's bulk
    container) is shown as 'n/a' for that sim rather than silently dropped,
    so a gap is visible.
    """
    ids = list(ids)
    if not ids:
        return ""

    def _one(mol_id):
        c1 = counts_1.get(mol_id)
        c2 = counts_2.get(mol_id)
        c1_str = f"{c1:.3g}{_format_conc(c1, ctm_1)}" if c1 is not None else "n/a"
        c2_str = f"{c2:.3g}{_format_conc(c2, ctm_2)}" if c2 is not None else "n/a"
        return f"&nbsp;&nbsp;- {mol_id} (sim1: {c1_str}, sim2: {c2_str})"

    shown = "<br>".join(_one(mid) for mid in ids[:MAX_HOVER_IDS])
    if len(ids) > MAX_HOVER_IDS:
        shown += f"<br>&nbsp;&nbsp;- (+{len(ids) - MAX_HOVER_IDS} more)"
    return shown


def bulk_ids_to_indices(mol_ids, bulk_id_to_idx):
    """Resolves a list of molecule ids to bulk indices for one sim's own bulk
    ordering.

    Returns (id_to_idx, missing_ids): id_to_idx holds only the ids that were
    mapped. missing_ids are the ones that did not map -- these are
    essentially only species that are not free bulk metabolites at all
    (mass-less class/pool terms, or carrier-/macromolecule-bound
    intermediates (like ACP-, tRNA-, and protein-conjugated species whose
    mass rides on the carrier)). Thus, a missing id is not an error here
    (the hover count is just omitted).

    Matching an id to its bulk count here first checks the exact
    compartment-tagged id (since small molecules can exist in several
    compartments), and if there does not exist a bulk id for that match, a
    fallback search is run on the compartment-stripped base id. This base-id
    fallback was built because a molecule's compartment is set independently
    in two places, and they disagree for a few proteins (around 19 as of
    7/01/26). The reaction definition (reaction_stoich) tags a participant
    with the compartment the reaction chemistry was defined in, while the
    bulk state tracks the molecule under the compartment it is localized to
    (see the _build_compartments() function in
    reconstruction/ecoli/dataclasses/getter_functions.py).

    For example, RXN0-312 (the ArcB dephosphorylation reaction) automatically
    tags ARCB-CPLX with [c] because the phospho-transfer chemistry is located
    in the cytoplasm, but ArcB physically sits in the inner membrane within
    the sim (i.e. its counts are stored in the bulk state as ARCB-CPLX[i]).
    The reaction definition and the bulk state are both correct, but they
    disagree on the compartment tag. Thus, the base-id fallback allows a
    reaction participant to be counted even if its compartment tag disagrees
    with the bulk state (NOTE: all proteins and complexes are only tracked in
    one compartment in the model (unlike metabolites), so the fallback is
    safe because there is no ambiguity in which bulk pool to use).
    """
    bulk_base_to_idx = {}
    for bid, i in bulk_id_to_idx.items():
        bulk_base_to_idx.setdefault(_strip_compartment(bid), i)

    id_to_idx = {}
    missing = []
    for mol_id in mol_ids:
        idx = bulk_id_to_idx.get(mol_id)
        if idx is None:
            # If the exact (tagged) id is absent, retry on the
            # compartment-stripped base:
            idx = bulk_base_to_idx.get(_strip_compartment(mol_id))
        if idx is None:
            missing.append(mol_id)
        else:
            id_to_idx[mol_id] = idx
    return id_to_idx, missing


def read_bulk_means_for_experiment(
    conn: DuckDBPyConnection,
    history_sql: str,
    experiment_id: str,
    bulk_indices: list[int],
    gen_floor: int,
) -> dict[int, float]:
    """Mean-of-per-cell-mean bulk counts for one experiment's bulk indices.

    Mirrors read_bulk_rna_means_for_experiment in cistron_count_comparison_plotly.py:
    reads experiment_id's rows only, averages per cell (generation >= gen_floor),
    then averages across cells. Returns {bulk_idx: mean_count}.

    NOTE: bulk_indices must be indices into this experiment's own sim_data bulk-
    molecule ordering (integer bulk indices are not interchangeable between
    experiments, so this must be called once per experiment with that
    experiment's own index list (see the sim_data_1 / sim_data_2 loading below)).
    """
    if not bulk_indices:
        return {}
    bulk_indices = [int(i) for i in bulk_indices]
    names = [f"bulkrxn_{i}" for i in bulk_indices]
    bulk_expr = named_idx("bulk", names, [bulk_indices])
    subquery = cast(
        str, read_stacked_columns(history_sql, [bulk_expr], order_results=False)
    )
    avg_exprs = ", ".join(f'avg("{n}") AS "{n}"' for n in names)
    exp_literal = experiment_id.replace("'", "''")
    per_cell = conn.sql(
        f"""
        SELECT experiment_id, {avg_exprs}
        FROM ({subquery})
        WHERE generation >= {gen_floor}
            AND experiment_id = '{exp_literal}'
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).pl()
    if per_cell.height == 0:
        return {idx: 0.0 for idx in bulk_indices}
    means = per_cell.drop("experiment_id").mean()
    row = means.row(0, named=True)
    return {idx: row[f"bulkrxn_{idx}"] for idx in bulk_indices}


def read_counts_to_molar_mean_for_experiment(
    conn: DuckDBPyConnection,
    history_sql: str,
    experiment_id: str,
    gen_floor: int,
) -> float | None:
    """Mean-of-per-cell-mean counts_to_molar scalar for one experiment, in
    mM per count (i.e. conc_mM = bulk_count * ctm).

    Reads the model's own per-timestep scalar
    ``listeners__enzyme_kinetics__counts_to_molar`` ( = 1/(nAvogadro *
    cell_volume)), averaging it the same way the bulk counts are averaged
    (per cell over generation >= gen_floor, then across cells) so the hover's
    counts and the concentration derived from them use matching windows. The
    first timestep of each cell has counts_to_molar == 0 (the volume listener
    is not yet populated), so those rows are excluded to avoid biasing the
    mean low.

    Returns None if the enzyme-kinetics listener was not emitted (the hover
    then just shows bare counts with no concentration).
    """
    col = "listeners__enzyme_kinetics__counts_to_molar"
    exp_literal = experiment_id.replace("'", "''")
    try:
        subquery = cast(
            str, read_stacked_columns(history_sql, [col], order_results=False)
        )
        per_cell = conn.sql(
            f"""
            SELECT avg("{col}") AS ctm
            FROM ({subquery})
            WHERE generation >= {gen_floor}
                AND experiment_id = '{exp_literal}'
                AND "{col}" > 0
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
            """
        ).pl()
    except Exception as e:
        print(
            f"  NOTE: could not read counts_to_molar for {experiment_id} "
            f"({type(e).__name__}); hover concentrations omitted for this sim."
        )
        return None
    if per_cell.height == 0:
        return None
    return float(per_cell["ctm"].mean())


def direction_partner(rxn_id):
    """Returns (forward_id, reverse_id, this_direction) for an FBA reaction id.

    The FBA network stores every reversible reaction as two independent
    reactions: ``X`` and ``X (reverse)`` (because the linear program solver
    holds all fluxes at >= 0 (GLPK's default lower bound), so a reaction can
    never carry negative flux itself). ``REVERSE_TAG`` is imported from the
    metabolism process so this stays in step with the id construction.

    this_direction is "forward" or "reverse" and describes the id passed in.
    The returned pair of ids is always (forward, reverse) regardless.
    """
    if rxn_id.endswith(REVERSE_TAG):
        return rxn_id[: -len(REVERSE_TAG)], rxn_id, "reverse"
    return rxn_id, rxn_id + REVERSE_TAG, "forward"


def _format_direction_block(rxn_id, avg_1, avg_2, flux_unit):
    """Format hover lines reporting both directions of a reaction for both sims.

    The plot plots each direction as its own point on the scatter to avoid
    plotting net fluxes at one point (as a lot would crowd near zero and it
    would then be difficult to tell the magnitude of each direction on its
    own visually). A consequence of this is that a single plotted point cannot
    fully describe whether the opposite direction is also running, so this
    function looks up the other direction's flux and derives:
      - net   = forward - reverse (the mass that actually moved -- this is
                what the model's own ``base_reaction_fluxes`` column
                aggregates)
      - total = |forward| + |reverse| (total turnover regardless of direction)

    avg_1/avg_2 must be maps over all shared reaction ids (not just the
    plotted/active subset), so the partner is still found even when it is
    inactive. A reaction with no partner in the map is irreversible in this
    model (or reverse-only, as some reverse ids have no forward entry) and is
    labeled as such rather than being shown with a 0.
    """
    # Resolve both directions' ids and look up each one's avg flux in both sims
    # (None where that id is absent from the map):
    fwd_id, rev_id, this_dir = direction_partner(rxn_id)
    f1, f2 = avg_1.get(fwd_id), avg_2.get(fwd_id)
    r1, r2 = avg_1.get(rev_id), avg_2.get(rev_id)

    # The partner is whichever of the two ids is not the plotted point:
    partner_present = (
        (r1 is not None or r2 is not None)
        if this_dir == "forward"
        else (f1 is not None or f2 is not None)
    )
    if not partner_present:
        kind = (
            "irreversible (no reverse partner in the model)"
            if this_dir == "forward"
            else "reverse-only (no forward partner in the model)"
        )
        return f"Direction: {this_dir} -- {kind}<br>"

    def _pair(label, v1, v2):
        s1 = f"{v1:.3e}" if v1 is not None else "n/a"
        s2 = f"{v2:.3e}" if v2 is not None else "n/a"
        return f"&nbsp;&nbsp;- {label}: sim1 {s1}, sim2 {s2}<br>"

    lines = [
        f"Direction: <b>{this_dir}</b> (this point)<br>",
        f"Both directions (avg flux, {flux_unit}):<br>",
        _pair("forward", f1, f2),
        _pair("reverse", r1, r2),
    ]

    # net/total only where both directions resolved for that sim:
    def _derive(f, r):
        if f is None or r is None:
            return None, None
        return f - r, f + r

    n1, t1 = _derive(f1, r1)
    n2, t2 = _derive(f2, r2)
    if n1 is not None or n2 is not None:
        lines.append(_pair("net (fwd-rev)", n1, n2))
        lines.append(_pair("total (fwd+rev)", t1, t2))
        # Flag a near-futile cycle per sim when both directions run hard
        # (total > 0) but nearly cancel, i.e. |net| is a tiny fraction of total:
        futile = [
            name
            for name, n, t in (("sim1", n1, t1), ("sim2", n2, t2))
            if n is not None
            and t is not None
            and t > 0
            and abs(n) < FUTILE_NET_FRACTION * t
        ]
        if futile:
            lines.append(
                f"&nbsp;&nbsp;<b>near-futile cycle in {', '.join(futile)}</b> "
                f"(|net| &lt; {FUTILE_NET_FRACTION:.0%} of total)<br>"
            )
    return "".join(lines)


def get_kinetic_constraint_info(sim_data):
    """Obtains per-reaction kcat value, constraint-type, and constraining-enzyme
    info, keyed by reaction id.

    This function reads the parallel arrays that
    ``Metabolism._lambdify_constraints`` produces (all indexed the same as
    ``kinetic_constraint_reactions``):
      - ``_kcats``: (n rxns, 3) array of [min, mean, max] kcat, already
        temperature-adjusted to 1/s.
      - ``constraint_is_kcat_only``: True when the constraint is a bare kcat with
        no saturation term (so the target is just kcat x [enzyme]); False when a
        substrate saturation expression is also applied.
      - ``_enzymes``: a *string* repr() of a sympy list like ``'[e[250], e[199]]'``,
        one entry per reaction, where the index selects into
        ``kinetic_constraint_enzymes``. It is stored as a string (it gets
        compiled at sim time), so the indices are recovered by regex
        rather than by reevaluating it.

    Returns {rxn_id: {"kcat_min", "kcat_mean", "kcat_max", "is_kcat_only",
    "enzyme"}}. NOTE: This assumes that this structural data from sim_data is
    identical for both sims being compared, so it cannot explain sim-to-sim
    flux differences (see read_kinetic_targets_for_experiment for the per-sim
    target values).
    """
    metabolism = sim_data.process.metabolism
    # These arrays are all parallel to kinetic_constraint_reactions, so index i
    # keys the same reaction in each of them:
    rxns = list(metabolism.kinetic_constraint_reactions)
    kcats = np.asarray(metabolism._kcats, dtype=float)
    is_kcat_only = np.asarray(metabolism.constraint_is_kcat_only, dtype=bool)
    enzyme_ids = list(metabolism.kinetic_constraint_enzymes)
    # _enzymes is a sympy list stored as a STRING (e.g. '[e[250], e[199]]'); the
    # regex pulls each 'e[N]' index out to recover which enzyme constrains each rxn:
    enzyme_idxs = [int(i) for i in re.findall(r"e\[(\d+)\]", str(metabolism._enzymes))]

    info = {}
    for i, rxn in enumerate(rxns):
        # Translate the parsed enzyme index into an enzyme id, guarding a
        # short/mismatched parse or an out-of-range index:
        enzyme = (
            enzyme_ids[enzyme_idxs[i]]
            if i < len(enzyme_idxs) and enzyme_idxs[i] < len(enzyme_ids)
            else None
        )
        info[rxn] = {
            # kcats columns are [min, mean, max] (1/s, temperature-adjusted):
            "kcat_min": float(kcats[i, 0]),
            "kcat_mean": float(kcats[i, 1]),
            "kcat_max": float(kcats[i, 2]),
            "is_kcat_only": bool(is_kcat_only[i]) if i < len(is_kcat_only) else None,
            "enzyme": enzyme,
        }
    return info


def read_kinetic_targets_for_experiment(
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    experiment_id: str,
    gen_floor: int,
) -> dict[str, dict[str, float]]:
    """Reads data from listeners and computes the mean of the per-cell-means of
    kinetic target/actual fluxes for one experiment.

    Averages exactly like the reaction-flux function does
    (read_bulk_means_for_experiment()).

    Returns {rxn_id: {"target", "lower", "upper", "actual"}} in mM/s, or {} if
    this experiment did not emit the enzyme-kinetics listener.
    """
    # Enzyme-kinetics listener columns holding the per-timestep kinetic target
    # flux (and the flux the solver actually achieved) for each kinetically
    # constrained reaction. All four are indexed by the same id list (the
    # process's kinetics_constrained_reactions (the constraints left active after
    # constraints_to_disable, a subset of sim_data's kinetic_constraint_reactions)),
    # so the ids are read from field_metadata per experiment. These are
    # written as (target / timestep) in CONC_UNITS = mmol/L, i.e. mM/s:
    target_columns = {
        "target": "listeners__enzyme_kinetics__target_fluxes",
        "lower": "listeners__enzyme_kinetics__target_fluxes_lower",
        "upper": "listeners__enzyme_kinetics__target_fluxes_upper",
        "actual": "listeners__enzyme_kinetics__actual_fluxes",
    }
    # keys fixes a stable order:
    keys = list(target_columns)
    cols = [target_columns[k] for k in keys]
    exp_config = (
        f"SELECT * FROM ({config_sql}) WHERE experiment_id = "
        f"'{experiment_id.replace(chr(39), chr(39) * 2)}'"
    )
    try:
        # IDs come from the listener's own metadata, since the active constraint
        # set is a subset of sim_data's kinetic_constraint_reactions. A raised
        # exception means the listener was never emitted, so {} is returned
        # and the hover skips the kinetic-target block:
        target_rxn_ids = field_metadata(conn, exp_config, cols[0])
    except Exception as e:
        print(
            f"  NOTE: {experiment_id} has no '{cols[0]}' metadata "
            f"({type(e).__name__}); per-sim kinetic targets omitted from hover."
        )
        return {}
    if not target_rxn_ids:
        return {}

    exp_literal = experiment_id.replace("'", "''")
    try:
        subquery = cast(
            str, read_stacked_columns(history_sql, cols, order_results=False)
        )
        # All four lists share one length/ordering, so unnesting them together
        # zips them row-wise against a single generate_subscripts index:
        unnest_exprs = ",\n                ".join(
            f"unnest({col}) AS {k}" for k, col in zip(keys, cols)
        )
        per_cell_aggs = ", ".join(f'avg("{k}") AS "{k}"' for k in keys)
        across_cell_aggs = ", ".join(f'avg("{k}") AS "{k}"' for k in keys)
        # unnested: explode all four per-timestep arrays together, tagging each
        # row with idx = its position in the constraint list. per_cell:
        # average each column within a cell (idx in the GROUP BY keeps
        # reactions separate). Final select: averages those per-cell means across
        # cells, one row per idx:
        result = conn.sql(
            f"""
            WITH unnested AS (
                SELECT {unnest_exprs},
                    generate_subscripts({cols[0]}, 1) AS idx,
                    experiment_id, variant, lineage_seed, generation, agent_id
                FROM ({subquery})
                WHERE generation >= {gen_floor}
                    AND experiment_id = '{exp_literal}'
            ),
            per_cell AS (
                SELECT idx, {per_cell_aggs}
                FROM unnested
                GROUP BY experiment_id, variant, lineage_seed,
                    generation, agent_id, idx
            )
            SELECT idx, {across_cell_aggs}
            FROM per_cell
            GROUP BY idx
            ORDER BY idx
            """
        ).pl()
    except Exception as e:
        print(
            f"  NOTE: could not read enzyme-kinetics targets for {experiment_id} "
            f"({type(e).__name__}: {e}), so the per-sim kinetic targets omitted from hover data."
        )
        return {}

    if result.height == 0:
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in result.iter_rows(named=True):
        # generate_subscripts is 1 index based, so subtract 1 to index target_rxn_ids
        # and recover which reaction this row's averaged targets belong to:
        i = int(row["idx"]) - 1
        if 0 <= i < len(target_rxn_ids):
            out[target_rxn_ids[i]] = {k: row[k] for k in keys}
    print(
        f"  Kinetic targets read for {experiment_id}: {len(out)} constrained "
        f"reaction(s) (mM/s)."
    )
    return out


def _format_kinetic_block(rxn_id, kinetic_info, targets_1, targets_2):
    """Generates hover text lines describing a reaction's kinetic constraint,
    or '' if the reaction is not kinetically constrained.

    This obtains and formats two pieces of information:
      1. Per-sim target/actual flux from the enzyme-kinetics listener (mM/s,
         same units as the plotted flux).
      2. Static kcat / constraint type / constraining enzyme from sim_data,
         assumed to be identical for both sims.
    """
    # info = static kcat/enzyme (same for both sims); t1/t2 = per-sim listener
    # targets:
    info = kinetic_info.get(rxn_id)
    t1 = targets_1.get(rxn_id)
    t2 = targets_2.get(rxn_id)
    if info is None and t1 is None and t2 is None:
        return ""

    lines = []
    # Part 1: per-sim target flux block, only when a sim reported a target here:
    if t1 is not None or t2 is not None:
        lines.append("Kinetic target flux (mM/s):<br>")

        def _verdict(t):
            # Per-sim judgment of whether the kinetic target was reached,
            # i.e. whether the actual flux landed inside this sim's own
            # [lower, upper] band (if the value is outside the band, the
            # homeostatic objective won over the kinetic target):
            a, lo, hi = t["actual"], t["lower"], t["upper"]
            if any(
                v is None or (isinstance(v, float) and np.isnan(v)) for v in (a, lo, hi)
            ):
                return None
            if lo <= a <= hi:
                return "yes (within bounds)"
            if a > hi:
                return "no, overshoots (homeostatic objective override)"
            return "no, undershoots (homeostatic objective override)"

        def _one(label, t):
            if t is None:
                return f"&nbsp;&nbsp;- {label}: n/a (not constrained in this sim)"
            verdict = _verdict(t)
            reached = (
                f"; kinetic target reached: {verdict}" if verdict is not None else ""
            )
            return (
                f"&nbsp;&nbsp;- {label}: {t['target']:.3e} "
                f"[lo {t['lower']:.3e}, hi {t['upper']:.3e}] "
                f"{reached}"
            )

        lines.append(_one("Sim 1", t1) + "<br>")
        lines.append(_one("Sim 2", t2) + "<br>")

    # Part 2: static kcat / constraint-type / enzyme block, shown whenever
    # structural info exists (this is independent of the per-sim block above):
    if info is not None:
        kmin, kmean, kmax = info["kcat_min"], info["kcat_mean"], info["kcat_max"]
        # min/mean/max collapse to one number whenever only a single kcat was
        # listed for the reaction:
        if kmin == kmax:
            kcat_str = f"{kmean:.4g}"
        else:
            kcat_str = f"{kmean:.4g} (min {kmin:.4g}, max {kmax:.4g})"
        lines.append(f"kcat (1/s): {kcat_str}<br>")
        if info["is_kcat_only"] is not None:
            kind = (
                "kcat-only (no saturation term)"
                if info["is_kcat_only"]
                else "kcat x saturation term"
            )
            lines.append(f"Constraint type: {kind}<br>")
        if info["enzyme"]:
            lines.append(f"Constraining enzyme: {info['enzyme']}<br>")
    return "".join(lines)


def compute_metabolite_flux_totals(shared_avg, stoich):
    """Computes total production and consumption flux of each metabolite across
    every shared reaction in a given simulation.

    For each reaction R (average flux F, always >= 0 since reverse reactions
    are separate ids) and each participant metabolite M with stoichiometric
    coefficient C, the molar rate F*C is production when C > 0 and
    consumption (counted as -F*C) when C < 0. Summed over all reactions this
    gives {M: {"prod": total_produced, "cons": total_consumed}} (mM/s).

    At steady state prod ~= cons for an internal metabolite (that has no mass
    or bulk count tracked). These are the denominators for the hover data's
    per-reaction share (e.g. "reaction Y consumes X% of M's total consumption
    flux"). This share is a metabolite-pool fraction, not an attribution to a
    specific upstream reaction, because the flux through a shared pool cannot
    be traced back to one source.
    """
    totals: dict[str, dict[str, float]] = {}
    for rxn, f in shared_avg.items():
        if not f:
            continue
        for met, coeff in stoich.get(rxn, {}).items():
            rate = f * coeff
            if rate > 0:
                totals.setdefault(met, {"prod": 0.0, "cons": 0.0})["prod"] += rate
            elif rate < 0:
                totals.setdefault(met, {"prod": 0.0, "cons": 0.0})["cons"] += -rate
    return totals


def _format_participants_with_share(
    ids,
    counts_1,
    counts_2,
    rxn_flux_1,
    rxn_flux_2,
    coeffs,
    met_totals_1,
    met_totals_2,
    role,
    ctm_1=None,
    ctm_2=None,
):
    """Constructs the reactant/product hover list.

    This is similar to _format_id_list_with_counts (each id's per-sim
    average bulk count, plus the concentration in parentheses when a
    counts_to_molar scalar is given for that sim; see _format_conc), but
    every metabolite also carries this reaction's share of that metabolite's
    total network flux (per simulation).

    The "role" refers to "cons" for reactants (this reaction consumes them)
    or "prod" for products (it produces them). The per-sim share is
    calculated as (|coeff| * this reaction's avg flux) / metabolite's total
    <role> flux, i.e. the fraction of all flux through that metabolite this
    reaction accounts for.
    """
    ids = list(ids)
    if not ids:
        return ""
    verb = "consumes" if role == "cons" else "produces"
    met_totals_1 = met_totals_1 or {}
    met_totals_2 = met_totals_2 or {}

    def _pct(flux, coeff, totals, met):
        pool = totals.get(met, {}).get(role, 0.0)
        if flux is None or coeff is None or pool <= 0:
            return "n/a"
        return f"{abs(coeff * flux) / pool * 100:.1f}%"

    def _one(met):
        c1, c2 = counts_1.get(met), counts_2.get(met)
        c1s = f"{c1:.3g}{_format_conc(c1, ctm_1)}" if c1 is not None else "n/a"
        c2s = f"{c2:.3g}{_format_conc(c2, ctm_2)}" if c2 is not None else "n/a"
        coeff = coeffs.get(met)
        p1 = _pct(rxn_flux_1, coeff, met_totals_1, met)
        p2 = _pct(rxn_flux_2, coeff, met_totals_2, met)
        # Each molecule gets its own bullet, with a separate line for each
        # sim's stats:
        return (
            f"&nbsp;&nbsp;- {met}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sim1: {c1s}, {verb} {p1} of its total flux<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sim2: {c2s}, {verb} {p2} of its total flux"
        )

    shown = "<br>".join(_one(m) for m in ids[:MAX_HOVER_IDS])
    if len(ids) > MAX_HOVER_IDS:
        shown += f"<br>&nbsp;&nbsp;- (+{len(ids) - MAX_HOVER_IDS} more)"
    return shown


def make_hover_texts(
    rxn_ids,
    f1_signed,
    f2_signed,
    categories,
    molecule_bases,
    kinetic_info=None,
    kinetic_targets_1=None,
    kinetic_targets_2=None,
    shared_avg_1=None,
    shared_avg_2=None,
    flux_unit="mM/s",
    catalyst_counts_1=None,
    catalyst_counts_2=None,
    reactant_counts_1=None,
    reactant_counts_2=None,
    product_counts_1=None,
    product_counts_2=None,
    met_totals_1=None,
    met_totals_2=None,
    ctm_1=None,
    ctm_2=None,
):
    """Generate per-reaction hover text.

    Shows, for each reaction:
      - average flux for both sims (labeled "Sim 1"/"Sim 2", see the plot
        subtitle for which experiment_id each one is),
      - the other direction of a reversible reaction (forward/reverse,
        net/total, near-futile-cycle flag) via _format_direction_block,
      - whether the reaction is kinetically constrained (assuming the same
        reconstruction for both sims), and if so its kcat/constraint-type/
        enzyme info plus each sim's own kinetic listener target (if emitted)
        via _format_kinetic_block,
      - the catalyzing enzyme(s) with each sim's average bulk count,
      - the reactant(s)/product(s), each with both sims' average bulk count
        (when resolvable) and this reaction's share of that metabolite's
        total consumption/production flux across all shared reactions, via
        _format_participants_with_share,
      - and, when molecule_bases is non-empty, which molecule(s) of interest
        this reaction touches and in what role(s) (reactant/product/catalyst).

    The count maps (catalyst_counts_1/2, reactant_counts_1/2,
    product_counts_1/2) are keyed by molecule id to get the average bulk
    count for that sim. shared_avg_1/2 and met_totals_1/2 must be maps over
    ALL shared reactions/metabolites (not just the plotted subset), since the
    direction-partner and flux-share lookups need to see reactions that may
    not themselves be plotted.
    """
    kinetic_set = categories["kinetic_set"]
    catalyst_map = categories["catalyst_map"]
    stoich = categories["stoich"]
    molecule_bases = molecule_bases or set()
    kinetic_info = kinetic_info or {}
    kinetic_targets_1 = kinetic_targets_1 or {}
    kinetic_targets_2 = kinetic_targets_2 or {}
    shared_avg_1 = shared_avg_1 or {}
    shared_avg_2 = shared_avg_2 or {}
    met_totals_1 = met_totals_1 or {}
    met_totals_2 = met_totals_2 or {}
    catalyst_counts_1 = catalyst_counts_1 or {}
    catalyst_counts_2 = catalyst_counts_2 or {}
    reactant_counts_1 = reactant_counts_1 or {}
    reactant_counts_2 = reactant_counts_2 or {}
    product_counts_1 = product_counts_1 or {}
    product_counts_2 = product_counts_2 or {}

    texts = []
    for i, rxn_id in enumerate(rxn_ids):
        a, b = f1_signed[i], f2_signed[i]
        catalysts = catalyst_map.get(rxn_id, [])
        reactants = reaction_reactants(rxn_id, stoich)
        products = reaction_products(rxn_id, stoich)
        constrained = "yes" if rxn_id in kinetic_set else "no"
        kinetic_block = _format_kinetic_block(
            rxn_id, kinetic_info, kinetic_targets_1, kinetic_targets_2
        )
        lines = [
            f"<b>{rxn_id}</b><br>",
            f"Sim 1 avg flux: {a:.3e}<br>",
            f"Sim 2 avg flux: {b:.3e}<br>",
            _format_direction_block(rxn_id, shared_avg_1, shared_avg_2, flux_unit),
            f"Kinetically constrained: {constrained}<br>",
            kinetic_block,
            f"Catalyst avg count(s):<br>"
            f"{_format_id_list_with_counts(catalysts, catalyst_counts_1, catalyst_counts_2, ctm_1, ctm_2) or 'none'}<br>",
            f"Reactant avg count(s) & share of each metabolite's total flux:<br>"
            f"{_format_participants_with_share(reactants, reactant_counts_1, reactant_counts_2, shared_avg_1.get(rxn_id), shared_avg_2.get(rxn_id), stoich.get(rxn_id, {}), met_totals_1, met_totals_2, 'cons', ctm_1, ctm_2) or 'none'}<br>",
            f"Product avg count(s) & share of each metabolite's total flux:<br>"
            f"{_format_participants_with_share(products, product_counts_1, product_counts_2, shared_avg_1.get(rxn_id), shared_avg_2.get(rxn_id), stoich.get(rxn_id, {}), met_totals_1, met_totals_2, 'prod', ctm_1, ctm_2) or 'none'}",
        ]
        if molecule_bases:
            matches = reaction_highlight_matches(
                rxn_id, stoich, catalyst_map, molecule_bases
            )
            if matches:
                match_lines = "<br>".join(
                    f"&nbsp;&nbsp;- {b} (as {', '.join(sorted(roles))})"
                    for b, roles in matches.items()
                )
                lines.append(f"<br>Molecule(s) of interest:<br>{match_lines}")
        texts.append("".join(lines))
    return texts


def _flux_log_transform(sim1_avg, sim2_avg, flux_unit):
    """Calcualtes ε and computes log10(|avg flux| + ε) for both sims.

    If a reaction has zero flux in both simulations passed through, the
    reaction is excluded from the plot. If a reaction has a nonzero in one
    simulation and zero flux in the other, to see that reaction plotted, a tiny
    epsilon (eps) value must be added to the average flux to avoid computing
    log10(0) (which is undefined). Thus, an eps offset is added to |flux|
    before the log so reactions that have zero-flux are still plottable.

    Eps is computed as the smallest nonzero |avg flux| across both sims, so
    zeros sit just below the smallest real point. The eps value sometimes is
    floored in the case where the smallest nonzero flux is more than
    10**MAX_LOG_ORDERS times smaller than the largest. In this case, eps is
    raised to 10**(-MAX_LOG_ORDERS) x the largest |flux| so one near-zero
    outlier can't stretch the axis too heavily.

    Returns (sim1_log, sim2_log, eps, eps_desc).
    - note: eps_desc is a short human label of how the eps value was chosen.
    """
    # Merge the datasets
    both = np.abs(np.concatenate([sim1_avg, sim2_avg]))
    # Find nonzero values of both
    nz = both[both > 0]
    # Compute the floor value based on the max flux value:
    floor = float(both.max()) * 10.0 ** (-MAX_LOG_ORDERS)

    # If there are no reactions with zero flux, eps=0, as no flux should cause
    # an undefined log10() value (note: having no reactions with zero flux would be very rate,
    # typically only ~1900 of ~9500 have nonzero flux in a given sim).
    if not nz.size:
        eps, eps_desc = (
            0.0,
            (
                "a fallback value due to no nonzero flux. "
                "NOTE: this is rare, double check flux values"
                " manually. "
            ),
        )
    # Designate an eps value if there are reactions with nonzero flux values:
    else:
        eps = float(nz.min())
        # check if eps is less than the floor value:
        if eps < floor:
            eps = floor
            eps_desc = (
                f"flooring to {10.0 ** (-MAX_LOG_ORDERS):.0e} x the largest |flux| "
                "(smallest nonzero flux was significantly smaller than the "
                "largest flux, and could mess up the plot if used)."
            )
        else:
            eps_desc = "the smallest nonzero |avg flux|."

    print(
        f"Flux log-transform is computed as: log10(|avg flux| + eps), where "
        f"the shared eps = {eps:.3e} ({flux_unit}). Epsilon value assigned "
        f"via: {eps_desc}"
    )
    sim1_log = np.log10(np.abs(sim1_avg) + eps)
    sim2_log = np.log10(np.abs(sim2_avg) + eps)
    return sim1_log, sim2_log, eps, eps_desc


def _add_parity_line(fig: go.Figure, sim1_log: np.ndarray, sim2_log: np.ndarray):
    """Adds a dashed y=x reference line spanning the data range."""
    min_val = min(float(sim1_log.min()), float(sim2_log.min()))
    max_val = max(float(sim1_log.max()), float(sim2_log.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=0.5),
            name="y=x",
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _add_active_count_legend(fig: go.Figure, active_legend: str):
    """Add a marker-less legend entry reporting how many reactions are plotted
    (active in >=1 sim out of the shared total)."""
    if not active_legend:
        return
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)"),
            name=active_legend,
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _add_stats_annotation(
    fig: go.Figure, r_value: float, pearson_r2: float, cod_r2: float
):
    """Add the Pearson / coefficient of determination (COD) stats box
    (computed over all plotted reactions)."""
    fig.add_annotation(
        x=0.95,
        y=0.05,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Statistics (plotted reactions):</b><br>"
            f"Pearson r = {r_value:.3f}<br>"
            f"Pearson R² = {pearson_r2:.3f}<br>"
            f"COD R² = {cod_r2:.3f}"
        ),
        showarrow=False,
        align="right",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        borderpad=10,
        font=dict(size=11, family="monospace"),
    )


def _apply_square_layout(fig: go.Figure, title: str, xlabel: str, ylabel: str):
    """Square log-axis layout shared by all figures, with the title
    left-aligned and the legend moved outside/right of the plot area (the
    plot is widened and given a right margin so the legend isn't clipped)."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left"),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=1350,
        height=1000,
        margin=dict(r=300),
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            x=1.01,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(constrain="domain")


def _kinetic_symbols_for(idxs, rxn_ids, kinetic_set):
    """Generates a per-point marker symbol array for a list of point indices,
    so one trace keeps a single color/legend entry while each point still
    shows its kinetic-constraint status (square = kcat-constrained, circle =
    everything else)."""
    return ["square" if rxn_ids[i] in kinetic_set else "circle" for i in idxs]


def build_highlighted_figure(
    rxn_ids,
    highlight_list,
    sim1_log,
    sim2_log,
    hover_texts,
    stats,
    title,
    xlabel,
    ylabel,
    active_legend="",
    kinetic_set=frozenset(),
):
    """Scatter with all reactions drawn in one background trace
    (lightseagreen) except a user-specified highlight list, where EACH
    highlighted reaction gets its own trace, its own color (cycling through
    SASHA_PALETTE), and its own legend entry (labeled with its reaction id),
    so several reactions of interest can be told apart at a glance instead of
    all sharing one color.

    Marker shape encodes kinetic-constraint status independently of color
    (square = kcat-constrained, circle = not), applied via a per-point
    marker.symbol array so a trace with mixed kinetic status still keeps one
    legend entry/color.
    """
    fig = go.Figure()
    id_to_pos = {r: i for i, r in enumerate(rxn_ids)}
    highlight_set = set(highlight_list)

    bg_idxs = [i for i, r in enumerate(rxn_ids) if r not in highlight_set]
    if bg_idxs:
        fig.add_trace(
            go.Scatter(
                x=sim1_log[bg_idxs],
                y=sim2_log[bg_idxs],
                mode="markers",
                marker=dict(
                    symbol=_kinetic_symbols_for(bg_idxs, rxn_ids, kinetic_set),
                    color="lightseagreen",
                    size=6,
                    opacity=0.5,
                    line=dict(width=0),
                ),
                name=f"All reactions ({len(bg_idxs)})",
                text=[hover_texts[i] for i in bg_idxs],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    # One trace (and one legend entry/color) per highlighted reaction, cycling
    # through the sasha palette so each stays visually distinct even with
    # several reactions highlighted at once:
    for k, rxn_id in enumerate(highlight_list):
        pos = id_to_pos.get(rxn_id)
        if pos is None:
            # Not found among the plotted (shared & active) reactions; already
            # warned about in plot(), so it's just skipped here silently:
            continue
        color = SASHA_PALETTE[k % len(SASHA_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=[sim1_log[pos]],
                y=[sim2_log[pos]],
                mode="markers",
                marker=dict(
                    symbol=_kinetic_symbols_for([pos], rxn_ids, kinetic_set),
                    color=color,
                    size=11,
                    opacity=0.95,
                    line=dict(width=1, color="black"),
                ),
                name=rxn_id,
                text=[hover_texts[pos]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    _add_parity_line(fig, sim1_log, sim2_log)
    _add_active_count_legend(fig, active_legend)
    _add_stats_annotation(fig, *stats)
    _apply_square_layout(fig, title, xlabel, ylabel)
    return fig


def build_molecule_highlight_figure(
    rxn_ids,
    sim1_log,
    sim2_log,
    hover_texts,
    stoich,
    catalyst_map,
    molecule_bases,
    stats,
    title,
    xlabel,
    ylabel,
    active_legend="",
    kinetic_set=frozenset(),
):
    """Scatter colored by which compartment-tagged form of a requested
    molecule-of-interest each reaction touches, as a reactant, product, or
    catalyst (see discover_highlight_molecule_matches()), instead of a fixed
    two-way category. Non-matching reactions are drawn in one background
    trace (lightseagreen, same base color as build_highlighted_figure's
    background so the two figures read as one family).

    Each discovered compartment-tagged molecule gets its own trace/color
    (cycling through SASHA_PALETTE) and its own legend entry labeled
    "<tagged id> (n=<count>)". A reaction touching two matching tagged
    molecules is drawn once per molecule (overlapping points at the same
    coordinates).

    Marker shape encodes kinetic-constraint status independently of color
    (square = kcat-constrained, circle = not), same as build_highlighted_figure.
    """
    fig = go.Figure()
    tagged_matches = discover_highlight_molecule_matches(
        rxn_ids, stoich, catalyst_map, molecule_bases
    )
    matched_positions: set[int] = set()
    for idxs in tagged_matches.values():
        matched_positions.update(idxs)
    other_idxs = [i for i in range(len(rxn_ids)) if i not in matched_positions]

    if other_idxs:
        fig.add_trace(
            go.Scatter(
                x=sim1_log[other_idxs],
                y=sim2_log[other_idxs],
                mode="markers",
                marker=dict(
                    symbol=_kinetic_symbols_for(other_idxs, rxn_ids, kinetic_set),
                    color="lightseagreen",
                    size=6,
                    opacity=0.5,
                    line=dict(width=0),
                ),
                name=f"Other reactions ({len(other_idxs)})",
                text=[hover_texts[i] for i in other_idxs],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    for k, (tagged_id, idxs) in enumerate(tagged_matches.items()):
        color = SASHA_PALETTE[k % len(SASHA_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=sim1_log[idxs],
                y=sim2_log[idxs],
                mode="markers",
                marker=dict(
                    symbol=_kinetic_symbols_for(idxs, rxn_ids, kinetic_set),
                    color=color,
                    size=10,
                    opacity=0.9,
                    line=dict(width=0.5, color="black"),
                ),
                name=f"{tagged_id} (n={len(idxs)})",
                text=[hover_texts[i] for i in idxs],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    _add_parity_line(fig, sim1_log, sim2_log)
    _add_active_count_legend(fig, active_legend)
    _add_stats_annotation(fig, *stats)
    _apply_square_layout(fig, title, xlabel, ylabel)
    return fig


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_paths: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    """
    # TODO: change comments to say Sim1 and Sim2 instead of full names to make the output less crowded
    Plot average FBA reaction fluxes from two simulations against each other,
    with reactions of interest highlighted (one color each) and one
    additional categorization scheme (molecule_of_interest, colored by
    compartment-tagged form).

    Args:
        params: Dictionary containing parameters of the format::

            {
                # Number of initial generations worth of data to skip
                "skip_n_gens": int (default: 2),
                # Optional override of PLOT_REACTIONS_OF_INTEREST
                "reactions_of_interest": list[str],
                # Base (compartment-stripped) molecule ids whose reactions
                # (as reactant, product, or catalyst) are colored in the
                # by-molecule highlight scheme; every compartment-tagged form
                # actually present is auto-discovered and given its own
                # color/legend entry. "metabolites_of_interest" is accepted
                # as a backward-compat alias for this key.
                "molecules_of_interest": list[str]
                    (default: DEFAULT_METABOLITES_OF_INTEREST),
                # Flux listener column to read (default reaction_fluxes)
                "flux_column": str,
            }
    """
    skip_gens = params.get("skip_n_gens", 2)
    reactions_of_interest = params.get(
        "reactions_of_interest", PLOT_REACTIONS_OF_INTEREST
    )
    # "molecules_of_interest" is the current name; "metabolites_of_interest"
    # is kept as a backward-compat alias so existing configs keep working:
    molecules_of_interest = params.get(
        "molecules_of_interest",
        params.get("metabolites_of_interest", DEFAULT_METABOLITES_OF_INTEREST),
    )
    flux_column = params.get("flux_column", FLUX_COLUMN)
    flux_unit = flux_units(flux_column)

    # "skip the first skip_gens generations of each seed" -> derive the floor
    # from the data's actual minimum generation (generations are 1-indexed), so
    # skip_gens=2 drops exactly the first two generations per seed:
    min_gen = int(
        conn.sql(f"SELECT min(generation) AS g FROM ({history_sql})").pl()["g"][0]
    )
    gen_floor = min_gen + skip_gens

    # Per-reaction average flux (one averaged value per cell, skipping early
    # generations), grouped so each row is one cell's per-reaction average list:
    # TODO: determine if overall time average would be better
    subquery = cast(
        str, read_stacked_columns(history_sql, [flux_column], order_results=False)
    )
    all_fluxes = conn.sql(
        f"""
        WITH unnested_fluxes AS (
            SELECT unnest({flux_column}) AS flux,
                generate_subscripts({flux_column}, 1) AS idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({subquery})
            WHERE generation >= {gen_floor}
        ),
        avg_fluxes AS (
            SELECT avg(flux) AS avgFlux,
                experiment_id, variant, lineage_seed,
                generation, agent_id, idx
            FROM unnested_fluxes
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, idx
        )
        SELECT list(avgFlux ORDER BY idx) AS avgFlux,
               experiment_id
        FROM avg_fluxes
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).pl()

    unique_exp_ids = all_fluxes["experiment_id"].unique().to_list()
    if len(unique_exp_ids) < 2:
        raise ValueError(
            f"Expected 2 experiments but found {len(unique_exp_ids)}: "
            f"{unique_exp_ids}. Make sure both experiment_ids are in the config."
        )

    exp_id_1, exp_id_2 = unique_exp_ids[0], unique_exp_ids[1]

    print(f"Comparing {exp_id_1} (Sim1; x-axis) vs {exp_id_2} (Sim2; y-axis)")

    fluxes_exp1 = all_fluxes.filter(pl.col("experiment_id") == exp_id_1)
    fluxes_exp2 = all_fluxes.filter(pl.col("experiment_id") == exp_id_2)

    sim1_fluxes = ndlist_to_ndarray(fluxes_exp1["avgFlux"])
    sim2_fluxes = ndlist_to_ndarray(fluxes_exp2["avgFlux"])
    sim1_avg_raw = sim1_fluxes.mean(axis=0)
    sim2_avg_raw = sim2_fluxes.mean(axis=0)
    print(f"Sim1 has {len(sim1_fluxes)} cells")
    print(f"Sim2 has {len(sim2_fluxes)} cells")

    # Align reactions by ID, not by list position, using each sim's sim_data:
    exp1_config = f"SELECT * FROM ({config_sql}) WHERE experiment_id = '{exp_id_1}'"
    exp2_config = f"SELECT * FROM ({config_sql}) WHERE experiment_id = '{exp_id_2}'"
    rxn_ids_1 = field_metadata(conn, exp1_config, flux_column)
    rxn_ids_2 = field_metadata(conn, exp2_config, flux_column)
    if len(rxn_ids_1) != len(sim1_avg_raw):
        raise ValueError(
            f"{exp_id_1}: reaction id count ({len(rxn_ids_1)}) does not match "
            f"averaged flux length ({len(sim1_avg_raw)})."
        )
    if len(rxn_ids_2) != len(sim2_avg_raw):
        raise ValueError(
            f"{exp_id_2}: reaction id count ({len(rxn_ids_2)}) does not match "
            f"averaged flux length ({len(sim2_avg_raw)})."
        )

    pos1 = {r: i for i, r in enumerate(rxn_ids_1)}
    pos2 = {r: i for i, r in enumerate(rxn_ids_2)}

    # Sorted intersection is a deterministic, order-independent point list:
    rxn_ids = sorted(pos1.keys() & pos2.keys())
    if not rxn_ids:
        raise ValueError(
            f"No reaction ids shared between {exp_id_1} ({len(rxn_ids_1)} rxns) "
            f"and {exp_id_2} ({len(rxn_ids_2)} rxns); cannot compare."
        )
    sim1_avg = sim1_avg_raw[[pos1[r] for r in rxn_ids]]
    sim2_avg = sim2_avg_raw[[pos2[r] for r in rxn_ids]]

    # Map every shared reaction id (not just the active/plotted subset below)
    # to its avg flux in each sim. The direction-partner and metabolite-flux-
    # share hover machinery need to look up a reaction even when it itself is
    # inactive/unplotted (e.g. an inactive reverse partner of an active
    # forward reaction still needs to be found here):
    shared_avg_1 = {r: sim1_avg_raw[pos1[r]] for r in rxn_ids}
    shared_avg_2 = {r: sim2_avg_raw[pos2[r]] for r in rxn_ids}

    only_in_1 = sorted(pos1.keys() - pos2.keys())
    only_in_2 = sorted(pos2.keys() - pos1.keys())
    print(
        f"Reactions: {len(rxn_ids_1)} in {exp_id_1}, {len(rxn_ids_2)} in "
        f"{exp_id_2}; {len(rxn_ids)} shared (plotted)."
    )
    if only_in_1:
        preview = ", ".join(only_in_1[:10]) + (" ..." if len(only_in_1) > 10 else "")
        print(f"  {len(only_in_1)} only in {exp_id_1} (dropped): {preview}")
    if only_in_2:
        preview = ", ".join(only_in_2[:10]) + (" ..." if len(only_in_2) > 10 else "")
        print(f"  {len(only_in_2)} only in {exp_id_2} (dropped): {preview}")

    # Keep only reactions that actually carry flux: |avg flux| > 0 in at least one
    # sim. NOTE: Most of the ~9k FBA reactions are inactive (exactly zero) and
    # would pile up at the origin, so they are dropped  from the plot. The
    # legend reports how many of the shared reactions remained active:
    n_shared = len(rxn_ids)
    shared_id_set = set(rxn_ids)  # all shared ids, before the active filter
    active = (np.abs(sim1_avg) > 0) | (np.abs(sim2_avg) > 0)
    n_active = int(active.sum())
    print(f"Active reactions (|avg flux| > 0 in >=1 sim): {n_active} of {n_shared}")
    rxn_ids = [r for r, keep in zip(rxn_ids, active) if keep]
    sim1_avg = sim1_avg[active]
    sim2_avg = sim2_avg[active]
    active_legend = f"{n_active} of {n_shared} reactions active in ≥1 sim"

    # Log-transform the magnitude of the average flux (note that a small offset
    # is added to each value to avoid log10(0) calculations, but the hover text
    # retains the true value:
    # TODO: determine if there is a better way to handle the negative magnitude
    #  fluxes so we dont net them, look into how these are analyzed usually in
    #  wcEcoli scripts
    sim1_log, sim2_log, flux_eps, flux_eps_desc = _flux_log_transform(
        sim1_avg, sim2_avg, flux_unit
    )

    # Stats (computed over all plotted reactions):
    r_value = pearsonr(sim1_log, sim2_log)[0]
    pearson_r2 = r_value**2
    cod_r2 = r2_score(sim2_log, sim1_log)
    stats = (r_value, pearson_r2, cod_r2)

    # Category sets/maps from sim_data (ids align with the flux listener ids).
    # This one shared sim_data load is safe for reaction *definitions*
    # (reaction_stoich / reaction_catalysts / kinetic_constraint_reactions /
    # kcats), which are assumed consistent between the two sims being compared.
    # TODO: determine if it is just as fast to do each separately, and if so,
    #  what to do if there are mismatches
    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)
    categories = get_reaction_categories(sim_data, flux_column)
    kinetic_set = categories["kinetic_set"]
    catalyst_map = categories["catalyst_map"]
    stoich = categories["stoich"]
    kinetic_info = get_kinetic_constraint_info(sim_data)

    # Per metabolite total production/consumption flux across all shared
    # reactions (the denominators for the hover's per-reactant/product
    # "consumes/produces X% of this metabolite's total flux" share):
    met_totals_1 = compute_metabolite_flux_totals(shared_avg_1, stoich)
    met_totals_2 = compute_metabolite_flux_totals(shared_avg_2, stoich)

    molecule_bases = {_strip_compartment(m) for m in molecules_of_interest}

    n_kinetic = sum(1 for r in rxn_ids if r in kinetic_set)
    n_catalyzed = sum(1 for r in rxn_ids if r in catalyst_map)
    tagged_preview = discover_highlight_molecule_matches(
        rxn_ids, stoich, catalyst_map, molecule_bases
    )
    n_molecule_matched = (
        len(set().union(*tagged_preview.values())) if tagged_preview else 0
    )
    print(f"Kinetically constrained reactions: {n_kinetic}")
    print(f"Catalyzed reactions: {n_catalyzed}")
    print(
        f"Reactions touching molecule(s) of interest {sorted(molecule_bases)}: "
        f"{n_molecule_matched} ({len(tagged_preview)} distinct compartment-tagged "
        f"form(s) found: {', '.join(tagged_preview) if tagged_preview else 'none'})"
    )

    # Catalyst/reactant average bulk counts per sim for hover text. Only
    # computed for the reactions actually being plotted (the active/filtered
    # rxn_ids above):
    with open_arbitrary_sim_data({exp_id_1: sim_data_paths[exp_id_1]}) as f:
        sim_data_1 = pickle.load(f)
    with open_arbitrary_sim_data({exp_id_2: sim_data_paths[exp_id_2]}) as f:
        sim_data_2 = pickle.load(f)
    bulk_id_to_idx_1 = {
        bid: i
        for i, bid in enumerate(
            sim_data_1.internal_state.bulk_molecules.bulk_data["id"].tolist()
        )
    }
    bulk_id_to_idx_2 = {
        bid: i
        for i, bid in enumerate(
            sim_data_2.internal_state.bulk_molecules.bulk_data["id"].tolist()
        )
    }

    # Distinct catalyst + reactant + product ids across only the plotted
    # reactions:
    hover_molecule_ids: set[str] = set()
    for r in rxn_ids:
        hover_molecule_ids.update(catalyst_map.get(r, []))
        hover_molecule_ids.update(reaction_reactants(r, stoich))
        hover_molecule_ids.update(reaction_products(r, stoich))

    idx_map_1, missing_1 = bulk_ids_to_indices(hover_molecule_ids, bulk_id_to_idx_1)
    idx_map_2, missing_2 = bulk_ids_to_indices(hover_molecule_ids, bulk_id_to_idx_2)
    if missing_1:
        preview = ", ".join(sorted(missing_1)[:10]) + (
            " ..." if len(missing_1) > 10 else ""
        )
        print(
            f"  WARNING: {len(missing_1)} catalyst/reactant/product id(s) not "
            f"found in {exp_id_1}'s bulk container (hover count omitted): {preview}"
        )
    if missing_2:
        preview = ", ".join(sorted(missing_2)[:10]) + (
            " ..." if len(missing_2) > 10 else ""
        )
        print(
            f"  WARNING: {len(missing_2)} catalyst/reactant/product id(s) not "
            f"found in {exp_id_2}'s bulk container (hover count omitted): {preview}"
        )

    means_1 = read_bulk_means_for_experiment(
        conn, history_sql, exp_id_1, list(idx_map_1.values()), gen_floor
    )
    means_2 = read_bulk_means_for_experiment(
        conn, history_sql, exp_id_2, list(idx_map_2.values()), gen_floor
    )
    # mol id to average count mapping, for whichever ids resolved to a bulk index in
    # each sim (ids missing from a sim's bulk container are simply absent from
    # that sim's map, and make_hover_texts renders them as 'n/a'):
    # TODO: determine if this is the best way to handle this going forward
    counts_1 = {mol_id: means_1[idx] for mol_id, idx in idx_map_1.items()}
    counts_2 = {mol_id: means_2[idx] for mol_id, idx in idx_map_2.items()}

    # Per-sim average counts_to_molar scalar (mM per count), used to annotate
    # each hover count with its concentration in parentheses (conc_mM = count *
    # ctm). None if a sim did not emit the enzyme-kinetics listener (hover then
    # shows bare counts for that sim):
    ctm_1 = read_counts_to_molar_mean_for_experiment(
        conn, history_sql, exp_id_1, gen_floor
    )
    ctm_2 = read_counts_to_molar_mean_for_experiment(
        conn, history_sql, exp_id_2, gen_floor
    )
    print(
        f"counts_to_molar (mM/count) for hover concentrations: "
        f"Sim1={ctm_1 if ctm_1 is None else f'{ctm_1:.3e}'}, "
        f"Sim2={ctm_2 if ctm_2 is None else f'{ctm_2:.3e}'}"
    )

    # Per-sim kinetic listener targets for the hover's kinetic block (empty
    # dict if a sim never emitted the enzyme-kinetics listener):
    print("\nKINETIC TARGET DATA:")
    kinetic_targets_1 = read_kinetic_targets_for_experiment(
        conn, history_sql, config_sql, exp_id_1, gen_floor
    )
    kinetic_targets_2 = read_kinetic_targets_for_experiment(
        conn, history_sql, config_sql, exp_id_2, gen_floor
    )
    if not kinetic_targets_1 and not kinetic_targets_2:
        print(
            "  Neither sim emitted enzyme-kinetics target fluxes; hover will "
            "show only the static kcat block for kinetically constrained "
            "reactions."
        )

    hover_texts = make_hover_texts(
        rxn_ids,
        sim1_avg,
        sim2_avg,
        categories,
        molecule_bases,
        kinetic_info=kinetic_info,
        kinetic_targets_1=kinetic_targets_1,
        kinetic_targets_2=kinetic_targets_2,
        shared_avg_1=shared_avg_1,
        shared_avg_2=shared_avg_2,
        flux_unit=flux_unit,
        catalyst_counts_1=counts_1,
        catalyst_counts_2=counts_2,
        reactant_counts_1=counts_1,
        reactant_counts_2=counts_2,
        product_counts_1=counts_1,
        product_counts_2=counts_2,
        met_totals_1=met_totals_1,
        met_totals_2=met_totals_2,
        ctm_1=ctm_1,
        ctm_2=ctm_2,
    )

    def extract_short_id(exp_id):
        """Extract a short identifier from the full experiment ID."""
        parts = exp_id.split("_")
        for i, part in enumerate(parts):
            if "-" in part and len(part) == 15:
                # timestamp format YYYYMMDD-HHMMSS
                return "_".join(parts[:i])
        return exp_id

    sim1_short = extract_short_id(exp_id_1)
    sim2_short = extract_short_id(exp_id_2)

    comparison_outdir = outdir + f"_{exp_id_1}_vs_{exp_id_2}"
    os.makedirs(comparison_outdir, exist_ok=True)

    main_title = "FBA Flux Comparison"
    subtitle = (
        f"<sub>Sim 1 (x): {exp_id_1} (averaged over {len(sim1_fluxes)} cells) vs. "
        f"<br>Sim 2 (y): {exp_id_2} (averaged over {len(sim2_fluxes)} cells)"
        f"<br>First {skip_gens} generations of each seed are excluded in averages"
        f"<br>ε={flux_eps:.1e} added to each average to avoid log10(0)</sub>"
    )
    xlabel = f"log10(|Sim 1 avg flux| + {flux_eps:.1e}) ({flux_unit})"
    ylabel = f"log10(|Sim 2 avg flux| + {flux_eps:.1e}) ({flux_unit})"

    # Plot 1: highlighted reactions of interest (each its own color/trace):
    active_id_set = set(rxn_ids)  # active (plotted) shared ids
    print("\nHIGHLIGHT REACTIONS:")
    for r in reactions_of_interest:
        if r not in shared_id_set:
            print(f"  WARNING: '{r}' not found in '{flux_column}' for these sims.")
        elif r not in active_id_set:
            print(
                f"  NOTE: '{r}' has zero avg flux in both sims so "
                f"it is filtered out and will not appear on the plot."
            )
        else:
            print(f"  + '{r}' highlighted.")
    n_highlighted = sum(1 for r in reactions_of_interest if r in active_id_set)
    fig_hi = build_highlighted_figure(
        rxn_ids,
        reactions_of_interest,
        sim1_log,
        sim2_log,
        hover_texts,
        stats,
        f"{main_title} -- Highlighted Reactions<br>{subtitle}",
        xlabel,
        ylabel,
        active_legend,
        kinetic_set=kinetic_set,
    )
    hi_filename = os.path.join(
        comparison_outdir,
        f"fba_flux_comparison_plotly_highlighted_{sim1_short}_vs_{sim2_short}.html",
    )
    fig_hi.write_html(hi_filename)
    print(
        f"Highlighted reactions: {n_highlighted} of "
        f"{len(reactions_of_interest)} requested"
    )

    # Plot 2: molecule-of-interest highlight (one color per compartment-tagged
    # form actually found among the plotted reactions):
    molecules_title = (
        f"{main_title} -- Molecules of Interest "
        f"<br>(base id(s): {', '.join(sorted(molecule_bases))})<br>{subtitle}"
    )
    fig_mol = build_molecule_highlight_figure(
        rxn_ids,
        sim1_log,
        sim2_log,
        hover_texts,
        stoich,
        catalyst_map,
        molecule_bases,
        stats,
        molecules_title,
        xlabel,
        ylabel,
        active_legend,
        kinetic_set=kinetic_set,
    )
    mol_filename = os.path.join(
        comparison_outdir,
        f"fba_flux_comparison_plotly_molecules_{sim1_short}_vs_{sim2_short}.html",
    )
    fig_mol.write_html(mol_filename)

    # ---- CSV export: reactions of interest ----
    # One row per reaction the user listed (reactions_of_interest), with each
    # sim's average flux, the catalyzing enzyme concentration(s), and the average
    # bulk counts of every reactant and product. Multiple participants (enzymes,
    # reactants, products) are listed together in a single cell as
    # "id: value, id: value". Participant counts for the LISTED reactions are read
    # directly here (not reused from the plotted-subset counts_1/2) so a reaction
    # of interest that is inactive in both sims -- and thus filtered out of the
    # plot -- still gets its enzyme/reactant/product values in the CSV.
    csv_rxn_ids = list(dict.fromkeys(reactions_of_interest))  # dedupe, keep order

    # Gather every catalyst/reactant/product id across the listed reactions, then
    # resolve + read their per-sim average bulk counts (same two-stage averaging
    # as the hover counts):
    csv_mol_ids: set[str] = set()
    for r in csv_rxn_ids:
        csv_mol_ids.update(catalyst_map.get(r, []))
        csv_mol_ids.update(reaction_reactants(r, stoich))
        csv_mol_ids.update(reaction_products(r, stoich))
    csv_idx_1, _ = bulk_ids_to_indices(csv_mol_ids, bulk_id_to_idx_1)
    csv_idx_2, _ = bulk_ids_to_indices(csv_mol_ids, bulk_id_to_idx_2)
    csv_means_1 = read_bulk_means_for_experiment(
        conn, history_sql, exp_id_1, list(csv_idx_1.values()), gen_floor
    )
    csv_means_2 = read_bulk_means_for_experiment(
        conn, history_sql, exp_id_2, list(csv_idx_2.values()), gen_floor
    )
    csv_counts_1 = {m: csv_means_1[i] for m, i in csv_idx_1.items()}
    csv_counts_2 = {m: csv_means_2[i] for m, i in csv_idx_2.items()}

    def _flux_for(rxn, avg_raw, pos):
        # That reaction's own average flux in one sim, or None if the reaction
        # is not present in that sim's flux listener (unshared):
        return float(avg_raw[pos[rxn]]) if rxn in pos else None

    def _conc_value_str(count, ctm):
        # Concentration string only (no parentheses), auto-scaled mM/µM/nM.
        # None when count or ctm is missing (conc cannot be formed).
        if count is None or ctm is None:
            return None
        conc_mM = count * ctm
        if conc_mM >= 1.0:
            return f"{conc_mM:.3g} mM"
        conc_uM = conc_mM * 1e3
        if conc_uM >= 1.0:
            return f"{conc_uM:.3g} µM"
        return f"{conc_mM * 1e6:.3g} nM"

    def _enzyme_cell(rxn, counts, ctm):
        # "enzyme: conc" per catalyst, joined; falls back to a bare count when
        # counts_to_molar is unavailable, or n/a when the count is unresolved:
        cats = catalyst_map.get(rxn, [])
        if not cats:
            return "none"
        parts = []
        for c in cats:
            cnt = counts.get(c)
            conc = _conc_value_str(cnt, ctm)
            if conc is not None:
                parts.append(f"{c}: {conc}")
            elif cnt is not None:
                parts.append(f"{c}: {cnt:.3g} counts (no counts_to_molar)")
            else:
                parts.append(f"{c}: n/a")
        return ", ".join(parts)

    def _conc_cell(ids, counts, ctm):
        # "id: conc" per participant, joined; falls back to a bare count when
        # counts_to_molar is unavailable, or n/a when the count is unresolved:
        if not ids:
            return "none"
        parts = []
        for m in ids:
            cnt = counts.get(m)
            conc = _conc_value_str(cnt, ctm)
            if conc is not None:
                parts.append(f"{m}: {conc}")
            elif cnt is not None:
                parts.append(f"{m}: {cnt:.3g} counts (no counts_to_molar)")
            else:
                parts.append(f"{m}: n/a")
        return ", ".join(parts)

    csv_rows = []
    for rxn in csv_rxn_ids:
        f1 = _flux_for(rxn, sim1_avg_raw, pos1)
        f2 = _flux_for(rxn, sim2_avg_raw, pos2)
        reactants = reaction_reactants(rxn, stoich)
        products = reaction_products(rxn, stoich)
        csv_rows.append(
            {
                "reaction_id": rxn,
                "kinetically_constrained": "yes" if rxn in kinetic_set else "no",
                f"{sim1_short} avg flux ({flux_unit})": (
                    f"{f1:.6g}" if f1 is not None else "n/a"
                ),
                f"{sim2_short} avg flux ({flux_unit})": (
                    f"{f2:.6g}" if f2 is not None else "n/a"
                ),
                f"{sim1_short} enzyme conc": _enzyme_cell(rxn, csv_counts_1, ctm_1),
                f"{sim2_short} enzyme conc": _enzyme_cell(rxn, csv_counts_2, ctm_2),
                f"{sim1_short} reactant conc": _conc_cell(
                    reactants, csv_counts_1, ctm_1
                ),
                f"{sim2_short} reactant conc": _conc_cell(
                    reactants, csv_counts_2, ctm_2
                ),
                f"{sim1_short} product conc": _conc_cell(products, csv_counts_1, ctm_1),
                f"{sim2_short} product conc": _conc_cell(products, csv_counts_2, ctm_2),
            }
        )

    csv_filename = os.path.join(
        comparison_outdir,
        f"fba_flux_comparison_reactions_of_interest_{sim1_short}_vs_{sim2_short}.csv",
    )
    if csv_rows:
        pl.DataFrame(csv_rows).write_csv(csv_filename)
        print(f"Saved reactions-of-interest CSV to {csv_filename}")
    else:
        print("NOTE: no reactions of interest listed, so no CSV was written.")

    # Override the default metadata saving file path:
    return {"metadata_path": comparison_outdir}
