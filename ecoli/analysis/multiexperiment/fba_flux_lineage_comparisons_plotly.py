"""
Multiexperiment FBA flux lineage comparison (plotly scatters)

Compares average FBA reaction fluxes between two simulations, but
unlike fba_flux_comparison_plotly.py, which plots every shared reaction, this
plot restricts each figure to the metabolic "lineage" of one center reaction:
the reactions reachable from it by following shared metabolites through the
FBA network. For each requested reaction of interest (ROI), two directions are
traced independently via a breadth-first search (BFS) over reaction_stoich:
  - upstream (ancestors): level k+1 = reactions that PRODUCE the reactants of
    level k, i.e. what feeds the center.
  - downstream (descendants): level k+1 = reactions that CONSUME the products of
    level k, i.e. what the center feeds.

A default list of currency/connector metabolites (WATER, ATP, PROTON, ...)
are excluded from the traversal so the graph does not explode into the whole
network (see DEFAULT_EXCLUDED_CONNECTOR_METABOLITES & resolve_excluded_metabolites()).

For every (center reaction, direction) pair, up to two HTML scatters are written
into a per-comparison output directory:
# TODO: update these with the finalized names when done editing this script
  1. fba_flux_lineage_comparison_<direction>_<rxn>_<exp1>_vs_<exp2>.html:
     log-log scatter of |avg flux| (sim1 on x, sim2 on y) for the lineage
     reactions, colored by BFS level (rainbow, level 0 = center as a red star)
     and shaped by kinetic-constraint status (square = kcat-constrained, circle
     = everything else).
  2. ..._<direction>_unique_<rxn>_...html: the same figure as 1 but filtered to
     "base unique" reactions, i.e. rxns whose base reaction lands on a single BFS
     level (see UNIQUE PLOTS section for more info).

Reactions outside the traced lineage are not plotted and lineage reactions with
no shared flux data between the two sims are dropped with a warning. Axes are
log10(|avg flux| + eps) so zero-flux reactions remain plottable (see
_flux_log_transform()).

Background on json config options and how to use them (see plot() for more
technical detail):

The BFS walks the network by following metabolites shared between reactions.
"Connector"/currency metabolites (WATER, ATP, PROTON, ...) are excluded from
that walk by default in DEFAULT_EXCLUDED_CONNECTOR_METABOLITES, otherwise the
lineage explodes into essentially the whole network. The next three options
tune that exclusion set (see resolve_excluded_metabolites for how they combine).

THINGS TO ADD TO THE DOCSTRING STILL:
- info on how the algorithm explores
- provide a detailed explaination for the three ways in which a kinetic reaction's
    target flux range is defined based on available kcat and Km data
- info on each config option and how they can be used to filter whats ploted
- warnings on how filtering can potentially cause important recations to be
 dropped from the plot so just be careful
- explaination as to how reversible reactions are plotted (and net fluxes of a reversible reaction is not plotted)
- explaination on how different reaction flux IDs are derived from base ids (since it can be confusing)
    ^ actually, double check how common it is for different versions of reactions
     to show up in one sim and not another, because maybe we would want zeros plotted for the other sim in this case.
     could probably check this easily by just doing something like set(sim 1 reaction IDs + sim 2 reaction IDs)?

# TODOs:
# TODO: finish making detailed docstring above
# TODO: consider making the unique plots a config option
# TODO: maybe exclude reactions that are 0 in both from the plot (and have how many are on each level in the legend instead?)
# TODO: consider moving all user inputs below into config options? but then again, i like these ones here
# TODO: make it clear that the count of any monomers or complexes plotted is the
free count (that is available to participate in metabolic reactions) in the docstrings.

"""

import os
import pickle
import re
from itertools import product
from typing import Any, cast

import matplotlib.colors as mcolors
import numpy as np
import polars as pl
from duckdb import DuckDBPyConnection
import plotly.colors as pc
import plotly.graph_objects as go

from ecoli.library.parquet_emitter import (
    field_metadata,
    named_idx,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
    read_stacked_columns,
)
from reconstruction.ecoli.dataclasses.process.metabolism import REVERSE_TAG

# PLOT DEFAULTS:

# Currency/connector metabolites excluded from lineage search when building
# the producer/consumer indexes. Most of these are universal energy/proton
# related and each appear in hundreds of reactions (across essentially every
# pathway). If lineages were traced with these included, the graph would get
# more crowded. Overridable via ``params["excluded_connector_metabolites"]``:
DEFAULT_EXCLUDED_CONNECTOR_METABOLITES = {
    "WATER",
    "ATP",
    "ADP",
    "AMP",
    "Pi",
    "PPI",
    "PROTON",
    "NAD",
    "NADH",
    "NADP",
    "NADPH",
    "CO-A",
    "CARBON-DIOXIDE",
    "OXYGEN-MOLECULE",
}

# Cap on BFS depth (overridable via params["max_levels"]):
DEFAULT_MAX_LEVELS = 20

# Floor for the log offset so that epsilon is never smaller than
# 10**(-MAX_LOG_ORDERS) times the largest |avg flux|. Without this, a single
# near-zero flux would drag epsilon down and stretch the axis into a huge empty
# low region (It does not normally set epsilon if not needed):
MAX_LOG_ORDERS = 6.0

# This |net| / total (|positive| or |negative|) ratio a forward/reverse pair is
# flagged in the hover data  as a "near-futile cycle" when both directions of
# a reversible reaction carry real flux, but they nearly cancel, so the pair
# moves almost no net mass. This is added because the magnitude of each
# direction is plotted separately (and not together as a net flux because
# otherwise so much would cancel to zero), and this helps point out that
# in actuality, the other direction may be canceling out the total flux in
# each direction. Change the value to indicate a different bound for
# near-futile cycles:
FUTILE_NET_FRACTION = 0.1

# Cap for long id lists (like the descendent/ancestor lists and reactant/product
# lists) shown in hover text (show first N, then "(+M more)"):
MAX_HOVER_IDS = 5

# END PLOT DEFAULTS


# FUNCTIONS:
def _safe_filename_part(text: str) -> str:
    """
    Makes a string safe to embed a reaction ID containing ``/`` characters
    into a filename.
    """
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)


def _ordinal(n):
    """
    Returns the ordinal spelling of n ("1st", "2nd", ..., "5th") for the
    per-level legend labels (e.g. "3rd level (k)").
    """
    # 1, 2, and 3 need special spellings, everything else gets  "th":
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


def flux_units(flux_column: str) -> str:
    """Returns the flux unit string for a given FBA flux listener column."""
    leaf = flux_column.rsplit("__", 1)[-1]
    if leaf in ("estimated_fluxes", "base_reaction_fluxes"):
        return "counts/timestep"
    return "mM/s"


def _strip_compartment(mol_id: str) -> str:
    """
    Removes a trailing compartment tag like ``[c]`` from a molecule id.
    """
    if mol_id.endswith("]") and len(mol_id) >= 3 and mol_id[-3] == "[":
        return mol_id[:-3]
    return mol_id


def reaction_reactants(rxn_id, stoich):
    """
    Returns the list of reactant participant ids for a reaction, in
    ``stoich`` dict order.
    """
    return [met_id for met_id, coeff in stoich.get(rxn_id, {}).items() if coeff < 0]


def reaction_products(rxn_id, stoich):
    """
    Returns the list of product participant ids for a reaction, in ``stoich``
    dict order.
    """
    return [met_id for met_id, coeff in stoich.get(rxn_id, {}).items() if coeff > 0]


def highlight_base_order(molecules_of_interest):
    """
    Generates a stable ordering of the compartment-stripped base molecule ids
    for the legend (uses the compartment-stripped molecule ids, de-duplicated
    but keeping the input order).

    This order fixes each molecule's legend color and its priority when a
    reaction contains more than one molecule of interest (the reaction is drawn
    once per matched molecule, so priority only affects overlap draw order).
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for m in molecules_of_interest:
        b = _strip_compartment(m)
        if b not in seen:
            seen.add(b)
            ordered.append(b)
    return ordered


def reaction_highlight_matches(rxn_id, stoich, catalyst_map, base_set):
    """
    Maps each molecule-of-interest base a reaction contains to the set of roles
    it plays in that reaction (reactant, product, or catalyst).
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


def _lineage_reactants(rxn_id, stoich, excluded_metabolites=frozenset()):
    """
    Returns the reactant ids for a reaction, skipping any whose
    compartment-stripped base id is in ``excluded_metabolites``.

    The compartment tag is kept so that SUC[c] and SUC[p] for example,
    are treated as distinct metabolites for graph-traversal purposes
    (only the exclusion check operates on the stripped base id).
    """
    return [
        met_id
        for met_id, coeff in stoich.get(rxn_id, {}).items()
        if coeff < 0 and _strip_compartment(met_id) not in excluded_metabolites
    ]


def _lineage_products(rxn_id, stoich, excluded_metabolites=frozenset()):
    """
    Return the product ids for a reaction, skipping any whose
    compartment-stripped base id is in ``excluded_metabolites``.
    """
    return [
        met_id
        for met_id, coeff in stoich.get(rxn_id, {}).items()
        if coeff > 0 and _strip_compartment(met_id) not in excluded_metabolites
    ]


def build_producer_consumer_index(reaction_stoich, excluded_metabolites):
    """
    Builds ``metabolite_id -> set of reaction ids`` indexes.
    Builds one for producers and one for consumers, keyed by the
    compartment-tagged metabolite id and skipping any whose
    compartment-stripped base id is in ``excluded_metabolites``.
    """
    producers: dict[str, set[str]] = {}
    consumers: dict[str, set[str]] = {}
    for rxn_id, participants in reaction_stoich.items():
        for met_id, coeff in participants.items():
            # Exclusion is decided on the base id, but the index is keyed by the
            # compartment-tagged id, so SUC[c] and SUC[p] stay separate keys:
            base = _strip_compartment(met_id)
            if base in excluded_metabolites:
                continue
            # Sign of the coefficient decides producer vs consumer:
            if coeff < 0:
                consumers.setdefault(met_id, set()).add(rxn_id)
            elif coeff > 0:
                producers.setdefault(met_id, set()).add(rxn_id)
    return producers, consumers


def metabolite_reaction_degree(reaction_stoich):
    """
    Return ``{base_metabolite_id: number_of_reactions_it_appears_in}`` over
    ``reaction_stoich``.
    """
    degree: dict[str, int] = {}
    for participants in reaction_stoich.values():
        # Per-reaction seen set so a metabolite present in several compartments
        # (or on both sides) still counts this reaction only once toward its degree:
        seen: set[str] = set()
        for met_id in participants:
            base = _strip_compartment(met_id)
            if base in seen:
                continue
            seen.add(base)
            degree[base] = degree.get(base, 0) + 1
    return degree


def high_degree_metabolites(reaction_stoich, threshold):
    """
    Finds base metabolite ids that appear in strictly more than ``threshold``
    reactions (set at the top in PLOT DEFAULTS) reactions. Basically, this
    catches common connector metabolites (So this catches molecules that are
    not in the ``DEFAULT_EXCLUDED_CONNECTOR_METABOLITES`` list but does appear
    in so many reactions that it can bring in so many unique reactions that
    end up overcrowding the plot if reactions connected only by this metabolite
    are allowed to stay in).
    """
    return {
        met
        for met, deg in metabolite_reaction_degree(reaction_stoich).items()
        if deg > threshold
    }


def resolve_excluded_metabolites(params, reaction_stoich):
    """
    Combines the curated/overridden connector-metabolite exclusion set with an
    optional automatic frequency-based exclusion, then re-includes any
    metabolites the caller explicitly wants kept.

    - ``params["excluded_connector_metabolites"]``: overrides the curated
      ``DEFAULT_EXCLUDED_CONNECTOR_METABOLITES`` (NOTE: if one were hoping to
      add more to the list, they would have to paste in the molecules from the
      current default list and add the new ones to this).
      # TODO: consider changing this config option that is just called
      "add_extra_exclusions" so the whole list
       would not need to be repasted in the config just to change whats on it.
    - ``params["exclude_metabolite_degree_over"]``: additionally excludes every
      metabolite that appears in more than that many reactions.
    - ``params["always_include_metabolites"]``: a list of compartment-stripped
      metabolite ids to be subtracted back out of the final exclusion set, so
      a metabolite named here is never excluded even if the
      curated list or the degree threshold would otherwise drop it.

    Returns ``(excluded_set, auto_excluded_set, readded_set)``:
      - excluded_set: the final base ids to exclude from traversal.
      - auto_excluded_set: the frequency-based additions that survived
        re-inclusion.
      - readded_set: the always-include ids that had actually been excluded
        before being re-added (ids that weren't excluded anyway
        are not listed).
    """
    # Start from the curated connector list (or its param override):
    excluded = set(
        params.get(
            "excluded_connector_metabolites", DEFAULT_EXCLUDED_CONNECTOR_METABOLITES
        )
    )
    threshold = params.get("exclude_metabolite_degree_over")
    auto: set[str] = set()
    # Fold in every metabolite exceeding the degree threshold too:
    if threshold is not None:
        auto = high_degree_metabolites(reaction_stoich, int(threshold))
        excluded |= auto

    # Subtract it back out of both the final exclusion set and the auto set if
    # there are molecules to always be included. Readded records only those
    # that had actually been excluded (intersection before subtraction):
    always_include = set(params.get("always_include_metabolites", ()))
    readded = always_include & excluded
    excluded -= always_include
    auto -= always_include
    return excluded, auto, readded


def compute_lineage_levels(
    reaction_stoich,
    center_reaction_id,
    direction,
    excluded_metabolites=DEFAULT_EXCLUDED_CONNECTOR_METABOLITES,
    max_levels=DEFAULT_MAX_LEVELS,
):
    """
    Computes linage levels using a Breadth-First Search (BFS) level-assignment
    method (over the FBA reaction network), so that it is decoupled from any
    DuckDB/plotly/params and can be unit-tested directly against
    ``reaction_stoich`` (e.g. loaded from ``simData.cPickle``).

    Args:
        reaction_stoich: ``{reaction_id: {metabolite_id_with_compartment:
            signed_coeff}}``, i.e. ``sim_data.process.metabolism.reaction_stoich``.
        center_reaction_id: the reaction to trace ancestry/descendants from.
        direction: ``"upstream"`` (ancestors: level k+1 = producers of level
            k's reactants) or ``"downstream"`` (descendants: level k+1 =
            consumers of level k's products).
        excluded_metabolites: base (compartment tag stripped) metabolite ids to
            ignore when looking for shared-metabolite edges.
        max_levels: safety cap on BFS depth (Note: this seems to be hit for
        most reactions...).

    Returns:
        levels: ``{level: [reaction_ids]}``, level 0 = ``[center_reaction_id]``.
            A reaction appears in exactly one level (its shallowest).
        edges: ``[(child_reaction, parent_reaction, connecting_metabolite),
            ...]``, where child is in level k+1, parent is the specific level-k
            reaction that discovered it, and connecting_metabolite is the
            non-excluded (base) metabolite whose sharing created the edge (child
            produces it / parent consumes it upstream; parent produces it /
            child consumes it downstream). A child can have multiple parent
            edges if several level-k reactions share a metabolite with it, or if
            the same parent shares more than one metabolite.
        hit_cap: True if ``max_levels`` was reached before the BFS terminated
            naturally (an empty next level). If this is hit, the result is a
            known-incomplete/truncated ancestry (noted in the plot).
    """
    # Affirm reaction is valid:
    if center_reaction_id not in reaction_stoich:
        raise ValueError(
            f"Reaction id {center_reaction_id!r} not found in reaction_stoich."
        )
    # Check direction is valid (this should always pass):
    if direction not in ("upstream", "downstream"):
        raise ValueError(
            f"direction must be 'upstream' or 'downstream', got {direction!r}"
        )

    producers, consumers = build_producer_consumer_index(
        reaction_stoich, excluded_metabolites
    )
    # The two directions differ only in 1. which side of each reaction supplies
    # the boundary metabolites and 2. which index (producers vs consumers) is
    # searched for the neighbors (the BFS loop below is otherwise identical):
    if direction == "upstream":
        # Ancestors: look at each level-k reaction's reactants, and find who
        # produces those metabolites:
        def get_boundary_metabolites(rxn):
            return _lineage_reactants(rxn, reaction_stoich, excluded_metabolites)

        neighbor_index = producers
    else:
        # Descendants: look at each level-k reaction's products, and find who
        # consumes those metabolites:
        def get_boundary_metabolites(rxn):
            return _lineage_products(rxn, reaction_stoich, excluded_metabolites)

        neighbor_index = consumers

    # levels hold the final assignment (and doubles as the BFS visited-set
    # and the level lookup, so a reaction already in assigned is never revisited
    # (guaranteeing each reaction lands on exactly one (its shallowest) level)):
    levels: dict[int, list[str]] = {0: [center_reaction_id]}
    assigned: dict[str, int] = {center_reaction_id: 0}
    edges: list[tuple[str, str]] = []
    hit_cap = False

    level = 0
    current_level_rxns = [center_reaction_id]
    while current_level_rxns:
        # Stop before growing past the depth cap and flag the result as truncated:
        if level + 1 > max_levels:
            hit_cap = True
            print(
                f"WARNING: reaction_lineage_network ({direction}) hit max_levels="
                f"{max_levels} before terminating naturally. Ancestry is "
                f"truncated/incomplete beyond level {level}, consider "
                f"increasing 'max_levels' in plot script if interested in "
                f"viewing more."
            )
            break
        next_level_set: set[str] = set()
        level_edges: list[tuple[str, str]] = []
        # Expand the current search area: for each boundary metabolite of each
        # frontier reaction, every reaction on the other side of that metabolite
        # is a candidate neighbor. Skips self-loops and already-assigned
        # reactions, and records an edge (neighbor, discovering rxn, connecting
        # metabolite) even when the neighbor is reached via several
        # metabolites/parents:
        for rxn in current_level_rxns:
            for met in get_boundary_metabolites(rxn):
                for neighbor in neighbor_index.get(met, ()):
                    if neighbor == rxn or neighbor in assigned:
                        continue
                    next_level_set.add(neighbor)
                    level_edges.append((neighbor, rxn, met))
        # Natural termination (nothing new was discovered at this depth):
        if not next_level_set:
            break
        level += 1
        # Commit the new frontier (sorted for determinism, mark each as assigned
        # to this level so deeper passes treat them as visited, then advance):
        levels[level] = sorted(next_level_set)
        for r in next_level_set:
            assigned[r] = level
        edges.extend(level_edges)
        current_level_rxns = levels[level]

    return levels, edges, hit_cap


def build_parent_links(edges):
    """
    Using ``compute_lineage_levels`` edges ``[(child, parent, metabolite),
    ...]``, builds ``{child_reaction: [(parent_reaction, connecting_metabolite),
    ...]}`` (duplicates removed, discovery order preserved). The connecting
    metabolite is the non-excluded metabolite whose sharing created the edge.
    """
    links: dict[str, list[tuple[str, str]]] = {}
    for child, parent, met in edges:
        # Invert the edge list into a map from child reaction to its parent
        # metabolite links (the membership check drops duplicate (parent, met)
        # pairs while append keeps the original BFS discovery order):
        lst = links.setdefault(child, [])
        if (parent, met) not in lst:
            lst.append((parent, met))
    return links


def format_lineage_link_lines(rxn_id, parent_links, direction, cap=MAX_HOVER_IDS):
    """
    Hover text line(s) naming the parent reaction(s) that discovered ``rxn_id``
    in the BFS and the metabolite each reaction is linked to the ROI by.
    Returns "" for the center reaction (which has no parents).

    If upstream, then the reaction produces a metabolite that each listed
    parent consumes. If downstream, this reaction consumes a metabolite that
    each listed parent produces.

    This gets capped at max hover IDs to try to avoid overcrowding in the
    hover text, but it can be overridden by passing a different ``cap`` value
    in the defaults at the top of this file.
    """
    # If the reaction has no parents, the center reaction (level 0), prints no link line:
    links = parent_links.get(rxn_id, [])
    if not links:
        return ""
    # Select wording based off the relationship to the reaction of interest:
    header = (
        "Ancestor of (produces their reactant):"
        if direction == "upstream"
        else "Descendant of (consumes their product):"
    )
    # Cap the bullet list like the other hover id lists, adding a "(+N more)" tail:
    bullets = [f"&nbsp;&nbsp;- {parent} via {met}" for parent, met in links[:cap]]
    if len(links) > cap:
        bullets.append(f"&nbsp;&nbsp;- (+{len(links) - cap} more)")
    return header + "<br>" + "<br>".join(bullets)


def level_colorscale(colors, n_levels):
    """
    Samples ``n_levels`` colors along a sequential colorscale built from
    ``colors``, one per level. Built as a real sampled colorscale since the
    number of levels is data-dependent.

    ``plotly.colors.sample_colorscale`` requires numeric (rgb-string) colors,
    not CSS names, so named colors are converted via matplotlib first.
    """
    if n_levels <= 0:
        return []
    # sample_colorscale needs numeric rgb strings, so convert each CSS name to
    # 0-255 rgb via matplotlib first:
    rgb_colors = [
        "rgb({},{},{})".format(*(int(round(c * 255)) for c in mcolors.to_rgb(color)))
        for color in colors
    ]
    scale = pc.make_colorscale(rgb_colors)
    # Single level: just take the scale's start color (avoids a divide-by-zero):
    if n_levels == 1:
        return [pc.sample_colorscale(scale, [0.0])[0]]
    # Otherwise sample n_levels evenly spaced points across [0, 1], so more levels
    # give finer rainbow steps:
    fractions = [i / (n_levels - 1) for i in range(n_levels)]
    return pc.sample_colorscale(scale, fractions)


def _format_id_list_with_counts(ids, counts_1, counts_2):
    """
    Formats a list of ids for hover data, each annotated with each sim's average
    bulk count, one per line but capped at MAX_HOVER_IDS.

    An id missing from a sim's count map (unresolved bulk index, or absent
    from that sim's bulk container) is shown as 'n/a' for that sim (rather
    than being silently dropped).
    """
    ids = list(ids)
    if not ids:
        return ""

    def _one(mol_id):
        c1 = counts_1.get(mol_id)
        c2 = counts_2.get(mol_id)
        c1_str = f"{c1:.3g}" if c1 is not None else "n/a"
        c2_str = f"{c2:.3g}" if c2 is not None else "n/a"
        return f"&nbsp;&nbsp;- {mol_id} (sim1: {c1_str}, sim2: {c2_str})"

    shown = "<br>".join(_one(mid) for mid in ids[:MAX_HOVER_IDS])
    if len(ids) > MAX_HOVER_IDS:
        shown += f"<br>&nbsp;&nbsp;- (+{len(ids) - MAX_HOVER_IDS} more)"
    return shown


def bulk_ids_to_indices(mol_ids, bulk_id_to_idx):
    """
    Resolves a list of molecule ids to bulk indices for one sim's own bulk
    ordering.

    Returns (id_to_idx, missing_ids): id_to_idx holds only the ids that were
    mapped. missing_ids are the ones that did not map these are essentially
    only species that are not free bulk metabolites at all (mass-less class/pool
    terms, or carrier-/macromolecule-bound intermediates (like ACP-, tRNA-, and
    protein-conjugated species whose mass rides on the carrier)). Thus, a
    missing ID is not an error (so the hover data count is just omitted).

    Matching an ID to its bulk count here first checks the exact
    compartment-tagged id (since small molecules can exist in several
    compartments), and if there does not exist a bulk ID for that match, a
    fallback search is run on the compartment-stripped base ID. This base-id
    fallback was built because a molecule's compartment is set independently
    in two places, they disagree for a few proteins (around 19 as of 7/01/26).
    The reaction definition (reaction_stoich) tags a participant with the
    compartment the reaction chemistry was defined in, while the bulk state
    tracks the molecule under the compartment it is localized to (see the
    _build_compartments() function in
    reconstruction/ecoli/dataclasses/getter_functions.py).

    For example, RXN0-312 (the ArcB dephosphorylation reaction) automatically
    tags ARCB-CPLX with [c] because the phospho-transfer chemistry is located
    in the cytoplasm, but ArcB physically sits in the inner membrane within
    the sim (i.e. its counts are stored in the bulk state as ARCB-CPLX[i]).
    The reaction definition and the bulk state are both correct, but they
    disagree on the compartment tag. Thus, the base-id fallback allows a
    reaction  participant to be counted even if its compartment tag disagrees
    with the bulk state (NOTE: all proteins and complexes are only tracked in
    one compartment in the model (unlike metabolites), so the fallback is safe
    because there is no ambiguity in which bulk pool to use).

    Note: These fallback cases are usually regulatory/membrane proteins
    invovled in two-component systems reactions, which is actually taken care of in the
    two_component_systems processes (that why they typicaly carry 0 FBA flux).
    The fallback is inherently an edge case for non-metabolic species, never
    for the metabolites that dominate the plot, but I (Mia) still wanted to
    describe it here just in case it ever becomes relevant and causes confusion
    (given complexes and proteins can be involved in metabolic reactions).
    """
    bulk_base_to_idx = {}
    for bid, i in bulk_id_to_idx.items():
        bulk_base_to_idx.setdefault(_strip_compartment(bid), i)

    id_to_idx = {}
    missing = []
    for mol_id in mol_ids:
        idx = bulk_id_to_idx.get(mol_id)
        if idx is None:
            # if the exact (tagged) id absent, retry on the
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
    """
    Mean each generation's mean bulk counts for one experiment's bulk indices.

    Averages the counts for each free molecule generation wise (i.e. averages
    within each cell over generations >= gen_floor, then average across cells)
    so hover counts sit on the same footing as the plotted average flux.

    Returns {bulk_idx: mean}.

    NOTE: bulk_indices must be indices into this experiment's own sim_data bulk
    ordering (integer bulk indices may not be interchangeable between experiments,
    so this is called once per experiment with that sim's own index list).
    """
    if not bulk_indices:
        return {}
    bulk_indices = [int(i) for i in bulk_indices]
    # Give each requested bulk index a stable synthetic column name (bulkrxn_<i>)
    # so it can be referenced by name through the SQL below:
    names = [f"bulkrxn_{i}" for i in bulk_indices]
    bulk_expr = named_idx("bulk", names, [bulk_indices])
    subquery = cast(
        str, read_stacked_columns(history_sql, [bulk_expr], order_results=False)
    )
    avg_exprs = ", ".join(f'avg("{n}") AS "{n}"' for n in names)
    exp_literal = experiment_id.replace("'", "''")
    # First averaging pass: one avg count per cell (each experiment_id + variant +
    # lineage_seed + generation + agent_id group is a single cell), restricted to
    # the current simulation experiment and to specified post-skip generations:
    per_cell = conn.sql(
        f"""
        SELECT experiment_id, {avg_exprs}
        FROM ({subquery})
        WHERE generation >= {gen_floor}
            AND experiment_id = '{exp_literal}'
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).pl()
    # If no rows survived the filter, report 0 for every requested index rather
    # than leaving them unresolved:
    if per_cell.height == 0:
        return {idx: 0.0 for idx in bulk_indices}
    # Second averaging pass: collapse the per-cell rows to one across-cell mean
    # per column (drop experiment_id so only the numeric columns average):
    means = per_cell.drop("experiment_id").mean()
    row = means.row(0, named=True)
    return {idx: row[f"bulkrxn_{idx}"] for idx in bulk_indices}


def direction_partner(rxn_id):
    """
    Returns (forward_id, reverse_id, this_direction) for an FBA reaction id.

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
    """
    Format hover lines reporting both directions of a reaction for both sims.

    The plot plots each direction is its own point on the scatter to avoid
    plotting net fluxes at one point (as a lot would crowd near zero and it
    would then be difficult to tell the magnitude of each direction on its
    own visually). A consequence of this is that a single plotted point cannot
    fully describe whether the opposite direction is also running, so this
    function looks up the other direction's flux and derives:
      - net   = forward - reverse (the mass that actually moved--this is what the
                model's own ``base_reaction_fluxes`` column aggregates)
      - total = |forward| + |reverse| (total turnover regardless of direction)

    avg_1/avg_2 must be maps over all shared reaction ids (not just the plotted
    subset), so the partner is still found when it is inactive or outside the
    lineage. A reaction with no partner in the map is irreversible in this model
    (or reverse-only, as some reverse ids that have no forward entry) and is
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
    """
    Obtains per-reaction kcat value, constraint-type, and constraining-enzyme
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
    flux differences (read_kinetic_targets_for_experiment for the per-sim
    target values).
    # TODO: consider allowing this info to be read per sim so that hover can
    show the per-sim kcat target fluxes if the two sims have different kcat
    values (or one has kcats and another does not).
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
    """
    Reads data from listeners and computes the mean of the per-cell-means of
    kinetic target/actual fluxes for one experiment.

    Averages exactly like the reaction-flux function does
    (read_bulk_means_for_experiment()).

    Returns {rxn_id: {"target", "lower", "upper", "actual"}} in mM/s, or {} if
    this experiment did not emit the enzyme-kinetics listener.
    # TODO: put a docstring note in the top docstring explaining that the
    listener needs to be on to get this data for the hover data.
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
        # reactions separate).Final select: averages those per-cell means across
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
    """
    Generates hover text lines describing a reaction's kinetic constraint,
    or '' if the reaction is not kinetically constrained.

    This obtains and formats two pieces of information:
      1. Per-sim target/actual flux from the enzyme-kinetics listener (mM/s,
         same units as the plotted flux).
      2. Static kcat / constraint type / constraining enzyme from sim_data,
         assumed to be identical for both sims.
    # TODO: consider allowing this info to be read per sim as well
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
    """
    Computes total production and consumption flux of each metabolite across every
    shared reaction in a given simulation.

    For each reaction R (average flux F, always >= 0 since reverse reactions are
    separate ids) and each participant metabolite M with stoichiometric
    coefficient C, the molar rate F*C is production when C > 0 and consumption
    (counted as -F*C) when C < 0. This summed over all reactions gives the
    following for each metabolite:
    {M: {"prod": total_produced, "cons": total_consumed}} (mM/s).

    At steady state prod ~= cons for an internal metabolite (that has no mass
    or bulk count tracked). These are the denominators for the hover data info's
    per-reaction share (e.g. "reaction Y consumes X% of M's total consumption
    flux"). This share is a metabolite-pool fraction, not an attribution to a
    specific upstream reaction because the flux through a shared pool cannot be
    traced back to one source.
    # TODO: consider adding listeners to track the pie slices per reaction here better
    (however I think this is probably overkill, ie. we get enough info already
    from this and the listeners would probably be large and complicated to build)
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
):
    """
    Constructs the reactant/product hover list.
    This is similar to _format_id_list_with_counts (each ID's
    per-sim average bulk count), but every metabolite also carries this reaction's
    share of that metabolite's total network flux (per simulation).

    The "role" refers to "cons" for reactants (this reaction consumes them) or
    "prod" for products (it produces them). The per-sim share is calculated as
    (|coeff| * this reaction's avg flux) / metabolite's total <role> flux, the
    fraction of all flux through that metabolite this reaction accounts for.
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
        c1s = f"{c1:.3g}" if c1 is not None else "n/a"
        c2s = f"{c2:.3g}" if c2 is not None else "n/a"
        coeff = coeffs.get(met)
        p1 = _pct(rxn_flux_1, coeff, met_totals_1, met)
        p2 = _pct(rxn_flux_2, coeff, met_totals_2, met)
        # Each molecule gets its own bullet, and make a sparate line for the
        # stats from each sim:each simulation gets its own li sim
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
    levels_by_id,
    f1_signed,
    f2_signed,
    catalyst_map,
    stoich,
    kinetic_set,
    parent_links,
    direction,
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
    highlight_bases=None,
):
    """
    Builds hover text for each reaction. It includes a lot:
    - which lineage level the reaction is relative to the reaction of interest,
    - average flux for both sims (with sign, even though |flux| is plotted),
    - whether the reaction is kinetically constrained (assuming the same
      reconstruction for both sims) and info related to kinetics if so,
    - and the catalyzing enzyme(s)/reactant(s)/product(s) with both sims'
      average bulk count when resolvable (intermediates are N/A).

    The reaction of interest is marked with a star if it is not kinetically
    constrained and a diamond if it is kinetically constrained.
    Kinetically constrained reactions are marked with squares and contain
    extra information in the hover data from _format_kinetic_block().
    Reactions that are not kinetically constrained are marked with circles.

    When ``highlight_bases`` (compartment-stripped molecule-of-interest ids)
    is not used, the legend contains each level found (up to the max level or
    truncation limit).

    When ``highlight_bases`` (compartment-stripped molecule-of-interest ids) is
    actively used as a config option (i.e. not empty), a trailing block names
    which molecule(s) of interest this reaction contains, in what role
    (reactant/product/catalyst), in addition to how many BFS levels
    upstream/downstream of the reaction of interest it sits.

    # TODO: determine if this is the right place to talk about how the bulk
    count is used to report counts here since the legit total count of
    proteins/complexes in the model might not be in free form (available for metabolism)
    """
    highlight_bases = highlight_bases or set()
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
        level = levels_by_id[rxn_id]
        level_label = "center" if level == 0 else f"level {level}"
        catalysts = catalyst_map.get(rxn_id, [])
        reactants = reaction_reactants(rxn_id, stoich)
        products = reaction_products(rxn_id, stoich)
        constrained = "yes" if rxn_id in kinetic_set else "no"
        link_line = format_lineage_link_lines(rxn_id, parent_links, direction)
        kinetic_block = _format_kinetic_block(
            rxn_id, kinetic_info, kinetic_targets_1, kinetic_targets_2
        )
        lines = [
            f"<b>{rxn_id}</b><br>",
            f"Lineage: {level_label}<br>",
            f"{link_line}<br>" if link_line else "",
            f"Sim 1 avg flux: {a:.3e}<br>",
            f"Sim 2 avg flux: {b:.3e}<br>",
            _format_direction_block(rxn_id, shared_avg_1, shared_avg_2, flux_unit),
            f"Kinetically constrained: {constrained}<br>",
            kinetic_block,
            f"Catalyst avg count(s):<br>"
            f"{_format_id_list_with_counts(catalysts, catalyst_counts_1, catalyst_counts_2) or 'none'}<br>",
            f"Reactant avg count(s) & share of each metabolite's total flux:<br>"
            f"{_format_participants_with_share(reactants, reactant_counts_1, reactant_counts_2, shared_avg_1.get(rxn_id), shared_avg_2.get(rxn_id), stoich.get(rxn_id, {}), met_totals_1, met_totals_2, 'cons') or 'none'}<br>",
            f"Product avg count(s) & share of each metabolite's total flux:<br>"
            f"{_format_participants_with_share(products, product_counts_1, product_counts_2, shared_avg_1.get(rxn_id), shared_avg_2.get(rxn_id), stoich.get(rxn_id, {}), met_totals_1, met_totals_2, 'prod') or 'none'}",
        ]
        # Molecule-highlight mode: append which molecule(s) of interest this
        # reaction contains, their role(s), and its position relative to the
        # center (the level color/legend now carries the molecule, not the level,
        # so the level moves into the hover here):
        if highlight_bases:
            matches = reaction_highlight_matches(
                rxn_id, stoich, catalyst_map, highlight_bases
            )
            if matches:
                # Phrase the center as "at the center reaction"; every other level
                # as "<k> level(s) upstream/downstream of the center":
                where = (
                    "involved in the reaction of interest"
                    if level == 0
                    else f"{level} level(s) {direction} of the center reaction"
                )
                match_lines = "<br>".join(
                    f"&nbsp;&nbsp;- {b} (as {', '.join(sorted(roles))})"
                    for b, roles in matches.items()
                )
                lines.append(
                    f"<br>Molecule(s) of interest -- {where}:<br>{match_lines}"
                )
        texts.append("".join(lines))
    return texts


def _flux_log_transform(sim1_avg, sim2_avg, flux_unit, tag=""):
    """
    Computes the log10(|avg flux| + ε) for both sims (using the same
    epsilon value for each).

    Epsilon (eps) is the offset added to |flux| before the log so zero-flux
    reactions are still plottable (as log10(0) is undefined). It is the
    smallest nonzero |avg flux| across both sims, so zeros sit just below the
    smallest real point. eps is floored in the rare case where the smallest
    nonzero flux is more than 10**MAX_LOG_ORDERS times smaller than the largest.
    If this happens, eps is raised to 10**(-MAX_LOG_ORDERS) x the largest
    |flux| so one near-zero outlier can't stretch the axis too much.

    Returns (sim1_log, sim2_log, eps, eps_desc). Note: eps_desc is a short
    label describing how eps was chosen.
    """
    # Pool both sims' magnitudes so one shared eps/floor applies to both axes:
    both = np.abs(np.concatenate([sim1_avg, sim2_avg]))
    # Only strictly-positive magnitudes are eps candidates (since zeros are
    # what eps lifts for fluxes to be visible on the plot):
    nz = both[both > 0]

    # If there are no reactions with zero flux, eps=0, as no flux should cause
    # an undefined log10() value (note: having no reactions with zero flux
    # would be very rare, typically only ~1900 of ~9500 have nonzero flux in a given sim).
    if not nz.size:
        eps, eps_desc = (
            1.0,
            (
                "a fallback value due to no nonzero flux. "
                "NOTE: it is rare to have no reactions with zero flux, double "
                "check flux values manually. "
            ),
        )
    else:
        # Default eps = smallest real magnitude, so zeros sit just below it:
        eps = float(nz.min())
        # Floor = a millionth (10**-MAX_LOG_ORDERS) of the largest magnitude. eps is
        # never allowed below this so one tiny outlier can't stretch the axis:
        floor = float(both.max()) * 10.0 ** (-MAX_LOG_ORDERS)
        if eps < floor:
            eps = floor
            eps_desc = (
                f"flooring to ε={10.0 ** (-MAX_LOG_ORDERS):.0e} x the largest |flux| "
                "(smallest nonzero flux was significantly smaller than the "
                "largest flux, and could mess up the plot if used as ε value "
                "added to the log10())."
            )
        else:
            eps_desc = "the smallest nonzero |avg flux|."
    print(
        f"Flux log-transform is computed as: log10(|avg flux| + ε), where "
        f"the shared ε = {eps:.3e} ({flux_unit}). Epsilon value was assigned "
        f"via: {eps_desc}"
    )
    # abs() before adding eps since only magnitude is plotted (sign is in the hover),
    # and eps guarantees a positive argument so log10 is always defined:
    sim1_log = np.log10(np.abs(sim1_avg) + eps)
    sim2_log = np.log10(np.abs(sim2_avg) + eps)
    return sim1_log, sim2_log, eps, eps_desc


def _add_parity_line(fig: go.Figure, sim1_log: np.ndarray, sim2_log: np.ndarray):
    """
    Adds a dashed y=x reference line spanning the data range.
    """
    # Span the true data range:
    min_val = min(float(sim1_log.min()), float(sim2_log.min()))
    max_val = max(float(sim1_log.max()), float(sim2_log.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=2),
            name="y = x",
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _add_active_count_legend(fig: go.Figure, legend_text: str):
    """
    Adds a marker-less legend-only entry (a no-data x/y = None trace) carrying
    legend_text with the  "N reactions not shown (not part of the lineage)"
    note. Skipped when legend_text is empty.
    # TODO: decide if this should be moved to the subtitle lines intstead.
    """
    if not legend_text:
        return
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)"),
            name=legend_text,
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _apply_square_layout(fig: go.Figure, title: str, xlabel: str, ylabel: str):
    """
    Applies a square log-axis layout with the legend placed outside/right of
    the plot area.

    The plot is widened and given a right margin so the legend isn't clipped.
    # TODO: mess with the sizing more as things still are overlapping.
    """
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


def build_lineage_scatter_figure(
    rxn_ids,
    levels_by_id,
    max_level,
    direction,
    center_reaction_id,
    sim1_log,
    sim2_log,
    hover_texts,
    title,
    xlabel,
    ylabel,
    not_shown_legend,
    kinetic_set=frozenset(),
):
    """
    Builds the log-log avg-flux scatter restricted to one direction's lineage.
    The center reaction as a large red star plus one trace per level
    1 - max_level colored along the direction's gradient.

    The reaction of interest is represented by a red star (red diamond if it
    is kinetically constrained).

    NOTE: Marker shape encodes the kinetic constraint independently of level
    color so that reactions in ``kinetic_set`` are squares, and the rest are
    circles, applied via a per-point marker.symbol array so each level keeps
    one legend entry.
    """
    # Both directions use the same full ROYGBIV rainbow, sampled across the BFS
    # levels, so adjacent levels are easy to tell apart:
    level_colors = level_colorscale(
        ["red", "orange", "gold", "green", "blue", "indigo", "violet"], max_level
    )

    id_to_pos = {r: i for i, r in enumerate(rxn_ids)}
    fig = go.Figure()

    # Center reaction (level 0) is drawn first as a single large red star:
    if center_reaction_id in id_to_pos:
        i = id_to_pos[center_reaction_id]
        center_constrained = center_reaction_id in kinetic_set
        fig.add_trace(
            go.Scatter(
                x=[sim1_log[i]],
                y=[sim2_log[i]],
                mode="markers",
                marker=dict(
                    symbol="star-square" if center_constrained else "star",
                    size=22,
                    color="red",
                    line=dict(width=1, color="black"),
                ),
                name=(
                    "reaction of interest"
                    + (" [kinetically constrained]" if center_constrained else "")
                ),
                text=[hover_texts[i]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    # Create traces per BFS level, all sharing that level's rainbow color:
    for level in range(1, max_level + 1):
        # Point indices whose reaction was assigned to this level:
        idxs = [i for i, r in enumerate(rxn_ids) if levels_by_id[r] == level]
        if not idxs:
            continue
        # Split each level into up to two same-colored legend entries by kinetic
        # status: non-kinetic circles ("Nth level (n)") and kinetic squares
        # ("Nth level kinetic (n)"):
        non_idxs = [i for i in idxs if rxn_ids[i] not in kinetic_set]
        kin_idxs = [i for i in idxs if rxn_ids[i] in kinetic_set]
        # circle = not kinetically constrained, square = constrained:
        for group_idxs, symbol, name in (
            (
                non_idxs,
                "circle",
                f"{_ordinal(level)} level ({len(non_idxs)})",
            ),
            (
                kin_idxs,
                "square",
                f"{_ordinal(level)} level kinetic ({len(kin_idxs)})",
            ),
        ):
            if not group_idxs:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sim1_log[group_idxs],
                    y=sim2_log[group_idxs],
                    mode="markers",
                    marker=dict(
                        symbol=symbol,
                        size=8,
                        color=level_colors[level - 1],
                        opacity=0.85,
                        line=dict(width=0),
                    ),
                    name=name,
                    text=[hover_texts[i] for i in group_idxs],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                )
            )

    _add_parity_line(fig, sim1_log, sim2_log)
    _add_active_count_legend(fig, not_shown_legend)
    _apply_square_layout(fig, title, xlabel, ylabel)
    return fig


def _kinetic_symbols_for(idxs, rxn_ids, kinetic_set):
    """
    Generates per-flux point marker symbol array for a list of point indices,
    so one trace keeps a single color/legend entry while each point still
    shows its constraint status.
    """
    return ["square" if rxn_ids[i] in kinetic_set else "circle" for i in idxs]


def build_molecule_highlight_figure(
    rxn_ids,
    levels_by_id,
    direction,
    center_reaction_id,
    sim1_log,
    sim2_log,
    hover_texts,
    stoich,
    catalyst_map,
    highlight_order,
    title,
    xlabel,
    ylabel,
    not_shown_legend,
    kinetic_set=frozenset(),
):
    """
    Generates a log-log plot of the average flux scatter for one direction's
    lineage, colored by molecules of interest, rather than by BFS level.

    Each molecule of interest in ``highlight_order`` gets one trace that covers
    every plotted lineage reaction that contains it as a reactant, product, or
    catalyst; a reaction containing two molecules is drawn once per molecule
    (overlapping dots at the same coordinates, one per color).

    NOTE: passing through less than 15 distinct molecules is best here to
    avoid color palette cycling.


    # TODO: add the color cycling comment to the main docstring as a warning
    """
    id_to_pos = {r: i for i, r in enumerate(rxn_ids)}
    base_set = set(highlight_order)
    fig = go.Figure()

    # Center reaction (level 0) first and exclude it from the molecule/other
    # groups below so it is never double-drawn:
    center_pos = id_to_pos.get(center_reaction_id)
    if center_pos is not None:
        center_constrained = center_reaction_id in kinetic_set
        fig.add_trace(
            go.Scatter(
                x=[sim1_log[center_pos]],
                y=[sim2_log[center_pos]],
                mode="markers",
                marker=dict(
                    symbol="star-square" if center_constrained else "star",
                    size=22,
                    color="red",
                    line=dict(width=1, color="black"),
                ),
                name=(
                    "reaction of interest"
                    + (" [kinetically constrained]" if center_constrained else "")
                ),
                text=[hover_texts[center_pos]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    # Partition the (non-center) plotted reactions: each molecule's point list is
    # every reaction that contains it (so multi-molecule reactions appear in more
    # than one list) and reactions matching nothing fall to the gray catch-all:
    mol_point_indices: dict[str, list[int]] = {b: [] for b in highlight_order}
    other_indices: list[int] = []
    for i, r in enumerate(rxn_ids):
        if i == center_pos:
            continue
        matches = reaction_highlight_matches(r, stoich, catalyst_map, base_set)
        if matches:
            for b in matches:
                mol_point_indices[b].append(i)
        else:
            other_indices.append(i)

    # Gray background first so the colored molecule points draw on top of it:
    if other_indices:
        fig.add_trace(
            go.Scatter(
                x=sim1_log[other_indices],
                y=sim2_log[other_indices],
                mode="markers",
                marker=dict(
                    symbol=_kinetic_symbols_for(other_indices, rxn_ids, kinetic_set),
                    size=8,
                    color="lightgray",
                    opacity=0.4,
                    line=dict(width=0),
                ),
                name=f"Other lineage reactions ({len(other_indices)})",
                text=[hover_texts[i] for i in other_indices],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    # Distinct high-contrast qualitative palette giving each highlighted molecule
    # its own legend color from https://sashamaps.net/docs/resources/20-colors/
    # (cycles if there are more molecules than colors):
    palette = [
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
    # Make one trace per molecule of interest:
    for k, b in enumerate(highlight_order):
        idxs = mol_point_indices[b]
        color = palette[k % len(palette)]
        if idxs:
            fig.add_trace(
                go.Scatter(
                    x=sim1_log[idxs],
                    y=sim2_log[idxs],
                    mode="markers",
                    marker=dict(
                        symbol=_kinetic_symbols_for(idxs, rxn_ids, kinetic_set),
                        size=10,
                        color=color,
                        opacity=0.9,
                        line=dict(width=0.5, color="black"),
                    ),
                    name=f"{b} (n={len(idxs)})",
                    text=[hover_texts[i] for i in idxs],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                )
            )
        else:
            # Molecule matched nothing in this lineage: still emit an empty legend
            # entry so the requested molecule and its color remain visible:
            # TODO: need to make sure this empty is explained if there are
            #  reactions it participates in that are not plotted
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(
                        size=9, color=color, line=dict(width=0.5, color="black")
                    ),
                    name=f"{b} (n=0)",
                    showlegend=True,
                    hoverinfo="skip",
                )
            )

    _add_parity_line(fig, sim1_log, sim2_log)
    _add_active_count_legend(fig, not_shown_legend)
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
    Plots the average FBA reaction fluxes from two simulations against each other,
    restricted to the upstream and downstream BFS "lineage"
    levels of one center reaction of interest.

    Reactions outside the lineage are not plotted at all, though note that
    without some filtering (via the config options), almost everything connects! :)

    Args:
        params: Dictionary containing parameters of the format::

            {
                # REQUIRED CONFIG OPTION:

                "reaction_id"  (str OR list[str], no default):
                #   The FBA reaction ID(s) to trace the lineage from. Each ID
                #   produces its own pair of plots: one upstream, one downstream.
                #   Pass a single id string to trace one reaction, or a list to
                #   trace several in one run. "reaction_ids" is accepted as an
                #   exact alias. Omitting it (or passing an empty value) raises
                #   ValueError. IDs not found in the model are skipped.
                examples:
                "reaction_id": "RXN0-1234",
                or: "reaction_id": ["RXN0-1234", "RXN0-2345"],

                # OPTIONAL CONFIG OPTIONS:

                "skip_n_gens"  (int, default 2):
                #   Number of initial generations (per seed) to drop before the
                #   fluxes are averaged, so early (and possibly
                #   non-equilibrated/steady state) generations do not skew the
                #   plotted averages.
                example:
                "skip_n_gens": 2

                "excluded_connector_metabolites"  (
                list[str], default = DEFAULT_EXCLUDED_CONNECTOR_METABOLITES):
                #   Replaces the curated 14-ID currency set with this list of
                #   base IDs. Pass [] to exclude nothing, but note that the
                #   the lineage will likely blow up.
                example:
                "excluded_connector_metabolites": ["WATER", "ATP", "PROTON"]

                "exclude_metabolite_degree_over"  (int, default None = off):
                #   Also auto-exclude any metabolite that participates in more
                #   than this many reactions. A smaller number results in
                #   more metabolites pruned (and likely a smaller lineage), but
                #   be cautious using this as one might end up excluding
                #   molecules that might be important for a lineage, even if common.
                example:
                "exclude_metabolite_degree_over": 100

                "always_include_metabolites"  (list[str], default []):
                #   This is a way to override the other filters above to list
                #   base ids to always keep in the walk even if the curated
                #   list or the degree threshold above would have excluded
                #   them. Applied last, so it overrides both. Use it to re-add a
                #   metabolite that was dropped for being too common but whose
                #   edges you still want followed in this particular lineage.
                example:
                "always_include_metabolites": ["PYRUVATE"]

                "max_levels"  (int, default DEFAULT_MAX_LEVELS = 20):
                #   Safety cap on BFS depth so the legend does not explode with
                #   too many layers. The cap only guards against a nearly
                #   fully-connected exclusion set. Reaching it is
                #   flagged as "TRUNCATED" in the plot title/subtitle.
                example:
                "max_levels": 20

                "only_show_reactions_containing_molecules"  (
                list[str], default [] = off):
                #   Switches every figure from BFS-level coloring to molecule-
                #   highlight coloring. Each listed molecule (matched as a
                #   reactant, product, or catalyst) gets its own color and
                #   legend entry. If empty, the fluxes are colored by level.
                example:
                "only_show_reactions_containing_molecules": ["TMP", "MET"]
            }
    """
    # Ensure reactions were passed through:
    raw_rxn = params.get("reaction_id", params.get("reaction_ids"))
    if not raw_rxn:
        raise ValueError(
            "fba_flux_lineage_comparisons_plotly requires params['reaction_id'] "
            "(a single FBA reaction id, or a list of them, to trace "
            "ancestry/descendants from); none was provided."
        )
    # Ensure the reactions passed through are valid:
    reaction_ids = [raw_rxn] if isinstance(raw_rxn, str) else list(raw_rxn)
    max_levels = params.get("max_levels", DEFAULT_MAX_LEVELS)
    skip_gens = params.get("skip_n_gens", 2)

    # Extract the FBA solver's per-reaction velocities:
    flux_column = "listeners__fba_results__reaction_fluxes"
    flux_unit = flux_units(flux_column)

    # Check if optional molecule-highlight mode is active:
    highlight_order = highlight_base_order(
        params.get("only_show_reactions_containing_molecules", [])
    )
    highlight_bases = set(highlight_order)
    if highlight_order:
        print(
            "Molecule-highlight mode is on (only_show_reactions_containing_molecules): "
            f"coloring lineage reactions by {highlight_order} (level detail is "
            f"still included in the hover data)."
        )

    min_gen = int(
        conn.sql(f"SELECT min(generation) AS g FROM ({history_sql})").pl()["g"][0]
    )
    gen_floor = min_gen + skip_gens

    subquery = cast(
        str, read_stacked_columns(history_sql, [flux_column], order_results=False)
    )

    # Extract fluxes:
    # unnested_fluxes: explode the per-timestep flux array to one row per
    # (reaction position idx, timestep). avg_fluxes: average over timesteps
    # within each cell for each reaction idx. final select: re-collect each
    # cell's per-reaction means into one list ordered by idx.
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

    # x-axis = the first experiment_id listed in the config, y-axis = second:
    present = set(unique_exp_ids)
    ordered_exp_ids = [e for e in sim_data_paths if e in present] or unique_exp_ids
    exp_id_1, exp_id_2 = ordered_exp_ids[0], ordered_exp_ids[1]
    print(f"Comparing {exp_id_1} (Sim 1; x-axis) vs {exp_id_2} (Sim 2; y-axis)")

    fluxes_exp1 = all_fluxes.filter(pl.col("experiment_id") == exp_id_1)
    fluxes_exp2 = all_fluxes.filter(pl.col("experiment_id") == exp_id_2)

    # Stack each sim's per-cell flux vectors into a (cells x reactions) array, then
    # average down the cells axis to one mean flux per reaction position:
    sim1_fluxes = ndlist_to_ndarray(fluxes_exp1["avgFlux"])
    sim2_fluxes = ndlist_to_ndarray(fluxes_exp2["avgFlux"])
    sim1_avg_raw = sim1_fluxes.mean(axis=0)
    sim2_avg_raw = sim2_fluxes.mean(axis=0)
    print(f"Exp1 has {len(sim1_fluxes)} cells")
    print(f"Exp2 has {len(sim2_fluxes)} cells")

    # Align reactions by ID:
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

    # Build ID to position map in each sim's own listener ordering (they need
    # not agree, so alignment is by id). The sorted intersection is the shared
    # reaction set:
    pos1 = {r: i for i, r in enumerate(rxn_ids_1)}
    pos2 = {r: i for i, r in enumerate(rxn_ids_2)}
    shared_rxn_ids = sorted(pos1.keys() & pos2.keys())
    if not shared_rxn_ids:
        raise ValueError(
            f"No reaction IDs shared between {exp_id_1} ({len(rxn_ids_1)} rxns) "
            f"and {exp_id_2} ({len(rxn_ids_2)} rxns); cannot compare. Manually "
            f"check for errors in the plot code because this should be rare."
        )
    # Map ID to avg flux over every shared reaction, and each read at that
    # sim's own position. The whole lineage plotting downstream looks fluxes
    # up by id here, so the hover can still report a reaction that falls
    # outside a given lineage:
    shared_avg_1 = {r: sim1_avg_raw[pos1[r]] for r in shared_rxn_ids}
    shared_avg_2 = {r: sim2_avg_raw[pos2[r]] for r in shared_rxn_ids}
    print(
        f"Reactions: {len(rxn_ids_1)} in {exp_id_1}, {len(rxn_ids_2)} in "
        f"{exp_id_2}; {len(shared_rxn_ids)} shared."
    )

    # Obtain other simulation data:
    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)
    stoich = sim_data.process.metabolism.reaction_stoich
    # Per metabolite total production/consumption flux across all shared
    # reactions (the denominators for the hover's per-reactant/product
    # "consumes/produces X% of this metabolite's total flux" share):
    met_totals_1 = compute_metabolite_flux_totals(shared_avg_1, stoich)
    met_totals_2 = compute_metabolite_flux_totals(shared_avg_2, stoich)
    missing_rxns = [r for r in reaction_ids if r not in stoich]
    if missing_rxns:
        preview = ", ".join(missing_rxns[:10]) + (
            " ..." if len(missing_rxns) > 10 else ""
        )
        print(
            f"WARNING: {len(missing_rxns)} requested reaction id(s) not in "
            f"sim_data.process.metabolism.reaction_stoich (skipped): {preview}"
        )
    reaction_ids = [r for r in reaction_ids if r in stoich]
    if not reaction_ids:
        raise ValueError(
            "None of the requested params['reaction_id'] value(s) were found in "
            "sim_data.process.metabolism.reaction_stoich; nothing to plot. "
            "Check plotting code or simulation reconstruction for errors."
        )
    kinetic_set = set(sim_data.process.metabolism.kinetic_constraint_reactions)
    kinetic_info = get_kinetic_constraint_info(sim_data)
    catalyst_map = {
        rxn: list(cats)
        for rxn, cats in sim_data.process.metabolism.reaction_catalysts.items()
        if cats
    }

    # Determine every reaction in the model to be used as the denominator
    # for the "N reactions not shown (outside the lineage)" legend/print below:
    total_reactions = len(stoich)

    excluded_metabolites, auto_excluded, readded = resolve_excluded_metabolites(
        params, stoich
    )
    if auto_excluded:
        print(
            f"Auto-excluded {len(auto_excluded)} metabolite(s) appearing in > "
            f"{params['exclude_metabolite_degree_over']} reactions: "
            f"{', '.join(sorted(auto_excluded))}"
        )
    if readded:
        print(
            f"Re-included {len(readded)} metabolite(s) via "
            f"always_include_metabolites (kept despite exclusion rules): "
            f"{', '.join(sorted(readded))}"
        )

    # Per-sim bulk indices for hover counts (not interchangeable between sims):
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

    # Determine per-sim kinetic target/actual fluxes for the hover data:
    print("REACTIONS WITH ENZYME KINTEICS TARGETS:")
    kinetic_targets_1 = read_kinetic_targets_for_experiment(
        conn, history_sql, config_sql, exp_id_1, gen_floor
    )
    kinetic_targets_2 = read_kinetic_targets_for_experiment(
        conn, history_sql, config_sql, exp_id_2, gen_floor
    )
    if not kinetic_targets_1 and not kinetic_targets_2:
        print(
            "  Neither sim emitted enzyme-kinetics target fluxes; hover will show "
            "only the static kcat block for kinetically constrained reactions."
        )

    def extract_short_id(exp_id):
        """
        Extracts a short identifier from the full experiment ID (since not all
        reactions contain characters that can be safely made into file names).
        """
        parts = exp_id.split("_")
        for i, part in enumerate(parts):
            if "-" in part and len(part) == 15:
                return "_".join(parts[:i])
        return exp_id

    sim1_short = extract_short_id(exp_id_1)
    sim2_short = extract_short_id(exp_id_2)

    comparison_outdir = outdir + f"_{exp_id_1}_vs_{exp_id_2}"
    os.makedirs(comparison_outdir, exist_ok=True)

    # Create one upstream + one downstream plot per reaction. product() groups
    # the output by reaction (r1 up, r1 down, r2 up, etc.):
    for reaction_id, direction in product(reaction_ids, ("upstream", "downstream")):
        levels, edges, hit_cap = compute_lineage_levels(
            stoich,
            reaction_id,
            direction,
            excluded_metabolites=excluded_metabolites,
            max_levels=max_levels,
        )
        # Invert {level: [rxns]} into {rxn: level} for O(1) per reaction lookups:
        levels_by_id = {r: lvl for lvl, rs in levels.items() for r in rs}
        parent_links = build_parent_links(edges)
        lineage_ids_all = list(levels_by_id.keys())

        # "found" counts discovered reactions excluding the level-0 center,
        # "not_shown" is everything in the model outside this lineage:
        found = sum(len(rs) for lvl, rs in levels.items() if lvl > 0)
        not_shown = total_reactions - (1 + found)  # + 1 for the ROI
        print(
            f"REACTION SEARCH RESULTS ({direction}): "
            f"\n    {len(levels) - 1} level(s) found in lineage "
            f"({'hit max_levels cap' if hit_cap else 'terminated naturally'})."
            f"\n     {found} reactions found in lineage, {not_shown} not shown "
            f"\n    due to no lineage relationship or max lineage level search cap hit."
        )

        # Only reactions that are both part of the lineage & have flux data
        # shared between the two sims are plottable. Reactions outside the
        # lineage are dropped entirely, and lineage reactions missing shared
        # flux data are dropped with a warning (rather than plotting the origin
        # so that if a reaction is missing from a simulation it is noted):
        rxn_ids = [r for r in lineage_ids_all if r in shared_avg_1]
        missing_from_flux = sorted(set(lineage_ids_all) - shared_avg_1.keys())
        if missing_from_flux:
            preview = ", ".join(missing_from_flux[:10]) + (
                " ..." if len(missing_from_flux) > 10 else ""
            )
            print(
                f"  {direction}: {len(missing_from_flux)} lineage reaction(s) not "
                f"found in both sims' flux data (dropped): {preview}"
            )

        if not rxn_ids:
            print(f"  {direction}: no plottable lineage reactions; skipping plot.")
            continue

        # Build this direction's flux vectors in rxn_ids order:
        sim1_avg_dir = np.array([shared_avg_1[r] for r in rxn_ids])
        sim2_avg_dir = np.array([shared_avg_2[r] for r in rxn_ids])
        sim1_log, sim2_log, flux_eps, flux_eps_desc = _flux_log_transform(
            sim1_avg_dir, sim2_avg_dir, flux_unit, tag=direction
        )

        # Union of every catalyst/reactant/product id across the plotted lineage
        # reactions, so each molecule's bulk count is read at most once per sim:
        hover_molecule_ids: set[str] = set()
        for r in rxn_ids:
            hover_molecule_ids.update(catalyst_map.get(r, []))
            hover_molecule_ids.update(reaction_reactants(r, stoich))
            hover_molecule_ids.update(reaction_products(r, stoich))

        idx_map_1, missing_1 = bulk_ids_to_indices(hover_molecule_ids, bulk_id_to_idx_1)
        idx_map_2, missing_2 = bulk_ids_to_indices(hover_molecule_ids, bulk_id_to_idx_2)

        # These "missing" ids are reaction participants that are not free bulk
        # metabolites at all:
        # TODO: see if I can remove this now (not really needed)
        for exp, missing in ((exp_id_1, missing_1), (exp_id_2, missing_2)):
            if missing:
                print(
                    f"NON-BULK SPECIES ({direction}): "
                    f"\n    {len(missing)} of {len(hover_molecule_ids)} hover "
                    f"ID(s) are non-bulk class/carrier-bound species "
                    f"\n    with no bulk count in {exp}."
                )

        means_1 = read_bulk_means_for_experiment(
            conn, history_sql, exp_id_1, list(idx_map_1.values()), gen_floor
        )
        means_2 = read_bulk_means_for_experiment(
            conn, history_sql, exp_id_2, list(idx_map_2.values()), gen_floor
        )
        counts_1 = {mol_id: means_1[idx] for mol_id, idx in idx_map_1.items()}
        counts_2 = {mol_id: means_2[idx] for mol_id, idx in idx_map_2.items()}

        hover_texts = make_hover_texts(
            rxn_ids,
            levels_by_id,
            sim1_avg_dir,
            sim2_avg_dir,
            catalyst_map,
            stoich,
            kinetic_set,
            parent_links,
            direction,
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
            highlight_bases=highlight_bases,
        )

        plotted_levels = [levels_by_id[r] for r in rxn_ids if levels_by_id[r] > 0]
        max_level = max(plotted_levels) if plotted_levels else 0
        # TODO: figure out a better way to word this so it isnt so awkward in the subtitle
        truncation_note = (
            " [max_levels lineage cap reached, plotted ancestry may be incomplete]"
            if hit_cap
            else ""
        )
        not_shown_legend = f"{not_shown} reactions not shown (not part\nof the {direction} lineage levels plotted)"

        # MAIN TITLE FORMATTING
        # Note: the "n of m" count uses the full lineage found within the max
        # levels (including the ROI center) as m (so if the max levels are hit,
        # technically, m can be larger than what the plot reports).
        # NOTE: n doesnt equal m when one simulation contained a reaction ID
        # that was found to be in the lineage that the other simulation does
        # not have data for:
        main_title = f"FBA Flux Comparison - {direction.capitalize()} Lineage"
        subtitle = (
            f"<sub>{reaction_id}"
            f"<br>{len(rxn_ids)} of total {len(lineage_ids_all)} {direction} lineage "
            f"reactions plotted {truncation_note}. "
            f"<br>Sim 1 (x): {exp_id_1} (averaged over {len(sim1_fluxes)} cells) vs. "
            f"<br>Sim 2 (y): {exp_id_2} (averaged over {len(sim2_fluxes)} cells)"
            f"<br>The first {skip_gens} generations in each seed were excluded before averaging."
            f"<br>ε={flux_eps:.1e} {flux_unit} added to |avg. flux| to avoid log10(0)."
        )
        xlabel = f"log10(|Sim 1 average flux| + {flux_eps:.1e}) ({flux_unit})"
        ylabel = f"log10(|Sim 2 average flux| + {flux_eps:.1e}) ({flux_unit})"

        # Molecule-highlight mode swaps the level-colored builder for the
        # molecule-colored one (same coordinates, hover, center star, axes):
        if highlight_order:
            fig = build_molecule_highlight_figure(
                rxn_ids,
                levels_by_id,
                direction,
                reaction_id,
                sim1_log,
                sim2_log,
                hover_texts,
                stoich,
                catalyst_map,
                highlight_order,
                f"{main_title}<br>{subtitle}",
                xlabel,
                ylabel,
                not_shown_legend,
                kinetic_set=kinetic_set,
            )
            # Create the output filename:
            filename = os.path.join(
                comparison_outdir,
                _safe_filename_part(
                    f"fba_flux_lineage_comparison_with_highlights"
                    f"_{direction}_{reaction_id}_"
                    f"{sim1_short}_vs_{sim2_short}"
                )
                + ".html",
            )
        else:
            # build the rainbow graph otherwise:
            fig = build_lineage_scatter_figure(
                rxn_ids,
                levels_by_id,
                max_level,
                direction,
                reaction_id,
                sim1_log,
                sim2_log,
                hover_texts,
                f"{main_title}<br>{subtitle}",
                xlabel,
                ylabel,
                not_shown_legend,
                kinetic_set=kinetic_set,
            )

            # Create the output filename:
            filename = os.path.join(
                comparison_outdir,
                _safe_filename_part(
                    f"fba_flux_lineage_comparison_{direction}_{reaction_id}_"
                    f"{sim1_short}_vs_{sim2_short}"
                )
                + ".html",
            )

        # Write out the figure:
        fig.write_html(filename)

        # UNIQUE PLOT:
        # Create a second figure set filtered to reactions that are classified
        # as unique if their base reaction occupies a single level in the
        # full discovered lineage. Note that several FBA reactions (forward,
        # reverse, and enzyme-split copies) can share one base, and because
        # those copies are separate reactions, the BFS can reach them at
        # different depths/levels. Base reactions spanning multiple levels
        # (forward/reverse/enzyme-split copies discovered at different BFS
        # depths) are dropped. If even one copy (say a flux-less reverse)
        # was discovered at a different level, the base is considered non-unique
        # and all its plotted copies are dropped. Everything else (level
        # coloring, per-level kinetic split, hover text, center marker, axes,
        # eps) is identical to the non-unique plots.

        # Obtain the base mapping:
        base_map = sim_data.process.metabolism.reaction_id_to_base_reaction_id

        # Resolve an FBA id to its base reaction using base_map (returns base
        # ID if it already is the base ID):
        def _base_of(fba_id):
            if fba_id in base_map:
                # Not the base reaction:
                return base_map[fba_id]
            if fba_id.endswith(REVERSE_TAG):
                # Strip reverse tag if needed:
                return fba_id[: -len(REVERSE_TAG)]
            return fba_id

        # Determine which BFS levels each base reaction's copies landed
        # on across the full lineage (a base spanning >1 levels is "non-unique"):
        base_to_levels: dict[str, set[int]] = {}
        for fba_id, lvl in levels_by_id.items():
            base_to_levels.setdefault(_base_of(fba_id), set()).add(lvl)
        nonunique_bases = {b for b, lvls in base_to_levels.items() if len(lvls) > 1}
        n_nonunique = len(nonunique_bases)

        # Keep only plotted points whose base reaction sits on a single level
        # (note: this marks all copies of a multi-level base for removal at once):
        keep_positions = [
            i for i, r in enumerate(rxn_ids) if _base_of(r) not in nonunique_bases
        ]
        # Force the ROI back in if it got removed due to non-uniqueness
        # and flag it in the subtitle rather than dropping it:
        center_nonunique = (
            reaction_id in rxn_ids and _base_of(reaction_id) in nonunique_bases
        )
        if center_nonunique:
            # Force the center's position back into the kept set (and re-sort) so
            # its hover survives even though its base spans multiple levels:
            center_pos = rxn_ids.index(reaction_id)
            keep_positions = sorted(set(keep_positions) | {center_pos})

        if not keep_positions:
            print(
                f"  {direction}: no base-unique lineage reactions; "
                f"skipping unique plot."
            )
            continue

        # Map every parallel array by the kept positions so ids, coords, hover
        # text and level map all stay aligned for the unique base figure:
        unique_rxn_ids = [rxn_ids[i] for i in keep_positions]
        unique_sim1_log = sim1_log[keep_positions]
        unique_sim2_log = sim2_log[keep_positions]
        unique_hover = [hover_texts[i] for i in keep_positions]
        unique_levels_by_id = {r: levels_by_id[r] for r in unique_rxn_ids}
        unique_plotted_levels = [
            levels_by_id[r] for r in unique_rxn_ids if levels_by_id[r] > 0
        ]
        unique_max_level = max(unique_plotted_levels) if unique_plotted_levels else 0
        n_removed_points = len(rxn_ids) - len(unique_rxn_ids)

        # Generate the subtitle for the unique graphs:
        unique_subtitle = (
            f"<sub>{reaction_id}"
            f"<br>Only plotting reactions that correspond to a base reaction ID found on only one lineage level."
            f"<br>{n_nonunique} non-unique base reactions filtered out "
            f"({n_removed_points} total reactions removed)."
            f"<br>{len(unique_rxn_ids)} of {len(rxn_ids)} plotted {direction} "
            f"lineage reaction(s) kept {truncation_note}. "  # TODO: double check logic here
            f"<br>Sim 1 (x): {exp_id_1} (averaged over {len(sim1_fluxes)} cells) vs. "
            f"<br>Sim 2 (y): {exp_id_2} (averaged over {len(sim2_fluxes)} cells)"
            f"<br>The first {skip_gens} generations in each seed were excluded before averaging."
            f"<br>ε={flux_eps:.1e} {flux_unit} added to |avg. flux| to avoid log10(0)."
        )
        # Plot the figure highlighting the specified highlighted molecules:
        if highlight_order:
            unique_fig = build_molecule_highlight_figure(
                unique_rxn_ids,
                unique_levels_by_id,
                direction,
                reaction_id,
                unique_sim1_log,
                unique_sim2_log,
                unique_hover,
                stoich,
                catalyst_map,
                highlight_order,
                f"{main_title} (filtered to include unique base reactions only)<br>{unique_subtitle}",
                xlabel,
                ylabel,
                not_shown_legend,
                kinetic_set=kinetic_set,
            )
            # Create the output filename:
            unique_filename = os.path.join(
                comparison_outdir,
                _safe_filename_part(
                    f"unique_{direction}_fba_flux_lineage_"
                    f"comparison_with_highlights"
                    f"_{reaction_id}_"
                    f"{sim1_short}_vs_{sim2_short}"
                )
                + ".html",
            )
        else:
            # Plot the lineage colored by legend:
            unique_fig = build_lineage_scatter_figure(
                unique_rxn_ids,
                unique_levels_by_id,
                unique_max_level,
                direction,
                reaction_id,
                unique_sim1_log,
                unique_sim2_log,
                unique_hover,
                f"{main_title} (filtered to include unique base reactions only)<br>{unique_subtitle}",
                xlabel,
                ylabel,
                not_shown_legend,
                kinetic_set=kinetic_set,
            )
            unique_filename = os.path.join(
                comparison_outdir,
                _safe_filename_part(
                    f"unique_{direction}_fba_flux_lineage_comparison"
                    f"_{reaction_id}_"
                    f"{sim1_short}_vs_{sim2_short}"
                )
                + ".html",
            )

        # Save the unique output:
        unique_fig.write_html(unique_filename)

    return {"metadata_path": comparison_outdir}
