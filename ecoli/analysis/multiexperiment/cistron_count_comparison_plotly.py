"""
# TODOs:
# TODO: confirm how tRNA are implemented post maturation (double check that the
 maturation process is the same for tRNA and rRNA on more time)
# TODO: decide whether or not to include rRNAs in ribosomal complexes within
the total counts on the plots (this might be interesting to include as it would
give an idea on how many ribosomes are present in each sim)
# TODO: fix the output folder location logic
# TODO: figure out why new sims still say plotted 4537/4538 total (where is
ftsO being listed? dont want it to forever say plotted 4537/4538 total, might
need to implement a specific fix for that)
# TODO: write a special note about ftsO in the docstring and consider making
some sort of skip fix for it in the plot.
# TODO: do a test of operons off vs on of this plot and see if another special
note should be addded to describe what to expect in that situation.


Multiexperiment RNA count comparison scatter plots

Produces two .html plots:
  1. cistron_count_comparison_by_type_<exp1>_vs_<exp2>.html: every RNA
     colored by type (mRNA / tRNA / rRNA / miscRNA), with the number of each in
     the legend.
  2. cistron_count_comparison_highlighted_<exp1>_vs_<exp2>.html: every RNA
     in one color, except for those in a user-specified highlight list (red). If
     no highlight list is given, every RNA is simply shown in the one color.

PLOT #1:
This analysis plots the average RNA counts from two simulation outputs against
each other on a log-log scatter. mRNA cistron counts come from the per-cistron
listener (listeners__rna_counts__mRNA_cistron_counts, indexed by
mRNA_cistron_ids). non-mRNA RNAs (tRNA / rRNA / miscRNA) are not included
in that listener, so their full-length counts are read from the bulk
container instead. This split reflects how the model stores transcripts: when a
transcript finishes elongating, only non-mRNA full-length transcripts are
removed from the unique "RNAs" store and have counts added as bulk molecules,
whereas full-length mRNAs REMAIN unique molecules (so translation/degradation
can act on them individually). mRNAs are therefore never tracked in bulk.
Every TU with is_mRNA == False is moved to bulk when its transcript
terminates -- tRNA TUs, rRNA TUs, and standalone miscRNA TUs after transcription.
tRNA/rRNA are then cleaved by RNA maturation into mature per-cistron species,
and a standalone miscRNA is counted as is. Each experiment's sim_data is
loaded separately to enumerate the non-mRNA RNAs, their types, and their bulk
indices separately, in case the two sims differ in their bulk-molecule set or
ordering (e.g. new TUs added/removed between simulations). The same per-sim
handling applies to mRNA cistrons (see below), where their id lists are read
per-experiment, so added/removed genes do not silently mispair points.

There is also an "other" category for any RNA that is neither mRNA nor
tRNA/rRNA/miscRNA. This is meant to catch cases where "pseudo" and "phantom"
RNAs make it into countable bulk molecules somehow. A (bulk) transcription unit
made only of such cistrons would therefore fall into "other". If "other" points
appear, determine the RNA's actual type and add it as its own
category/color to the plot.

Cistron IDs are the EcoCyc gene RNA ids (e.g. murD = EG10620_RNA) saved
within the sim_data that aligns with the values in the RNA_counts listener.
Non-mRNA RNAs are labeled by their bulk id (e.g. cysT = cysT-tRNA[c]).

All non-mRNA RNAs (tRNA / rRNA / miscRNA / "other") are read from the bulk
container and each type is handled differently based how the model processes it:

1. tRNA and rRNA: These transcribed (often as polycistronic operons) and then
  processed within RNA maturation, where unprocessed TUs are cleaved from their
  full-length operon transcript into separate mature tRNA/rRNA cistrons. Both
  forms are tracked in bulk (the unprocessed full-length transcript (one per
  operon) and each mature cistron product). This plot therefore shows tRNA/rRNA
  per cistron, where each point is one tRNA/rRNA cistron,
  and its plotted count is the total across both forms (mature + unprocessed).
  Note: a monocistronic tRNA/rRNA transcribed directly as mature simply has an
  unprocessed count of 0.

2. mRNA: mRNA are not processed the same way as non-mRNAs. They do not have
  any unprocessed-vs-mature split and their counts are recorded in the unique
  molecule space (not the bulk molecule space). Their counts are convienently
  tallied up in the rna_counts.py listener so one does not have to reconstruct
  their counts from the unique molecule container in analyses. An mRNA cistron
  is "made" as a single form, so mRNA points on the plot are a per-cistron count.

3. miscRNA: these are read from their single whole ``bulk`` entry (one count, no
  mature/unprocessed split), like a monocistronic non-mRNA transcript.

4. "other" (pseudo/phantom RNAs): If they existed, these would be read from
  their single whole bulk entry (one count, no split).

Another thing to note about how these plots are constructed is that mRNA cistrons
and non-mRNA RNAs are aligned by ID across the two sims, never by array
position. Only RNAs with a retrievable count in both sims sources are plotted
(i.e. the mRNA cistron listener for mRNAs, or bulk for non-mRNAs). For mRNA
cistrons, each sim's per-cistron counts are keyed by that experiment's own
listener id metadata (field_metadata returns only the first config row,
so the metadata is read per-experiment by filtering on experiment_id; this
matters if the two sims' mRNA cistron id lists differ).

For non-mRNA RNAs, experiment 1's count is read at sim_data_1's bulk index for
that id and experiment 2's at sim_data_2's bulk index because integer bulk
indices are not interchangeable between sims if the reconstructions differ.
In short, the two sims being plotted may differ in their cistron/RNA set or
ordering (e.g. genes added/removed, new transcription units), so positional
alignment would silently mispair points, and using id matching avoids that.

PLOT #2:
The second plot highlights a user-defined set of cistrons of interest in red
while all other RNAs are one color. The scatter is otherwise constructed the
same way as the first plot (same set of plotted RNAs).

OTHER NOTES:

Why miscRNAs usually read as zero counts always (dataset note):
As June 2026, no miscRNA gene has a measured value in the experimental
RNA-seq table used to seed basal expression
(reconstruction/ecoli/flat/rna_seq_data/rnaseq_rsem_tpm_mean.tsv, read in
reconstruction/ecoli/dataclasses/process/transcription.py to build
cistron_expression). A gene missing from that table defaults to
seq_data.get(gene_id, 0.0), which propagates to rna_expression["basal"] and
therefore rna_synth_prob["basal"] being 0. This is purely a dataset artifact,
not a hardcoded model rule. If a future experimental dataset did report values
for these miscRNA genes, they would receive nonzero basal expression and
synthesis probability and could then accumulate real counts in theory.

Special case of ftsO: ftsO is a miscRNA that technically has counts when it is
produced in the mRNA following mRNA operons: TU0-14439, TU0-14443, and TU0-941.
However, in reality, to obtain ftsO counts from ftsI (the mRNA that encapsulates
it), it needs to undergo some post-transcriptional processing that is currently
not modeled. Thus, the artifact counts that appear for it in unique molecules
are not realistic, and thus, there is no tracking listener for miscRNA in the
rna_counts.py listener currently. Its counts should be zero in actuality.

RNA cistrons mismatches between sims: an RNA with a retrievable count in only
one sim is not plotted. This happens when an RNA is present in one sim's
sources but not the other's. Each such RNA is dropped from the plot and
reported to stdout, naming the sim it is retrievable in and whether its counts
live in the mRNA listener or bulk, plus a reminder to double-check manually
where that cistron's counts are stored (because bulk may not be the right
location and should not be treated as a default here when
we have some cistron counts (mRNAs) being stored in unique molecules). If one
were to implement some change to where a count for a particular cistron is
stored, this plot would probably need to be edited to account for it properly.

Highlight gene list resolving: the user-supplied highlight input cistrons are
normalized to base ids by a single resolver captured from sim 1 only (see
build_non_mrna_bulk_records). The resolver just maps each input cistron (an RNA id, a
gene id, or a {gene}_RNA form input) to a base id, and via the gene lookup that base
id is a cistron id (cistron_data["id"], keyed by gene_id). This resolver is
reconstruction-dependent, but reusing sim 1's is safe here because the resolved
ids are only matched against point_ids, which are already the shared set of
RNAs plotted in both sims. So the only thing at stake is which of those already
plotted, shared points get highlighted. Sim 1's resolver could pick the wrong
shared point or miss one only if the two sims map the same gene via a different
cistron id (which would only happen in the case of different reconstructions
containing different cistron ids for the same gene).

This analysis may need to be updated for the following reasons if ever applicable:
- If non-mRNA RNA counts are updated to be tracked and updated somewhere other
  than bulk (for example, in the unique molecule space)
- If mRNA cistron counts are updated to be tracked somewhere other than the
  rna_counts listener (highly unlikely though)
- If new RNA types are added to the model that have counts recorded somewhere
- If cistron names attached to gene ids change between reconstructions and the
  user is interested in seeing those cistrons plotted (some new function would
  probably have to be written to decide what cistron name to use to plot, etc.)
"""

import os
import pickle
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

RNA_TYPE_COLORS = {
    "mRNA": "lightseagreen",
    "miscRNA": "violet",
    "tRNA": "orange",
    "rRNA": "orangered",
    "other": "#7f7f7f",
}

RNA_TYPE_SIZES = {
    "mRNA": 3,
    "miscRNA": 6,
    "tRNA": 5,
    "rRNA": 5,
    "other": 8,
}

RNA_TYPE_OPACITIES = {
    "mRNA": 0.4,
    "miscRNA": 0.7,
    "tRNA": 0.7,
    "rRNA": 0.7,
    "other": 0.8,
}


def _strip_compartment(mol_id: str) -> str:
    """Remove a trailing compartment tag from a molecule id."""
    if mol_id.endswith("]") and len(mol_id) >= 3 and mol_id[-3] == "[":
        return mol_id[:-3]
    return mol_id


def build_non_mrna_bulk_records(sim_data):
    """Return (id_to_record, resolve_interest) for non-mRNA RNAs of one sim.

    id_to_record: maps each non-mRNA point id to
    (rna_type, mature_idx, unproc_idxs) for the sim_data passed in:

    tRNA / rRNA are emitted per cistron (and these RNAs exist in two bulk forms
    a mature species and the unprocessed full-length operon transcript(s) they
    are cleaved from). mature_idx is the bulk index of the mature species, and
    unproc_idxs is the list of bulk indices of the unprocessed full-length TU
    transcripts that contain the cistron (each unprocessed transcript holds
    one copy of the cistron).

    miscRNA / "other" RNAs are not maturation-processed, so they are emitted
    as a single whole bulk entry: mature_idx is that entry's bulk index and
    unproc_idxs is empty. miscRNA are read from bulk here like tRNA/rRNA.

    resolve_interest(highlight_id) maps a user-supplied highlight id (RNA id,
    gene id, or {gene}_RNA) to a base id (no compartment) for matching.
    """
    transcription = sim_data.process.transcription
    cistron_data = transcription.cistron_data
    rna_data = transcription.rna_data
    mapping = transcription.cistron_tu_mapping_matrix
    bulk_ids = sim_data.internal_state.bulk_molecules.bulk_data["id"].tolist()
    bulk_id_to_idx = {bid: i for i, bid in enumerate(bulk_ids)}

    rna_ids = rna_data["id"]
    is_unprocessed = rna_data["is_unprocessed"]

    # (rna_type, mature_idx_or_None, unproc_idxs)
    id_to_record: dict[str, tuple[str, "int | None", list[int]]] = {}

    # 1) tRNA / rRNA: one point per cistron, with mature + unprocessed pools.
    for ci, cistron_id in enumerate(cistron_data["id"]):
        if cistron_data["is_rRNA"][ci]:
            rna_type = "rRNA"
        elif cistron_data["is_tRNA"][ci]:
            rna_type = "tRNA"
        else:
            continue
        mature_id = f"{cistron_id}[c]"
        mature_idx = bulk_id_to_idx.get(mature_id)
        # Unprocessed full-length TU transcripts that contain this cistron and
        # exist as bulk molecules, each contributes one unprocessed copy
        unproc_idxs = [
            bulk_id_to_idx[rna_ids[j]]
            for j in mapping.getrow(ci).indices
            if is_unprocessed[j] and rna_ids[j] in bulk_id_to_idx
        ]
        if mature_idx is None and not unproc_idxs:
            continue
        id_to_record[mature_id] = (rna_type, mature_idx, unproc_idxs)

    # 2) miscRNA / "other": whole bulk entry, no maturation split. These are
    # keyed by cistron id (not the rna_data/TU id), so they reconcile with the
    # cistron-id-keyed mRNA/miscRNA listeners across sims:
    misc_other_mask = ~rna_data["is_mRNA"] & ~rna_data["is_rRNA"] & ~rna_data["is_tRNA"]
    for j, keep in enumerate(misc_other_mask):
        if not keep:
            continue
        full_id = rna_ids[j]
        if full_id not in bulk_id_to_idx:
            continue
        is_misc = bool(rna_data["is_miscRNA"][j])
        label = "miscRNA" if is_misc else "other"
        bulk_idx = bulk_id_to_idx[full_id]
        # Row (cistron) indices of nonzeros in column j
        cistron_rows = mapping.getcol(j).nonzero()[0]
        if len(cistron_rows) == 0:
            # if no mapped cistron, falls back to the rna_data id (should not
            # happen for sims with the same reconstruction)
            id_to_record[str(full_id)] = (label, bulk_idx, [])
            continue
        for ci in cistron_rows:
            id_to_record[str(cistron_data["id"][ci])] = (label, bulk_idx, [])

    base_set = {_strip_compartment(rid) for rid in rna_ids}
    base_set.update(cistron_data["id"])
    gene_to_rna = {g: c for g, c in zip(cistron_data["gene_id"], cistron_data["id"])}

    def resolve_interest(highlight_id: str) -> str:
        """Map a highlight id (RNA id, gene id, or {gene}_RNA) to a base id.

        Defined here so it captures this sim's base_set and gene_to_rna maps.
        The caller then gets a resolver already bound to this sim's data and
        can call resolve_interest(id) without carrying those maps around.
        """
        if highlight_id in base_set:
            return highlight_id
        if highlight_id.endswith("_RNA"):
            stem = highlight_id[:-4]
            if stem in base_set:
                return stem
            if stem in gene_to_rna:
                return gene_to_rna[stem]
        if highlight_id in gene_to_rna:
            return gene_to_rna[highlight_id]
        return highlight_id

    return id_to_record, resolve_interest


def build_name_lookup(sim_data):
    """Return lookup(point_id) -> (gene_name, description, operons) for
    hoverdata."""
    transcription = sim_data.process.transcription
    common = sim_data.common_names
    cistron_data = transcription.cistron_data
    cistron_id_to_gene = dict(zip(cistron_data["id"], cistron_data["gene_id"]))
    cistron_id_to_idx = {c: i for i, c in enumerate(cistron_data["id"])}
    tu_ids = list(transcription.rna_data["id"])
    tu_id_to_idx = {t: j for j, t in enumerate(tu_ids)}
    # (n_cistrons, n_tus): row = cistron, nonzero columns = TUs containing it:
    mapping = transcription.cistron_tu_mapping_matrix
    # Cistrons per TU; only polycistronic TUs (>1) are treated as real operons:
    tu_cistron_counts = mapping.getnnz(axis=0)

    def _tu_label(tu_id: str) -> str:
        # Strip the compartment tag (TU ids in rna_data look like "TU0-1181[c]")
        # so the common-name lookup hits, then append the TU's common name:
        bare = _strip_compartment(tu_id)
        name = common.get_common_name(bare)
        return f"{bare} ({name})" if name != bare else bare

    def operon_field(point_id: str, base: str) -> list[str]:
        ci = cistron_id_to_idx.get(point_id)
        if ci is None:
            ci = cistron_id_to_idx.get(base)
        if ci is not None:
            # A cistron can sit on several TUs at once: its own monocistronic
            # TU(s) and/or one or more multi-gene operons. List every TU it is
            # on, tagged by kind, so a gene whose count differs from an operon-
            # mate (because it is also transcribed from a monocistronic TU of
            # its own) is not confusing:
            tu_idxs = list(mapping.getrow(ci).indices)
            mono = [
                f"monocistronic TU: {_tu_label(tu_ids[j])}"
                for j in tu_idxs
                if tu_cistron_counts[j] == 1
            ]
            operons = [
                f"operon ({tu_cistron_counts[j]} genes): {_tu_label(tu_ids[j])}"
                for j in tu_idxs
                if tu_cistron_counts[j] > 1
            ]
            return mono + operons or ["NA"]
        # If the cistron id is itself a TU id, the whole transcript is plotted
        # to represent it (e.g. a non-mRNA RNA keyed by its own TU):
        j = tu_id_to_idx.get(point_id)
        if j is None:
            j = tu_id_to_idx.get(base)
        if j is None:
            return ["NA"]
        if tu_cistron_counts[j] > 1:
            # A polycistronic transcript (e.g. an rRNA operon) is not a member
            # of a larger operon, it is the whole operon transcript itself:
            return ["RNA corresponds to its own whole operon transcript"]
        return [f"monocistronic TU: {_tu_label(tu_ids[j])}"]

    def lookup(point_id: str) -> tuple[str, str, list[str]]:
        base = _strip_compartment(point_id)
        gene_id = cistron_id_to_gene.get(point_id) or cistron_id_to_gene.get(base)
        gene_name = common.get_common_name(gene_id) if gene_id else ""
        # Only keep the RNA's common name if it is different from the id and symbol:
        rna_name = common.get_common_name(base)
        description = rna_name if rna_name not in (base, point_id, gene_name) else "NA"
        operons = operon_field(point_id, base)
        return gene_name, description, operons

    return lookup


def read_bulk_rna_means_for_experiment(
    conn: DuckDBPyConnection,
    history_sql: str,
    experiment_id: str,
    bulk_indices: list[int],
    gen_floor: int,
) -> dict[int, float]:
    """Mean-of-per-cell-mean bulk counts for one experiment's bulk indices.

    Reads experiment_id's rows only and attributes the results to that
    experiment. Returns {bulk_idx: mean_count}. Mirrors the mRNA averaging
    (per-cell average over timepoints with generation >= gen_floor, then mean
    across cells). gen_floor is min_generation + skip_gens.

    NOTE: bulk_indices must be indices into the current experiment's sim_data
    bulk-molecule ordering. Integer bulk indices are NOT interchangeable between
    experiments, so this function must be called once per experiment with that
    experiment's own index list.
    """
    if not bulk_indices:
        return {}
    bulk_indices = [int(i) for i in bulk_indices]
    names = [f"bulkrna_{i}" for i in bulk_indices]
    bulk_expr = named_idx("bulk", names, [bulk_indices])
    subquery = cast(
        str, read_stacked_columns(history_sql, [bulk_expr], order_results=False)
    )
    avg_exprs = ", ".join(f'avg("{n}") AS "{n}"' for n in names)
    # Escape single quotes in the experiment id for the SQL literal:
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
    return {idx: row[f"bulkrna_{idx}"] for idx in bulk_indices}


def _add_parity_line(fig: go.Figure, sim1_log: np.ndarray, sim2_log: np.ndarray):
    """Add a dashed y = x reference line spanning the data range."""
    max_val = max(float(sim1_log.max()), float(sim2_log.max()))
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="black", dash="dash", width=0.5),
            opacity=0.7,
            name="y=x",
            showlegend=True,
            hoverinfo="skip",
        )
    )


def _add_stats_annotation(
    fig: go.Figure, r_value: float, pearson_r2: float, cod_r2: float
):
    """Add the Pearson / and coefficient of determination (COD) stats box
    (computed over all plotted RNAs)."""
    fig.add_annotation(
        x=0.95,
        y=0.05,
        xref="paper",
        yref="paper",
        text=(
            f"<b>Statistics (all RNAs):</b><br>"
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
    """Square log-axis layout shared by both figures."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=900,
        height=900,
        template="plotly_white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1, constrain="domain")
    fig.update_yaxes(constrain="domain")


def mrna_cistron_ids_for_experiment(
    conn: DuckDBPyConnection, config_sql: str, experiment_id: str, count_column: str
) -> list[str]:
    """The mRNA cistron id list (listener column index labels) for current experiment.

    count_column is the per-cistron listener column whose metadata holds the
    id labels. field_metadata reads the first config row, so to get a specific
    experiment's metadata the config subquery is filtered to experiment_id
    first. The two sims may differ in their mRNA cistron set/order (e.g. genes
    added/removed), so the caller must align the two by id, not by position.
    """
    exp_literal = experiment_id.replace("'", "''")
    filtered = f"SELECT * FROM ({config_sql}) WHERE experiment_id = '{exp_literal}'"
    return field_metadata(conn, filtered, count_column)


def build_by_type_figure(
    point_types, sim1_log, sim2_log, hover_texts, stats, title, xlabel, ylabel
):
    """Scatter colored by RNA type, with per-type counts in the legend.

    This function generates plot #1.
    """
    point_types_arr = np.array(point_types, dtype=object)
    present = set(point_types)
    type_order = ["mRNA", "tRNA", "rRNA", "miscRNA", "other"]

    fig = go.Figure()
    for rna_type in [t for t in type_order if t in present]:
        mask = point_types_arr == rna_type
        if mask.sum() > 0:
            fig.add_trace(
                go.Scatter(
                    x=sim1_log[mask],
                    y=sim2_log[mask],
                    mode="markers",
                    marker=dict(
                        color=RNA_TYPE_COLORS.get(rna_type, "#000000"),
                        size=RNA_TYPE_SIZES.get(rna_type, 4),
                        opacity=RNA_TYPE_OPACITIES.get(rna_type, 0.6),
                        line=dict(width=0),
                    ),
                    name=f"{rna_type} (n={int(mask.sum())})",
                    text=[hover_texts[i] for i in np.where(mask)[0]],
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=True,
                )
            )

    _add_parity_line(fig, sim1_log, sim2_log)
    _add_stats_annotation(fig, *stats)
    _apply_square_layout(fig, title, xlabel, ylabel)
    return fig


def build_highlighted_figure(
    highlight_mask, sim1_log, sim2_log, hover_texts, stats, title, xlabel, ylabel
):
    """Scatter with all RNAs one color except a highlighted set (red).

    This function generates plot #2.
    """
    fig = go.Figure()

    bg = ~highlight_mask
    if bg.sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=sim1_log[bg],
                y=sim2_log[bg],
                mode="markers",
                marker=dict(
                    color="lightseagreen", size=6, opacity=0.5, line=dict(width=0)
                ),
                name=f"All RNAs ({int(bg.sum())})",
                text=[hover_texts[i] for i in np.where(bg)[0]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )
    if highlight_mask.sum() > 0:
        fig.add_trace(
            go.Scatter(
                x=sim1_log[highlight_mask],
                y=sim2_log[highlight_mask],
                mode="markers",
                marker=dict(
                    color="red",
                    size=11,
                    opacity=0.95,
                    line=dict(width=1, color="black"),
                ),
                name=f"Highlighted ({int(highlight_mask.sum())})",
                text=[hover_texts[i] for i in np.where(highlight_mask)[0]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=True,
            )
        )

    _add_parity_line(fig, sim1_log, sim2_log)
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
    Plot average RNA counts from two simulations against each other, optionally
    highlighting a user-specified set of cistrons in red. Note that the
    averaging is done per cell generation, not over all time. Each generation
    average is thus given equal weight.

    Args:
        params: Dictionary containing parameters of the format:

            {
                # Number of initial generations worth of data to skip
                "skip_n_gens": int (default: 2),
                # RNAs to highlight in red. If omitted/empty, every RNA is
                # plotted in a single color (nothing highlighted):
                "highlight_rnas": list[str] (default: []),
            }
    """
    skip_gens = params.get("skip_n_gens", 2)
    cistrons_of_interest = params.get("highlight_rnas", [])

    # "skip the first skip_gens generations of each seed" -> keep generations
    # whose index is at least skip_gens ABOVE the minimum. Generations in the
    # data are 1-indexed (first gen = 1), so we derive the floor from the data's
    # actual minimum generation rather than assuming a 0- or 1-based start; this
    # keeps skip_gens=2 dropping exactly the first two generations per seed.
    min_gen = int(
        conn.sql(f"SELECT min(generation) AS g FROM ({history_sql})").pl()["g"][0]
    )
    gen_floor = min_gen + skip_gens

    # Per-cistron mRNA listener column read here (kept local rather than a
    # module-level constant so it lives next to the query that uses it):
    cistron_count_column = "listeners__rna_counts__mRNA_cistron_counts"

    # Per-cistron average counts (one averaged value per cell, skipping early
    # generations), grouped so each row is one cell's per-cistron average list:
    subquery = cast(
        str,
        read_stacked_columns(history_sql, [cistron_count_column], order_results=False),
    )
    all_counts = conn.sql(
        f"""
        WITH unnested_counts AS (
            SELECT unnest({cistron_count_column}) AS counts,
                generate_subscripts({cistron_count_column}, 1) AS idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({subquery})
            WHERE generation >= {gen_floor}
        ),
        avg_counts AS (
            SELECT avg(counts) AS avgCounts,
                experiment_id, variant, lineage_seed,
                generation, agent_id, idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, idx
        )
        SELECT list(avgCounts ORDER BY idx) AS avgCounts,
               experiment_id
        FROM avg_counts
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """
    ).pl()

    unique_exp_ids = all_counts["experiment_id"].unique().to_list()
    if len(unique_exp_ids) < 2:
        raise ValueError(
            f"Expected 2 experiments but found {len(unique_exp_ids)}: "
            f"{unique_exp_ids}. Make sure both experiment_ids are in the config."
        )

    # x-axis = the first experiment_id in the config, y-axis = the second
    # (config order, via sim_data_paths):
    present = set(unique_exp_ids)
    ordered_exp_ids = [e for e in sim_data_paths if e in present] or unique_exp_ids
    exp_id_1, exp_id_2 = ordered_exp_ids[0], ordered_exp_ids[1]

    print(f"Comparing {exp_id_1} (Sim 1; x-axis) vs {exp_id_2} (Sim 2; y-axis)")

    counts_exp1 = all_counts.filter(pl.col("experiment_id") == exp_id_1)
    counts_exp2 = all_counts.filter(pl.col("experiment_id") == exp_id_2)

    sim1_counts = ndlist_to_ndarray(counts_exp1["avgCounts"])
    sim2_counts = ndlist_to_ndarray(counts_exp2["avgCounts"])
    sim1_avg = sim1_counts.mean(axis=0)
    sim2_avg = sim2_counts.mean(axis=0)
    print(
        f"Averaging Sim 1 over {len(sim1_counts)} cells (skipped first "
        f"{skip_gens} generations of each seed)"
    )
    print(
        f"Averaging Sim 2 over {len(sim2_counts)} cells (skipped first "
        f"{skip_gens} generations of each seed)"
    )

    # mRNA cistron ids are the listener column's index labels, read per sim,
    # then aligned by id to avoid mispairing points:
    cistron_ids_1 = mrna_cistron_ids_for_experiment(
        conn, config_sql, exp_id_1, cistron_count_column
    )
    cistron_ids_2 = mrna_cistron_ids_for_experiment(
        conn, config_sql, exp_id_2, cistron_count_column
    )

    # Check that the number of cistron ids matches the length of the averaged
    # counts for each simulation (this should always pass if the mRNA listener
    # is properly aligned):
    if len(cistron_ids_1) != len(sim1_avg):
        raise ValueError(
            f"Sim 1 mRNA cistron id count ({len(cistron_ids_1)}) does not match "
            f"its averaged count length ({len(sim1_avg)})."
        )
    if len(cistron_ids_2) != len(sim2_avg):
        raise ValueError(
            f"Sim 2 mRNA cistron id count ({len(cistron_ids_2)}) does not match "
            f"its averaged count length ({len(sim2_avg)})."
        )

    # Determine the mean count for each cistron in each sim, keyed by cistron id:
    mrna_means_1 = dict(zip(cistron_ids_1, (float(x) for x in sim1_avg)))
    mrna_means_2 = dict(zip(cistron_ids_2, (float(x) for x in sim2_avg)))

    # Obtain the bulk container from each simulations' own sim_data:
    with open_arbitrary_sim_data({exp_id_1: sim_data_paths[exp_id_1]}) as f:
        sim_data_1 = pickle.load(f)
    with open_arbitrary_sim_data({exp_id_2: sim_data_paths[exp_id_2]}) as f:
        sim_data_2 = pickle.load(f)

    # Issue a warning if the two sims' bulk-molecule lengths differ (note: this
    # code by default  resolves indices per-sim by RNA id, but a differing bulk
    # set still means an RNA may be present in one sim and absent in the other):
    bulk_ids_1 = sim_data_1.internal_state.bulk_molecules.bulk_data["id"].tolist()
    bulk_ids_2 = sim_data_2.internal_state.bulk_molecules.bulk_data["id"].tolist()
    if len(bulk_ids_1) != len(bulk_ids_2) or bulk_ids_1 != bulk_ids_2:
        reason = (
            f"Sims have different bulk container lengths "
            f"({len(bulk_ids_1)} vs {len(bulk_ids_2)})"
            if len(bulk_ids_1) != len(bulk_ids_2)
            else f"Sims have the same bulk container size but "
            f"({len(bulk_ids_1)}) but different id ordering"
        )
        print(
            "WARNING: bulk molecule data differs between the two simulations\n"
            f"  [Sim 1: {len(bulk_ids_1)} ids, Sim 2: {len(bulk_ids_2)} ids: \n"
            f"  {reason}].\n"
            f"  This analysis is designed to only plot cistrons that are \n"
            f"  present in both sims and keys each cistron by its specific \n"
            f"  index within each sim to avoid misalignment by assuming \n"
            f"  the ordering is the same in both sims, but some RNAs present in \n"
            f"  one sim may be absent in the other (and will not be plotted as \n"
            f"  a result)."
        )

    # Non-mRNA RNA records per sim, keyed by cistron id, read from bulk  (note:
    # check the analysis docstring at the top for a point about the resolver
    # being used here):
    id_to_record_1, resolve_interest = build_non_mrna_bulk_records(sim_data_1)
    id_to_record_2, _ = build_non_mrna_bulk_records(sim_data_2)

    # Read each sim's bulk container from its own indices and gather
    # every index a record references (mature species + unprocessed TU
    # transcripts), then split each count into mature / unprocessed / total:
    def all_bulk_indices(records):
        idxs: set[int] = set()
        for _type, mature_idx, unproc_idxs in records.values():
            if mature_idx is not None:
                idxs.add(mature_idx)
            idxs.update(unproc_idxs)
        return sorted(idxs)

    means_1 = read_bulk_rna_means_for_experiment(
        conn, history_sql, exp_id_1, all_bulk_indices(id_to_record_1), gen_floor
    )
    means_2 = read_bulk_rna_means_for_experiment(
        conn, history_sql, exp_id_2, all_bulk_indices(id_to_record_2), gen_floor
    )

    def mature_unproc(records, means, rid):
        """(mature, unprocessed) bulk counts for one record in one sim.

        For tRNA/rRNA the two pools come from the mature species and the
        unprocessed TU transcript(s); for miscRNA/"other" unproc_idxs is empty
        so unprocessed is 0 and mature is the single whole-entry count.
        """
        _type, mature_idx, unproc_idxs = records[rid]
        mature = (
            float(means.get(mature_idx, 0.0) or 0.0) if mature_idx is not None else 0.0
        )
        unproc = sum(float(means.get(i, 0.0) or 0.0) for i in unproc_idxs)
        return mature, unproc

    # Unified per-sim resolver across the two sources: Each RNA is reconciled
    # across the two sims by its base id (with the compartment tag stripped)
    # and gets each sim's count from that sim's own source: the mRNA listener
    # for mRNA cistrons, bulk for everything else (tRNA/rRNA/miscRNA/other).
    _NAN = float("nan")

    def build_sim_lookup(mrna_means, records, means):
        """base_id -> (value, rna_type, mature, unproc, source) for one sim.

        "source" is "bulk" or "mRNA" -- where this sim's count was read from.
        """
        lookup: dict[str, tuple[float, str, float, float, str]] = {}
        for rid, (rtype, _m, _u) in records.items():
            mature, unproc = mature_unproc(records, means, rid)
            lookup[_strip_compartment(rid)] = (
                mature + unproc,
                rtype,
                mature,
                unproc,
                "bulk",
            )
        for cid, val in mrna_means.items():
            lookup[_strip_compartment(cid)] = (val, "mRNA", _NAN, _NAN, "mRNA")
        return lookup

    lookup_1 = build_sim_lookup(mrna_means_1, id_to_record_1, means_1)
    lookup_2 = build_sim_lookup(mrna_means_2, id_to_record_2, means_2)

    # Plot only RNAs with a retrievable count in both sims that is present in
    # both lookups (note: an RNA retrievable in only one sim will not be plotted):
    plotted_ids = [bid for bid in lookup_1 if bid in lookup_2]
    not_plotted_1 = {bid: rec for bid, rec in lookup_1.items() if bid not in lookup_2}
    not_plotted_2 = {bid: rec for bid, rec in lookup_2.items() if bid not in lookup_1}
    n_total = len(plotted_ids) + len(not_plotted_1) + len(not_plotted_2)
    point_ids = plotted_ids

    # Per-sim count / type / mature / unproc for every plotted point:
    point_types: list[str] = []
    point_avg_1: list[float] = []
    point_avg_2: list[float] = []
    point_mature_1: list[float] = []
    point_unproc_1: list[float] = []
    point_mature_2: list[float] = []
    point_unproc_2: list[float] = []
    # RNAs whose counts are stored in different sources across the two sims
    # (e.g. mRNA listener in one sim, bulk in the other). Collected here and
    # reported below; the point is still plotted, but its hover type is taken
    # from sim 1, so sim 2's count warrants a manual check.
    source_mismatches: list[tuple[str, str, str, str, str]] = []
    for bid in point_ids:
        v1, rtype, m1, u1, source_1 = lookup_1[bid]
        v2, rtype_2, m2, u2, source_2 = lookup_2[bid]
        if source_1 != source_2:
            source_mismatches.append((bid, source_1, source_2, rtype, rtype_2))
        point_types.append(rtype)
        point_avg_1.append(v1)
        point_avg_2.append(v2)
        point_mature_1.append(m1)
        point_unproc_1.append(u1)
        point_mature_2.append(m2)
        point_unproc_2.append(u2)
    point_avg_1 = np.array(point_avg_1)
    point_avg_2 = np.array(point_avg_2)
    point_mature_1 = np.array(point_mature_1)
    point_unproc_1 = np.array(point_unproc_1)
    point_mature_2 = np.array(point_mature_2)
    point_unproc_2 = np.array(point_unproc_2)
    n_by_type: dict[str, int] = {}
    for t in point_types:
        n_by_type[t] = n_by_type.get(t, 0) + 1
    print(
        f"Plotted cistrons (present in both sims): {len(point_ids)}/{n_total}"
        f" {n_by_type}"
    )

    # RNAs not plotted (retrievable in only one sim):
    # A base id in only one sim's lookup has no retrievable count in the other
    # is dropped from the plot and a print statement is issued explaining where
    # its counts live in the sim that does have it:
    src_phrase = {"mRNA": "the mRNA listener", "bulk": "bulk (defined in rna_data)"}

    def _not_plotted_msgs(records, present_exp, other_exp):
        out = []
        for rna, (_v, rtype, _m, _u, source) in records.items():
            out.append(
                f"- {rtype} {rna}: only has retrievable counts in \n"
                f"  {present_exp} (counts live in {src_phrase.get(source, source)}); \n"
                f"  not in {other_exp}'s mRNA listener or bulk. \n"
                f"  Double-check manually where this cistron's counts are \n"
                f"  stored across sims, as it may have a bulk id but the \n"
                f"  bulk container may not be where the counts are actually \n"
                f"  updated each timestep."
            )
        return out

    not_plotted_msgs = _not_plotted_msgs(
        not_plotted_1, exp_id_1, exp_id_2
    ) + _not_plotted_msgs(not_plotted_2, exp_id_2, exp_id_1)
    if not_plotted_msgs:
        print(
            f"NOTE: {len(not_plotted_msgs)} RNA(s) not plotted (retrievable in only "
            "one sim):"
        )
        for msg in not_plotted_msgs:
            print(f"  {msg}")

    # RNAs plotted but have counts recorded in different containers across sims:
    # There is a chance that cistron could be present in both sims but have its
    # counts stored in different containers if the type of RNA changes (i.e.
    # if something were to go from mRNA to tRNA). This sort of change would be
    # very unusual and unlikely, hence why this warning is built in to flag
    # when/if it happens. The point can still be plotted if the counts are
    # obtainable in both sims, but its hover data type is assigned from sim 1,
    # so the flag also prompts the user to manually check how sim 2's total is
    # reported (if something is an mRNA in Sim 1 but a tRNA in sim2, the hover
    # data will be the mRNA type hover data, with the total for Sim 2 reflecting
    # the unprocessed + mature count, but no breakdown will be shown).
    if source_mismatches:
        print(
            f"NOTE: {len(source_mismatches)} RNA(s) plotted but their counts "
            f"are stored in different areas of the model across the two sims:"
        )
        for rna, source_1, source_2, rtype, rtype_2 in source_mismatches:
            print(
                f"- RNA {rna} is recorded within {src_phrase.get(source_1, source_1)} \n"
                f"  of Sim 1 and {src_phrase.get(source_2, source_2)} of \n"
                f"  Sim 2. Automatically assigning its type in the hover \n"
                f"  data as consistent with sim 1's type ({rtype}); sim 2 reports \n"
                f"  it as {rtype_2}. Double check Sim 2's counts manually."
            )

    # Compute log10(count + 1) for each point in each sim (for plotting and stats):
    sim1_log = np.log10(point_avg_1 + 1)  # Add 1 to avoid log10(0)
    sim2_log = np.log10(point_avg_2 + 1)

    # Stats (computed over all plotted RNAs):
    r_value = pearsonr(sim1_log, sim2_log)[0]
    pearson_r2 = r_value**2
    cod_r2 = r2_score(sim2_log, sim1_log)

    # Build the highlight mask: match interest input cistrons against each
    # point's id or base id (compartment stripped), resolving gene/{gene}_RNA
    # forms via sim_data:
    interest_bases = {resolve_interest(t) for t in cistrons_of_interest}
    interest_raw = set(cistrons_of_interest)
    highlighted_mask = np.array(
        [
            (pid in interest_raw) or (_strip_compartment(pid) in interest_bases)
            for pid in point_ids
        ]
    )

    # Hover text: shared identity fields (id / type / gene / description) on top,
    # then a per-sim block with the fields that can differ between the two sims
    # (operon membership and counts), each computed from that sim's sim_data:
    name_lookup_1 = build_name_lookup(sim_data_1)
    name_lookup_2 = build_name_lookup(sim_data_2)

    def tu_lines(entries: list[str], indent: str = "  ") -> list[str]:
        # One TU per line (each entry on its own line, including a single
        # monocistronic TU):
        return [f"{indent}Transcription units:"] + [f"{indent}  - {e}" for e in entries]

    def count_lines(
        rna_type: str, total: float, mature: float, unproc: float, indent="  "
    ) -> list[str]:
        # tRNA/rRNA exist as a mature pool plus the unprocessed full-length
        # transcript(s) they are cleaved from, so report both and their total
        # with the mature/unprocessed ratio (all other types are a single count):
        if rna_type in ("tRNA", "rRNA"):
            ratio = (
                f"{mature / unproc:.2f}"
                if unproc > 0
                else "n/a (no unprocessed pool; monocistronic)"
            )
            return [
                f"{indent}mature RNA count: {mature:.2f}",
                f"{indent}unprocessed count within operons: {unproc:.2f}",
                f"{indent}total count: {total:.2f} (ratio m/u = {ratio})",
            ]
        return [f"{indent}count: {total:.2f}"]

    hover_texts = []
    for i, pid in enumerate(point_ids):
        gene_name, description, operons_1 = name_lookup_1(pid)
        _, _, operons_2 = name_lookup_2(pid)
        lines = [
            f"<b>{pid}</b> ({point_types[i]})",
            f"Gene ID: {gene_name or 'NA'}",
            f"Description (if applicable): {description or 'NA'}",
            "",
            f"<b>Sim 1</b> ({exp_id_1})",
            *tu_lines(operons_1),
            *count_lines(
                point_types[i], point_avg_1[i], point_mature_1[i], point_unproc_1[i]
            ),
            f"  log: {sim1_log[i]:.3f}",
            "",
            f"<b>Sim 2</b> ({exp_id_2})",
            *tu_lines(operons_2),
            *count_lines(
                point_types[i], point_avg_2[i], point_mature_2[i], point_unproc_2[i]
            ),
            f"  log: {sim2_log[i]:.3f}",
        ]
        hover_texts.append("<br>".join(lines))

    stats = (r_value, pearson_r2, cod_r2)

    def extract_short_id(exp_id):
        """Extract a short identifier from the full experiment ID (to be used
        in the comparison plots' file names).
        """
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

    # Cell counts are the post-skip cells averaged over (the first skip_gens
    # generations of each seed are dropped):
    n_cells_1 = len(sim1_counts)
    n_cells_2 = len(sim2_counts)

    # "n_plotted" is how many of the total candidate RNAs (present in either sim)
    # were actually plotted (present and countable in both sims):
    n_plotted = len(point_ids)
    subtitle = (
        f"<sub>Sim 1 (x): {exp_id_1} "
        f"(n={n_plotted}/{n_total} RNAs plotted; averaged over {n_cells_1} cells) vs. "
        f"<br>Sim 2 (y): {exp_id_2} "
        f"(n={n_plotted}/{n_total} RNAs plotted; averaged over {n_cells_2} cells)</sub>"
    )
    xlabel = "log10(Sim 1 RNA Counts + 1)"
    ylabel = "log10(Sim 2 RNA Counts + 1)"

    # Plot 1: category breakdown by RNA type (per-type counts in legend):
    fig_type = build_by_type_figure(
        point_types,
        sim1_log,
        sim2_log,
        hover_texts,
        stats,
        f"Average RNA Cistron Count Comparison by Type<br>{subtitle}",
        xlabel,
        ylabel,
    )
    type_filename = os.path.join(
        comparison_outdir,
        f"cistron_count_comparison_by_type_{sim1_short}_vs_{sim2_short}.html",
    )
    fig_type.write_html(type_filename)

    # Plot 2: everything one color, with any user-requested RNAs highlighted in
    # red:
    hi_title = "RNA Cistron Count Comparison"
    fig_hi = build_highlighted_figure(
        highlighted_mask,
        sim1_log,
        sim2_log,
        hover_texts,
        stats,
        f"{hi_title}<br>{subtitle}",
        xlabel,
        ylabel,
    )
    hi_filename = os.path.join(
        comparison_outdir,
        f"cistron_count_comparison_highlighted_{sim1_short}_vs_{sim2_short}.html",
    )
    fig_hi.write_html(hi_filename)

    # Print a summary of the highlight results:
    if cistrons_of_interest:
        print(
            f"Highlighted RNAs summary: highlighted {int(highlighted_mask.sum())} of "
            f"{len(cistrons_of_interest)} requested."
        )
    else:
        print(
            "No highlight list provided for the highlight plot, so all RNAs "
            "are shown in a single color."
        )

    # Override the default metadata saving file path:
    return {"metadata_path": comparison_outdir}
