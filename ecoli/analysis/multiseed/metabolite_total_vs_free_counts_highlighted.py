"""
Metabolite Total vs Free Counts with highlighting capabilities!

Same as metabolite_total_vs_free_counts.py, but instead of coloring points
by sequestration type, ALL points are plotted in one color and a
user-specified list of metabolites is highlighted in red so they are easy
to locate on the plot.

Set HIGHLIGHT_METABOLITES at the top of this file to the metabolite IDs
to highlight (with or without compartment tags, e.g. 'ATP' or 'ATP[c]').

Compartment-tag messaging
-------------------------
HIGHLIGHT_METABOLITES entries may be bare ('ATP') or compartment tagged
('ATP[c]'):
  * bare entry, >1 compartment present  -> highlight ALL variants and note that
    the user can specify a single tag.
  * tagged entry, other variants present -> highlight just that species and note
    the other valid tags.
  * single compartment present -> highlight it silently.
These NOTE messages are printed out as the plot progresses for the user to see.

Free counts are read straight from the standard ``bulk`` table; total comes
from the listener; the sequestration breakdown is recomputed from sim_data.
"""

import os
import pickle
import re
from typing import Any

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    open_arbitrary_sim_data,
)
from ecoli.library.metabolite_sequestration import compute_avg_sequestration
from ecoli.analysis.multiseed.metabolite_total_vs_free_counts import (
    count_cells,
    read_avg_free_from_bulk,
    read_avg_listener_list,
)

# Metabolites to highlight in red (with or without compartment tags)
HIGHLIGHT_METABOLITES = ["ATP", "LEU[c]", "Pi", "CPD-12261", "TRP"]

_TAG_RE = re.compile(r"\[[a-z]\]$")


def _bare(species_id: str) -> str:
    """Strip a trailing [x] compartment tag, if present."""
    return _TAG_RE.sub("", species_id)


def _tag(species_id: str) -> str:
    """Return the trailing [x] tag (including brackets), or '' if none."""
    m = _TAG_RE.search(species_id)
    return m.group(0) if m else ""


def resolve_highlights(highlight_entries, metabolite_ids):
    """Resolve HIGHLIGHT_METABOLITES against the tracked species.

    Returns (highlight_set, messages):
        highlight_set: set of fully-tagged species ids to highlight
        messages: list of strings to print
    """
    bare_to_tags = {}
    for sid in metabolite_ids:
        bare_to_tags.setdefault(_bare(sid), []).append(_tag(sid))
    for k in bare_to_tags:
        bare_to_tags[k] = sorted(set(bare_to_tags[k]))

    highlight_set = set()
    messages = []

    for entry in highlight_entries:
        base = _bare(entry)
        entry_tag = _tag(entry)
        tags = bare_to_tags.get(base)

        if not tags:
            messages.append(f"'{entry}' is not tracked by the listener; skipping.")
            continue

        if entry_tag:
            if entry_tag not in tags:
                messages.append(
                    f"'{entry}' has tag {entry_tag} which is not present; "
                    f"valid tags for {base} are "
                    f"({', '.join(tags)}); skipping."
                )
                continue
            highlight_set.add(base + entry_tag)
            others = [t for t in tags if t != entry_tag]
            if others:
                messages.append(
                    f"NOTE: {base} also has {len(others)} other valid "
                    f"compartment tag(s) "
                    f"({', '.join(base + t for t in others)}); change input "
                    f"to '{base}' to plot all options for highlighting."
                )
        else:
            if len(tags) > 1:
                for t in tags:
                    highlight_set.add(base + t)
                messages.append(
                    f"{base} has multiple compartment tags "
                    f"({', '.join(base + t for t in tags)}); highlighting "
                    f"all by default. Specify {base + tags[0]} (etc.) to only "
                    f"highlight a specific compartment."
                )
            else:
                highlight_set.add(base + tags[0])

    return highlight_set, messages


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    success_sql: str,
    sim_data_dict: dict[str, dict[int, str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[str, dict[int, Any]],
    variant_names: dict[str, str],
):
    # Load metabolite IDs from listener metadata
    metabolite_ids = field_metadata(
        conn, config_sql, "listeners__metabolite_counts__totalMetaboliteCounts"
    )

    # Total from the listener; free straight from the standard bulk table:
    avg_total = read_avg_listener_list(
        conn, history_sql, "listeners__metabolite_counts__totalMetaboliteCounts"
    )

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    avg_free = read_avg_free_from_bulk(conn, history_sql, sim_data, metabolite_ids)

    # Recompute the time-averaged sequestration breakdown:
    seq = compute_avg_sequestration(conn, history_sql, sim_data, metabolite_ids)
    avg_eq = seq["avg_eq"]
    # Combine both TCS sources (Pi in phospho-proteins + ligand in TCS complexes)
    avg_tcs = seq["avg_tcs_pi"] + seq["avg_tcs_complex"]
    avg_bound_tf = seq["avg_bound_tf"]

    # Resolve the highlight list and emit compartment-tag messaging:
    highlight_set, messages = resolve_highlights(
        HIGHLIGHT_METABOLITES, list(metabolite_ids)
    )
    print("--- Highlight compartment-tag messaging ---")
    for msg in messages:
        print(msg)
    if not messages:
        print("(no messages)")

    is_highlighted = [sid in highlight_set for sid in metabolite_ids]

    # Plot every metabolite ever present (skipping those with entirely zero
    # counts that cannot go on a log axis):
    n_zero = int((avg_total <= 0).sum())
    n_cells = count_cells(conn, history_sql)
    print(
        f"{n_zero} of {len(metabolite_ids)} metabolites had zero counts the "
        f"entire simulation (not plotted)."
    )

    # Build per-metabolite DataFrame
    df = (
        pl.DataFrame(
            {
                "metabolite": metabolite_ids,
                "avg_total": avg_total.tolist(),
                "avg_free": avg_free.tolist(),
                "avg_eq_bound": avg_eq.tolist(),
                "avg_tcs_bound": avg_tcs.tolist(),
                "avg_bound_tf": avg_bound_tf.tolist(),
                "highlighted": is_highlighted,
            }
        )
        .with_columns(
            [
                (pl.col("avg_total") - pl.col("avg_free")).alias("avg_bound_total"),
                (pl.col("avg_free") / pl.col("avg_total").clip(lower_bound=1e-9)).alias(
                    "fraction_free"
                ),
            ]
        )
        .filter(pl.col("avg_total") > 0)
    )

    # Group label with per-group count in parentheses for the legend:
    n_highlighted = int(df["highlighted"].sum())
    n_other = len(df) - n_highlighted
    df = df.with_columns(
        pl.when(pl.col("highlighted"))
        .then(pl.lit(f"Highlighted ({n_highlighted})"))
        .otherwise(pl.lit(f"Other ({n_other})"))
        .alias("highlight_group")
    )

    # Tooltip
    tooltip = [
        alt.Tooltip("metabolite:N", title="Metabolite"),
        alt.Tooltip("avg_total:Q", title="Avg total", format=".1f"),
        alt.Tooltip("avg_free:Q", title="Avg free", format=".1f"),
        alt.Tooltip("avg_bound_total:Q", title="Avg bound (total)", format=".1f"),
        alt.Tooltip("avg_eq_bound:Q", title="  in eq complexes", format=".1f"),
        alt.Tooltip("avg_tcs_bound:Q", title="  in TCS", format=".1f"),
        alt.Tooltip("avg_bound_tf:Q", title="  in DNA-bound TFs", format=".1f"),
        alt.Tooltip("fraction_free:Q", title="Fraction free", format=".3f"),
    ]

    # Build scatter plot (log-log)
    min_val = float(df["avg_total"].min())
    max_val = float(df["avg_total"].max())
    log_scale = alt.Scale(type="log", domain=[min_val * 0.8, max_val * 1.25])

    ref_line = (
        alt.Chart(pl.DataFrame({"x": [min_val * 0.8, max_val * 1.25]}))
        .mark_line(strokeDash=[4, 2], color="black", opacity=0.5)
        .encode(
            x=alt.X("x:Q", scale=log_scale),
            y=alt.Y("x:Q", scale=log_scale),
        )
    )

    # All points one neutral color; highlighted inputs red and w/ larger size:
    color_scale = alt.Scale(
        domain=[f"Other ({n_other})", f"Highlighted ({n_highlighted})"],
        range=["lightslategray", "red"],
    )

    base = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("avg_total:Q", scale=log_scale, title="Log Average Total Count"),
            y=alt.Y("avg_free:Q", scale=log_scale, title="Log Average Free Count"),
            color=alt.Color("highlight_group:N", scale=color_scale, title="Group"),
            size=alt.condition(alt.datum.highlighted, alt.value(110), alt.value(35)),
            opacity=alt.condition(
                alt.datum.highlighted, alt.value(1.0), alt.value(0.45)
            ),
            order=alt.Order("highlighted:Q", sort="ascending"),
            tooltip=tooltip,
        )
    )

    chart = (
        (ref_line + base)
        .properties(
            title=f"Total vs Free Metabolite Counts (n={len(df)}, time-averaged over {n_cells} cells)",
            width=600,
            height=600,
        )
        .configure_view(fill="white")
        .interactive()
    )

    chart.save(os.path.join(outdir, "metabolite_total_vs_free_counts_highlighted.html"))
