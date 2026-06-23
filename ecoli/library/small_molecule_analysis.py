"""
Shared helpers for the small-molecule analysis plots.

These are the in-DB data-loading + categorization utilities used by both the
multiseed total-vs-free small molecule scatter plot
(``ecoli/analysis/multiseed/small_molecule_total_vs_free_counts_plotly.py``) and
the multiexperiment comparison small molecule scatter plot
(``ecoli/analysis/multiexperiment/small_molecule_comparison_plotly.py``).
"""

import re

import polars as pl

from ecoli.library.parquet_emitter import (
    ndidx_to_duckdb_expr,
    read_stacked_columns,
)

_TAG_RE = re.compile(r"\[[a-z]\]$")


def count_cells(conn, history_sql):
    """Number of distinct cells (seed x generation x agent) in the data."""
    subquery = read_stacked_columns(history_sql, ["time"], order_results=False)
    return conn.execute(
        f"""
        SELECT count(*) FROM (
            SELECT DISTINCT experiment_id, variant, lineage_seed,
                            generation, agent_id
            FROM ({subquery})
        )
        """
    ).fetchone()[0]


def read_avg_listener_list(conn, history_sql, column):
    """Time-average the total per-small-molecule list column in DuckDB.

    Avoids pulling the full (n_timesteps x n_small_molecules) time series into
    memory by unnesting the list, tagging each element with its position,
    grouping by position, and then averaging. Returns the per-small-molecule
    averages (in list order).
    """
    subquery = read_stacked_columns(
        history_sql, [f"{column} AS v"], order_results=False
    )
    df = conn.sql(
        f"""
        WITH unnested AS (
            SELECT unnest(v) AS val, generate_subscripts(v, 1) AS sm_idx
            FROM ({subquery})
        )
        SELECT avg(val) AS avg_val
        FROM unnested
        GROUP BY sm_idx
        ORDER BY sm_idx
        """
    ).pl()
    return df["avg_val"].to_numpy()


def read_avg_free_from_bulk(conn, history_sql, sim_data, metabolite_ids):
    """Time-average the free count of each small molecule from the bulk table.

    Avoids pulling one DuckDB column per species (named_idx) and averaging in
    Python and instead slices the bulk array down to the tracked species as a
    SINGLE list column (``ndidx_to_duckdb_expr`` -> ``list_select``), then
    unnest + group + average entirely inside DuckDB. Returns the averages
    per-small-molecule (in the same order as the input list).
    """
    bulk_ids = list(sim_data.internal_state.bulk_molecules.bulk_data["id"])
    bname_to_idx = {n: i for i, n in enumerate(bulk_ids)}
    idx = [bname_to_idx[m] for m in metabolite_ids]

    sublist_expr = ndidx_to_duckdb_expr("bulk", [idx])
    subquery = read_stacked_columns(history_sql, [sublist_expr], order_results=False)
    df = conn.sql(
        f"""
        WITH unnested AS (
            SELECT
                unnest("bulk") AS free_count,
                generate_subscripts("bulk", 1) AS sm_idx
            FROM ({subquery})
        )
        SELECT avg(free_count) AS avg_free
        FROM unnested
        GROUP BY sm_idx
        ORDER BY sm_idx
        """
    ).pl()
    return df["avg_free"].to_numpy()


# Generate categorization for sequestration phenotypes a small molecule in
# the simulation can have (i.e. in an eq complex / TCS / DNA-bound TF):
CATEGORY_COLORS = [
    ("No sequestration", "lightseagreen"),
    ("In eq complex", "darkorange"),
    ("In TCS", "magenta"),
    ("In DNA-bound TF", "royalblue"),
    ("In eq + TCS", "mediumpurple"),
    ("In eq + bound TF", "green"),
    ("In TCS + bound TF", "crimson"),
    ("In eq + TCS + bound TF", "black"),
]


def categorize_expr():
    """Assign each small molecule an 8-way category label."""
    has_eq = pl.col("avg_eq_bound") > 0
    has_tcs = pl.col("avg_tcs_bound") > 0
    has_btf = pl.col("avg_bound_tf") > 0
    return (
        pl.when(has_eq & has_tcs & has_btf)
        .then(pl.lit("In eq + TCS + bound TF"))
        .when(has_eq & has_tcs)
        .then(pl.lit("In eq + TCS"))
        .when(has_eq & has_btf)
        .then(pl.lit("In eq + bound TF"))
        .when(has_tcs & has_btf)
        .then(pl.lit("In TCS + bound TF"))
        .when(has_eq)
        .then(pl.lit("In eq complex"))
        .when(has_tcs)
        .then(pl.lit("In TCS"))
        .when(has_btf)
        .then(pl.lit("In DNA-bound TF"))
        .otherwise(pl.lit("No sequestration"))
        .alias("sequestration_type")
    )


def _bare(species_id: str) -> str:
    """Strip a trailing [x] compartment tag, if present."""
    return _TAG_RE.sub("", species_id)


def _tag(species_id: str) -> str:
    """Return the trailing [x] tag (including brackets), or '' if none."""
    m = _TAG_RE.search(species_id)
    return m.group(0) if m else ""


def resolve_highlights(highlight_entries, small_molecule_ids):
    """Resolve the highlight list passed in against the available small molecule ids list.

    NOTE: Since this function takes in a user specified small_molecules_ids
    list, there is a chance that a molecule passed in through highlight_entries
    is still present in the simulation, but not the small_molecule_ids list.
    This function will not catch that case, and will return a message saying
    the molecule is not present in the small_molecule_ids list specifically
    passed in through this function. It is encoded this way to catch certain
    instances where a small molecule may not be present in a specific
    list/group, rather than the entire simulation.

    Returns (highlight_set, messages):
        highlight_set: set of fully-tagged species ids to highlight
        messages: list of strings to print
    """
    bare_to_tags = {}
    for sid in small_molecule_ids:
        bare_to_tags.setdefault(_bare(sid), []).append(_tag(sid))
    for k in bare_to_tags:
        bare_to_tags[k] = sorted(set(bare_to_tags[k]))

    highlight_set = set()
    messages = []

    for entry in highlight_entries:
        base = _bare(entry)
        entry_tag = _tag(entry)
        tags = bare_to_tags.get(base)

        # If the base name isn't included in the small_molecule_ids list passed
        # through, print a message saying so and skip it:
        if not tags:
            messages.append(
                f"'{entry}' is not present in the "
                f"'small_molecule_ids' list passed through "
                f"resolve_highlights(); skipping."
            )
            continue

        # If the user specifies a tag that does not exist for a base molecule,
        # print a message saying so and skip it:
        if entry_tag:
            if entry_tag not in tags:
                messages.append(
                    f"'{entry}' has tag {entry_tag} which is not present in the "
                    f"'small_molecule_ids' list passed through resolve_highlights(); valid tags for {base} present "
                    f"in the list passed through are ({', '.join(tags)})."
                )
                continue
            highlight_set.add(base + entry_tag)
            others = [t for t in tags if t != entry_tag]
            # If a user specifies a valid tag but there are also other tagged
            # versions of that that molecule that exist and are tracked, let
            # them know so they are aware they can plot those as well (Pi is a
            # good example of a small molecule with nonzero counts in multiple
            # compartments that one may want to be able to see together):
            if others:
                messages.append(
                    f"NOTE: {base} also has {len(others)} other valid "
                    f"compartment tag(s) "
                    f"({', '.join(base + t for t in others)}) within the 'small_molecule_ids' list passed through; change input "
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
