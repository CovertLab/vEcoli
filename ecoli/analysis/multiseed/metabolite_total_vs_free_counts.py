"""
Metabolite Total vs Free Counts (y=x scatter)

For each tracked metabolite, plots the time-averaged total count (x-axis)
against the time-averaged free count (y-axis). Points on the y=x line have
all their counts in the free pool. Points below the line have metabolites
sequestered in equilibrium complexes, TCS phosphorylated molecules, or
DNA-bound transcription factors.

Free counts are read straight from the standard ``bulk`` table; total comes
from the listener's totalMetaboliteCounts; the sequestration breakdown is
recomputed from sim_data stoich maps.
"""

import os
import pickle
from typing import Any

import altair as alt
import polars as pl
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    field_metadata,
    ndidx_to_duckdb_expr,
    open_arbitrary_sim_data,
    read_stacked_columns,
)
from ecoli.library.metabolite_sequestration import compute_avg_sequestration


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
    """Time-average the total per-metabolite list column in DuckDB.

    Avoids pulling the full (n_timesteps x n_metabolites) time series into
    memory by unnesting the list, tagging each element with its position,
    grouping by position, and then averaging. Returns the per-metabolite
    averages (in list order).
    """
    subquery = read_stacked_columns(
        history_sql, [f"{column} AS v"], order_results=False
    )
    df = conn.sql(
        f"""
        WITH unnested AS (
            SELECT unnest(v) AS val, generate_subscripts(v, 1) AS met_idx
            FROM ({subquery})
        )
        SELECT avg(val) AS avg_val
        FROM unnested
        GROUP BY met_idx
        ORDER BY met_idx
        """
    ).pl()
    return df["avg_val"].to_numpy()


def read_avg_free_from_bulk(conn, history_sql, sim_data, metabolite_ids):
    """Time-average the free count of each metabolite from the bulk table.

    Scales to thousands of metabolites: rather than pulling one DuckDB column
    per metabolite (named_idx) and averaging in Python, slices the bulk array
    down to metabolites as a SINGLE list column
    (``ndidx_to_duckdb_expr`` -> ``list_select``), then unnest + group + average
    entirely inside DuckDB. Average is over all timesteps of all seeds.
    Returns averages in ``metabolite_ids`` order.
    """
    bulk_ids = list(sim_data.internal_state.bulk_molecules.bulk_data["id"])
    bname_to_idx = {n: i for i, n in enumerate(bulk_ids)}
    idx = [bname_to_idx[m] for m in metabolite_ids]

    # Single column: bulk sliced to just metabolite IDs (handles 1-indexing).
    sublist_expr = ndidx_to_duckdb_expr("bulk", [idx])
    subquery = read_stacked_columns(history_sql, [sublist_expr], order_results=False)
    df = conn.sql(
        f"""
        WITH unnested AS (
            SELECT
                unnest("bulk") AS free_count,
                generate_subscripts("bulk", 1) AS met_idx
            FROM ({subquery})
        )
        SELECT avg(free_count) AS avg_free
        FROM unnested
        GROUP BY met_idx
        ORDER BY met_idx
        """
    ).pl()
    return df["avg_free"].to_numpy()


# Categorize by which sequestration locations (eq complex / TCS / DNA-bound TF)
# are nonzero in the simulation:
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
    """Assign each metabolite an 8-way category label."""
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

    # Load metabolite IDs from listener metadata:
    metabolite_ids = field_metadata(
        conn, config_sql, "listeners__metabolite_counts__totalMetaboliteCounts"
    )

    # Obtain total counts from the listener and free straight from the
    # standard bulk table:
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

    # Plot every metabolite that is ever present in the simulation (a zero
    # average total means the count was 0 at every timestep, i.e. it never
    # appeared in this condition):
    n_zero = int((avg_total <= 0).sum())
    n_cells = count_cells(conn, history_sql)
    print(
        f"{n_zero} of {len(metabolite_ids)} metabolites had zero counts the "
        f"entire simulation (not plotted). Averaged over {n_cells} cells."
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

    df = df.with_columns(categorize_expr())

    # Append the per-category count to each label:
    cat_counts = dict(df.group_by("sequestration_type").len().iter_rows())
    df = df.with_columns(
        pl.col("sequestration_type")
        .map_elements(lambda c: f"{c} ({cat_counts.get(c, 0)})", return_dtype=pl.Utf8)
        .alias("sequestration_label")
    )

    # Build hover tooltip
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

    # Build scatter plot (log-log so values spanning many orders of
    # magnitude are not crammed into a corner)
    min_val = float(df["avg_total"].min())
    max_val = float(df["avg_total"].max())
    log_scale_x = alt.Scale(type="log", domain=[min_val * 0.8, max_val * 1.25])
    log_scale_y = alt.Scale(type="log", domain=[min_val * 0.8, max_val * 1.25])

    # y=x reference line
    ref_line = (
        alt.Chart(pl.DataFrame({"x": [min_val * 0.8, max_val * 1.25]}))
        .mark_line(strokeDash=[4, 2], color="black", opacity=0.5)
        .encode(
            x=alt.X("x:Q", scale=log_scale_x),
            y=alt.Y("x:Q", scale=log_scale_y),
        )
    )

    # Fixed base-category -> color (so only categories present are shown):
    domain_labels = []
    range_colors = []
    for base, color in CATEGORY_COLORS:
        if base in cat_counts:
            domain_labels.append(f"{base} ({cat_counts[base]})")
            range_colors.append(color)
    color_scale = alt.Scale(domain=domain_labels, range=range_colors)

    scatter = (
        alt.Chart(df)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X(
                "avg_total:Q",
                scale=log_scale_x,
                title="Log Average Total Count",
            ),
            y=alt.Y("avg_free:Q", scale=log_scale_y, title="Log Average Free Count"),
            color=alt.Color(
                "sequestration_label:N",
                scale=color_scale,
                title="Sequestration (count)",
            ),
            size=alt.condition(
                alt.datum.sequestration_type == "No sequestration",
                alt.value(40),
                alt.value(80),
            ),
            tooltip=tooltip,
        )
    )

    chart = (
        (ref_line + scatter)
        .properties(
            title=(
                f"Average Total vs Free Metabolite Counts  "
                f"(averaged over {n_cells} cells)  |  "
                f"n={len(df)} metabolites plotted ({n_zero} had zero counts "
                f"over the entire simulation) "
            ),
            width=600,
            height=600,
        )
        .configure_view(fill="white")
        .interactive()
    )

    chart.save(os.path.join(outdir, "metabolite_total_vs_free_counts.html"))
