import os
import pickle
from typing import Any

from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl
from scipy.stats import pearsonr
import altair as alt

from ecoli.library.parquet_emitter import (
    open_arbitrary_sim_data,
    ndlist_to_ndarray,
    read_stacked_columns,
)
from wholecell.utils.protein_counts import get_simulated_validation_counts


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
    with open_arbitrary_sim_data(sim_data_paths) as f:
        sim_data = pickle.load(f)
    with open(validation_data_paths[0], "rb") as f:
        validation_data = pickle.load(f)

    subquery = read_stacked_columns(
        history_sql, ["listeners__monomer_counts"], order_results=False
    )
    monomer_counts = conn.sql(f"""
        WITH unnested_counts AS (
            SELECT unnest(listeners__monomer_counts) AS counts,
                generate_subscripts(listeners__monomer_counts, 1) AS idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({subquery})
        ),
        avg_counts AS (
            SELECT avg(counts) AS avgCounts,
                experiment_id, variant, lineage_seed,
                generation, agent_id, idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, idx
        )
        SELECT list(avgCounts ORDER BY idx) AS avgCounts
        FROM avg_counts
        GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        """).pl()
    monomer_counts = ndlist_to_ndarray(monomer_counts["avgCounts"])

    sim_monomer_ids = sim_data.process.translation.monomer_data["id"]
    wisniewski_ids = validation_data.protein.wisniewski2014Data["monomerId"]
    schmidt_ids = validation_data.protein.schmidt2015Data["monomerId"]
    wisniewski_counts = validation_data.protein.wisniewski2014Data["avgCounts"]
    schmidt_counts = validation_data.protein.schmidt2015Data["glucoseCounts"]
    sim_wisniewski_counts, val_wisniewski_counts = get_simulated_validation_counts(
        wisniewski_counts, monomer_counts, wisniewski_ids, sim_monomer_ids
    )
    sim_schmidt_counts, val_schmidt_counts = get_simulated_validation_counts(
        schmidt_counts, monomer_counts, schmidt_ids, sim_monomer_ids
    )

    schmidt_chart = (
        alt.Chart(
            pl.DataFrame(
                {
                    "schmidt": np.log10(val_schmidt_counts + 1),
                    "sim": np.log10(sim_schmidt_counts + 1),
                }
            )
        )
        .mark_point()
        .encode(
            x=alt.X("schmidt", title="log10(Schmidt 2015 Counts + 1)"),
            y=alt.Y("sim", title="log10(Simulation Average Counts + 1)"),
        )
        .properties(
            title="Pearson r: %0.2f"
            % pearsonr(
                np.log10(sim_schmidt_counts + 1), np.log10(val_schmidt_counts + 1)
            )[0]
        )
    )
    wisniewski_chart = (
        alt.Chart(
            pl.DataFrame(
                {
                    "wisniewski": np.log10(val_wisniewski_counts + 1),
                    "sim": np.log10(sim_wisniewski_counts + 1),
                }
            )
        )
        .mark_point()
        .encode(
            x=alt.X("wisniewski", title="log10(Wisniewski 2014 Counts + 1)"),
            y=alt.Y("sim", title="log10(Simulation Average Counts + 1)"),
        )
        .properties(
            title="Pearson r: %0.2f"
            % pearsonr(
                np.log10(sim_wisniewski_counts + 1), np.log10(val_wisniewski_counts + 1)
            )[0]
        )
    )
    max_val = max(
        np.log10(val_schmidt_counts + 1).max(),
        np.log10(val_wisniewski_counts + 1).max(),
        np.log10(sim_schmidt_counts + 1).max(),
        np.log10(sim_wisniewski_counts + 1).max(),
    )
    parity = (
        alt.Chart(pl.DataFrame({"x": np.arange(max_val)}))
        .mark_line()
        .encode(x="x", y="x", color=alt.value("red"), strokeDash=alt.value([5, 5]))
    )
    chart = (schmidt_chart + parity) | (wisniewski_chart + parity)
    chart.save(os.path.join(outdir, "protein_counts_validation.html"))
