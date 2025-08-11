import pickle
from typing import Any

from duckdb import DuckDBPyConnection
import numpy as np
import polars as pl

from ecoli.library.parquet_emitter import (
    open_arbitrary_sim_data,
    read_stacked_columns,
    field_metadata,
    ndidx_to_duckdb_expr,
)
from wholecell.utils import units


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

    # Map monomer IDs to cistron indices
    monomer_data = sim_data.process.translation.monomer_data.struct_array
    monomer_to_cistron_id = dict(zip(monomer_data["id"], monomer_data["cistron_id"]))
    mrna_cistron_ids = field_metadata(
        conn, config_sql, "listeners__rna_counts__mRNA_cistron_counts"
    )
    cistron_idx_dict = {rna: i for i, rna in enumerate(mrna_cistron_ids)}
    monomer_ids = field_metadata(conn, config_sql, "listeners__monomer_counts")
    cistron_idx_for_monomers = [
        cistron_idx_dict[monomer_to_cistron_id[monomer_id]]
        for monomer_id in monomer_ids
    ]

    monomer_counts_sql = read_stacked_columns(
        history_sql,
        [
            "listeners__ribosome_data__ribosome_init_event_per_monomer AS init",
            "listeners__rna_counts__mRNA_cistron_counts AS mrna_counts",
        ],
        order_results=False,
    )

    monomer_mrna_counts = ndidx_to_duckdb_expr(
        "mrna_counts", [cistron_idx_for_monomers]
    )

    # Get aggregate ribosomes initiations per mRNA
    inits_per_mrna = conn.sql(f"""
        WITH extract_counts AS (
            SELECT init, {monomer_mrna_counts},
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({monomer_counts_sql})
        ),
        unnested AS (
            SELECT unnest(init) AS init,
                unnest(mrna_counts) AS mrna_counts,
                generate_subscripts(init, 1) AS idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM extract_counts
        ),
        ratio AS (
            SELECT 
                CASE
                    WHEN mrna_counts = 0 THEN 0
                    ELSE init / mrna_counts
                END AS inits_per_mrna, idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM unnested
        )
        SELECT idx, max(inits_per_mrna) AS max_inits,
            avg(inits_per_mrna) AS avg_inits,
            
        FROM ratio
        GROUP BY idx, 
        ORDER BY avg_inits DESC
    """).pl()

    ribosome_footprint_size = (
        sim_data.process.translation.active_ribosome_footprint_size.asNumber(units.nt)
    )
    nutrients = sim_data.conditions[sim_data.condition]["nutrients"]
    ribosome_elongation_rate = (
        sim_data.process.translation.ribosomeElongationRateDict[nutrients].asNumber(
            units.aa / units.s
        )
        * 3
    )

    ribosome_spacing = pl.DataFrame(
        [
            pl.Series(monomer_ids)[inits_per_mrna["idx"] - 1].alias("Monomer ID"),
            (ribosome_elongation_rate / inits_per_mrna["avg_inits"]).alias(
                "Average Ribosome Spacing (nt)"
            ),
            (ribosome_elongation_rate / inits_per_mrna["max_inits"]).alias(
                "Minimum Ribosome Spacing (nt)"
            ),
            pl.Series(np.full(len(inits_per_mrna), ribosome_footprint_size)).alias(
                "Ribosome Footprint Size (nt)"
            ),
        ]
    )

    ribosome_spacing = ribosome_spacing.select(
        ["Monomer ID"]
        + [
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
            for col in ribosome_spacing.columns
            if col != "Monomer ID"
        ]
    )

    ribosome_spacing.write_csv(f"{outdir}/ribosome_spacing.csv")
