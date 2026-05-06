import os
from typing import Any

from duckdb import DuckDBPyConnection
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
    skip_n_gens,
)


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
    Args:
        params: Dictionary containing parameters of the format::

            {
                # Number of initial generations worth of data to skip
                "skip_n_gens": int,
            }
    """
    if params.get("skip_n_gens"):
        history_sql = skip_n_gens(history_sql, params["skip_n_gens"])

    # --- Monomer counts ---
    all_monomer_ids_raw = field_metadata(conn, config_sql, "listeners__monomer_counts")
    all_monomer_ids = [m[:-3] for m in all_monomer_ids_raw]
    n_monomers = len(all_monomer_ids)

    if n_monomers:
        monomer_subquery = read_stacked_columns(
            history_sql,
            ["listeners__monomer_counts AS monomer_counts"],
            order_results=False,
        )
        # Average counts across all timesteps within each (seed, generation) cell
        long_df = conn.sql(f"""
            WITH unnested AS (
                SELECT
                    lineage_seed,
                    generation,
                    unnest(monomer_counts) AS count,
                    generate_subscripts(monomer_counts, 1) AS idx
                FROM ({monomer_subquery})
            )
            SELECT lineage_seed, generation, idx, avg(count) AS avg_count
            FROM unnested
            GROUP BY lineage_seed, generation, idx
            ORDER BY lineage_seed, generation, idx
        """).pl()

        # Map 1-based idx to monomer ID, then pivot to wide format
        monomer_map = pl.DataFrame(
            {"idx": list(range(1, n_monomers + 1)), "monomer_id": all_monomer_ids}
        )
        wide_df = (
            long_df.join(monomer_map, on="idx")
            .pivot(
                on="monomer_id",
                index=["lineage_seed", "generation"],
                values="avg_count",
                aggregate_function="first",
            )
            .rename({"lineage_seed": "seed"})
            .sort(["seed", "generation"])
        )

        out_path = os.path.join(outdir, "protein_counts.tsv")
        wide_df.write_csv(out_path, separator="\t")
        n_rows = len(wide_df)
        print(f"Wrote {n_rows} rows x {n_monomers} monomers to {out_path}")
    else:
        print("No monomer IDs found; protein_counts.tsv not written.")

    # --- Complex counts from bulk ---
    # bulk contains all molecules; strip compartment tags and write all counts.
    try:
        bulk_ids_raw = field_metadata(conn, config_sql, "bulk")
    except Exception as e:
        print(f"Could not load bulk metadata (complex counts skipped): {e}")
        return

    all_bulk_ids = [
        bid[:-3] if len(bid) > 3 and bid[-1] == "]" else bid for bid in bulk_ids_raw
    ]
    n_bulk = len(all_bulk_ids)

    if n_bulk:
        bulk_subquery = read_stacked_columns(
            history_sql,
            ["bulk AS selected_counts"],
            order_results=False,
        )
        bulk_long_df = conn.sql(f"""
            WITH unnested AS (
                SELECT
                    lineage_seed,
                    generation,
                    unnest(selected_counts) AS count,
                    generate_subscripts(selected_counts, 1) AS idx
                FROM ({bulk_subquery})
            )
            SELECT lineage_seed, generation, idx, avg(count) AS avg_count
            FROM unnested
            GROUP BY lineage_seed, generation, idx
            ORDER BY lineage_seed, generation, idx
        """).pl()

        bulk_map = pl.DataFrame(
            {"idx": list(range(1, n_bulk + 1)), "molecule_id": all_bulk_ids}
        )
        bulk_wide_df = (
            bulk_long_df.join(bulk_map, on="idx")
            .pivot(
                on="molecule_id",
                index=["lineage_seed", "generation"],
                values="avg_count",
                aggregate_function="first",
            )
            .rename({"lineage_seed": "seed"})
            .sort(["seed", "generation"])
        )

        out_path = os.path.join(outdir, "complex_counts.tsv")
        bulk_wide_df.write_csv(out_path, separator="\t")
        n_rows = len(bulk_wide_df)
        print(f"Wrote {n_rows} rows x {n_bulk} bulk molecules to {out_path}")
    else:
        print("No bulk molecule IDs found; complex_counts.tsv not written.")
