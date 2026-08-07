import os
from typing import Any

from duckdb import DuckDBPyConnection
import polars as pl

from ecoli.library.parquet_emitter import (
    field_metadata,
    read_stacked_columns,
    skip_n_gens,
)

_IDS_DIR = os.path.join(os.path.dirname(__file__), "cofactor_ids")
COMPLEX_IDS_PATH = os.path.join(_IDS_DIR, "complex_ids.txt")
PROTEIN_IDS_PATH = os.path.join(_IDS_DIR, "protein_ids.txt")


def _read_id_list(path: str) -> list[str]:
    """Read a one-ID-per-line truth file into a list of IDs.

    The first line is a header (``protein_id`` / ``Complex_id``), not an ID.
    """
    return pl.read_csv(path, has_header=True).to_series(0).to_list()


def _filter_to_truth_ids(wide_df: pl.DataFrame, truth_ids: list[str]) -> pl.DataFrame:
    """Keep only seed/generation + truth ID columns, padding any ID missing
    from ``wide_df`` with a column of zeros. Column order follows ``truth_ids``.
    """
    present = [i for i in truth_ids if i in wide_df.columns]
    missing = [i for i in truth_ids if i not in wide_df.columns]
    filtered = wide_df.select(["seed", "generation", *present])
    if missing:
        filtered = filtered.with_columns([pl.lit(0).alias(i) for i in missing])
    # Reorder
    return filtered.select(["seed", "generation", *truth_ids])


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

        for truth_path, out_name, label in (
            (PROTEIN_IDS_PATH, "proteins_filtered.tsv", "proteins"),
            (COMPLEX_IDS_PATH, "complexes_filtered.tsv", "complexes"),
        ):
            truth_ids = _read_id_list(truth_path)
            filtered = _filter_to_truth_ids(bulk_wide_df, truth_ids)
            out_path = os.path.join(outdir, out_name)
            filtered.write_csv(out_path, separator="\t")
            n_missing = sum(1 for i in truth_ids if i not in bulk_wide_df.columns)
            print(
                f"Wrote {len(filtered)} rows x {len(truth_ids)} cofactor "
                f"{label} ({n_missing} padded with 0) to {out_path}"
            )
    else:
        print(
            "No bulk molecule IDs found; "
            "proteins_filtered.tsv/complexes_filtered.tsv not written."
        )
