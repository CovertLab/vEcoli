import os
from typing import Any

from duckdb import DuckDBPyConnection
from scipy.constants import N_A
import polars as pl
from unum import Unum
from wholecell.utils import units
from ecoli.library.parquet_emitter import read_stacked_columns, field_metadata


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
    bulk_ids = field_metadata(conn=conn, config_subquery=config_sql, field="bulk")

    bulk_id_glycogen = "glycogen-monomer[c]"

    bulk_idx_glycogen = bulk_ids.index(bulk_id_glycogen)

    glycogen_sql = f"bulk[{bulk_idx_glycogen + 1}] AS glycogen_raw"

    data_columns = [
        "listeners__mass__cell_mass",
        "listeners__mass__dry_mass",
        "listeners__mass__volume",
        "listeners__mass__dna_mass",
        "listeners__mass__rna_mass",
        glycogen_sql,
    ]

    history_subquery = read_stacked_columns(
        history_sql, data_columns, order_results=False
    )

    filters = []
    if params.get("generation_lower_bound") is not None:
        filters.append(f"generation >= {params['generation_lower_bound']}")
    if params.get("time_lower_bound") is not None:
        filters.append(f"time >= {float(params['time_lower_bound'])}")
    filter_clause = ""
    if filters:
        filter_clause = "WHERE " + " AND ".join(filters)

    # (count / fg dry mass) * (1 mol / N_A count) * (1e3 mmol / mol) = mmol / g dry mass
    glycogen_scale = Unum.asNumber(units.g / units.fg) / N_A * 1e3  # type: ignore[attr-defined]
    # (fg) * (1 mg / 1e12 fg) / (1e9 cells) = mg per 1e9 cells
    mass_scale = Unum.asNumber(units.mg / units.fg) * 10**-9  # type: ignore[attr-defined]

    id_cols = [
        "experiment_id",
        "variant",
        "lineage_seed",
        "generation",
        "agent_id",
    ]

    query = f"""
        WITH history AS ({history_subquery}),
        filtered AS (
            SELECT * FROM history
            {filter_clause}
        )
        SELECT
            {", ".join(id_cols)},
            AVG(listeners__mass__cell_mass / {mass_scale}) AS mass_converted,
            AVG(listeners__mass__volume) AS cell_volume,
            AVG(listeners__mass__dna_mass / NULLIF(listeners__mass__dry_mass, 0)) AS dna_converted,
            AVG(listeners__mass__rna_mass / NULLIF(listeners__mass__dry_mass, 0)) AS rna_converted,
            AVG(glycogen_raw * {glycogen_scale} / NULLIF(listeners__mass__dry_mass, 0)) AS glycogen_converted
        FROM filtered
        GROUP BY {", ".join(id_cols)}
        ORDER BY {", ".join(id_cols)}
    """

    aggregated = conn.sql(query).pl()

    if aggregated.is_empty():
        output_final = pl.DataFrame({"Properties": [], "mean": [], "std": []})
    else:
        label_map = {
            "mass_converted": "Cell mass (mg/10^9 cells)",
            "cell_volume": "Cell volume (um^3)",
            "dna_converted": "Genetic material - DNA (g DNA/g dry weight)",
            "rna_converted": "Genetic material - RNA (g RNA/g dry weight)",
            "glycogen_converted": "Metabolism - Glycogen (mmol glycosyl units/g dry weight)",
        }

        aggregated = aggregated.rename(label_map)
        value_labels = list(label_map.values())

        aggregated = aggregated.with_columns(
            pl.format("Cell: {}_{}", pl.col("lineage_seed"), pl.col("agent_id")).alias(
                "cell_id"
            )
        )
        output_final = (
            aggregated.select(["cell_id", *value_labels])
            .unpivot(
                on=value_labels,
                index="cell_id",
                variable_name="Properties",
                value_name="value",
            )
            .pivot(
                values="value",
                index="Properties",
                on="cell_id",
                aggregate_function="first",
                sort_columns=True,
            )
        )

        cell_cols = [col for col in output_final.columns if col != "Properties"]
        if cell_cols:
            output_final = output_final.with_columns(
                [
                    pl.concat_list(cell_cols).list.mean().alias("mean"),
                    pl.concat_list(cell_cols).list.std().alias("std"),
                ]
            )
            output_final = output_final.select(
                ["Properties", "mean", "std", *cell_cols]
            )
        else:
            output_final = output_final.with_columns(
                [
                    pl.lit(None).alias("mean"),
                    pl.lit(None).alias("std"),
                ]
            ).select(["Properties", "mean", "std"])

    output_final.write_csv(
        os.path.join(outdir, "higher_order_properties.tsv"),
        separator="\t",
        include_header=True,
    )
