import altair as alt

# noinspection PyUnresolvedReferences
from duckdb import DuckDBPyConnection
import pyarrow as pa
from typing import Any

from ecoli.library.parquet_emitter import read_stacked_columns


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
    """
    Line plots of mRNA counts and mass. Only works for lineage
    simulations with ``single_daughters`` set to True.
    """
    mrna_data: pa.Table = read_stacked_columns(
        history_sql,
        [
            "list_sum(listeners__rna_counts__mRNA_counts)::BIGINT AS 'mRNA counts'",
            "listeners__mass__mRNA_mass AS 'mRNA mass (fg)'",
            "time / 3600 AS 'Time (hr)'",
        ],
        order_results=False,
        conn=conn,
    )

    # Subsample the data to include only every fifth row
    mrna_data = mrna_data.take(pa.array(range(0, mrna_data.num_rows, 20)))

    selection = alt.selection_point(fields=["lineage_seed"], bind="legend")
    mass_chart = (
        alt.Chart(mrna_data)
        .mark_line()
        .encode(
            x="Time (hr)",
            y="mRNA mass (fg)",
            color=alt.Color("lineage_seed:N"),
            tooltip=["Time (hr)", "mRNA mass (fg)"],
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
        )
        .add_params(selection)
    )
    count_chart = (
        alt.Chart(mrna_data)
        .mark_line()
        .encode(
            x="Time (hr)",
            y="mRNA counts",
            color=alt.Color("lineage_seed:N"),
            tooltip=["Time (hr)", "mRNA counts"],
            opacity=alt.when(selection).then(alt.value(1)).otherwise(alt.value(0.2)),
        )
    )
    chart = mass_chart & count_chart
    chart.save(f"{outdir}/mrna_counts.html")
