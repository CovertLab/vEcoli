import os
from typing import Any

from duckdb import DuckDBPyConnection
import polars as pl
import plotly.express as px

from ecoli.library.parquet_emitter import num_cells, read_stacked_columns

COLORS_256 = [  # From colorbrewer2.org, qualitative 8-class set 1
    [228, 26, 28],
    [55, 126, 184],
    [77, 175, 74],
    [152, 78, 163],
    [255, 127, 0],
    [255, 255, 51],
    [166, 86, 40],
    [247, 129, 191],
]

COLORS = ["#%02x%02x%02x" % (color[0], color[1], color[2]) for color in COLORS_256]

OBJECTIVE_WEIGHTS = {
    "secretion": 0.01,
    "efficiency": 0.0001,  # decrease efficiency
    "kinetic": 0.005,  # 0.00001
    # "diversity": 0.0001, # 0.001 Heena's addition to minimize number of reactions with no flow
    "homeostatic": 1,
}


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
    assert num_cells(conn, config_sql) == 1, (
        "objective function plot requires single-cell data."
    )

    objective_columns = dict()
    for term, weight in OBJECTIVE_WEIGHTS.items():
        listener = f"listeners__fba_results__{term}_term"
        objective_columns[term] = listener

    objective_data = pl.DataFrame(
        read_stacked_columns(history_sql, list(objective_columns.values()), conn=conn)
    )

    new_columns = {
        "Time (min)": (objective_data["time"] - objective_data["time"].min()) / 60,
        **{f"{k} weighted": objective_data[v] for k, v in objective_columns.items()},
        **{
            f"{k} unweighted": objective_data[v] / OBJECTIVE_WEIGHTS[k]
            for k, v in objective_columns.items()
        },
    }

    df = pl.DataFrame(new_columns)

    # # Altair requires long form data (also no periods in column names)
    # melted_df = df.melt(
    #     id_vars="Time (min)",
    #     variable_name="Term",
    #     value_name="Objective Term",
    # )
    #
    # chart = (
    #     alt.Chart(melted_df)
    #     .mark_line()
    #     .encode(
    #         x=alt.X("Time (min):Q", title="Time (min)"),
    #         y=alt.Y("Objective Term:Q"),
    #         color=alt.Color("Term:N", scale=alt.Scale(range=COLORS)),
    #     )
    #     .properties(
    #         title="Weighted Objective Function Terms"
    #     )
    # )
    # chart.save(os.path.join(outdir, "weighted_objective_terms.html"))

    melted_df = df.melt(
        id_vars="Time (min)",
        variable_name="Term",
        value_name="Objective Term",
    )

    pdf = melted_df.to_pandas()  # Plotly expects pandas-like

    fig = px.line(
        pdf,
        x="Time (min)",
        y="Objective Term",
        color="Term",
        title="Weighted Objective Function Terms",
        color_discrete_sequence=COLORS,  # your hex palette
    )

    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="Objective Term",
        legend_title="Term",
        # template="plotly_white",
        width=800,
        height=500,
        margin=dict(l=80, r=20, t=60, b=40),
    )

    fig.write_html(
        os.path.join(outdir, "weighted_objective_terms.html"), include_plotlyjs="cdn"
    )
