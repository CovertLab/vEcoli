import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import altair as alt
    import tempfile

    from ecoli.library.parquet_emitter import get_dataset_sql
    from runscripts.analysis import create_duckdb_conn

    return alt, create_duckdb_conn, get_dataset_sql, mo, tempfile


@app.cell
def _(create_duckdb_conn, tempfile):
    temp_dir = tempfile.TemporaryDirectory()
    conn = create_duckdb_conn(temp_dir.name, None)
    return (conn,)


@app.cell
def _(mo):
    out_cfg = mo.ui.dictionary({"Out dir": mo.ui.text(), "Exp ID": mo.ui.text()})
    out_cfg
    return (out_cfg,)


@app.cell
def _(get_dataset_sql, out_cfg):
    h, c, s = get_dataset_sql(out_cfg.value["Out dir"], [out_cfg.value["Exp ID"]])
    return c, h


@app.cell
def _(c, conn, mo):
    cell_ids = mo.sql(
        f"""
        SELECT
            variant,
            lineage_seed,
            generation,
            agent_id
        FROM
            ({c})
        """,
        output=False,
        engine=conn,
    )
    return (cell_ids,)


@app.cell
def _(cell_ids, mo):
    selected_variant = mo.ui.dropdown(
        options=cell_ids["variant"], label="Variant", searchable=True
    )
    selected_variant
    return (selected_variant,)


@app.cell
def _(cell_ids, mo, selected_variant):
    selected_seed = mo.ui.dropdown(
        options=sorted(
            cell_ids.filter(cell_ids["variant"] == selected_variant.value)[
                "lineage_seed"
            ]
        ),
        label="Seed",
        searchable=True,
    )
    selected_seed
    return (selected_seed,)


@app.cell
def _(cell_ids, mo, selected_seed, selected_variant):
    selected_generation = mo.ui.dropdown(
        options=sorted(
            cell_ids.filter(
                (cell_ids["variant"] == selected_variant.value)
                & (cell_ids["lineage_seed"] == selected_seed.value)
            )["generation"]
        ),
        label="Generation",
        searchable=True,
    )
    selected_generation
    return (selected_generation,)


@app.cell
def _(cell_ids, mo, selected_generation, selected_seed, selected_variant):
    selected_agent = mo.ui.dropdown(
        options=sorted(
            cell_ids.filter(
                (cell_ids["variant"] == selected_variant.value)
                & (cell_ids["lineage_seed"] == selected_seed.value)
                & (cell_ids["generation"] == selected_generation.value)
            )["agent_id"]
        ),
        label="Agent ID",
        searchable=True,
    )
    selected_agent
    return (selected_agent,)


@app.cell
def _(conn, h, mo):
    listener_names = mo.sql(
        f"""
        SELECT column_name FROM (DESCRIBE ({h}))
        """,
        output=False,
        engine=conn,
    )
    return (listener_names,)


@app.cell
def _(listener_names, mo):
    selected_listener = mo.ui.dropdown(
        options=listener_names["column_name"], label="Listener", searchable=True
    )
    selected_listener
    return (selected_listener,)


@app.cell
def _(selected_agent, selected_generation, selected_seed, selected_variant):
    filter_clause = []
    if selected_variant.value is not None:
        filter_clause.append(f"WHERE variant = {selected_variant.value}")
    if selected_seed.value is not None:
        filter_clause.append(f"lineage_seed = {selected_seed.value}")
    if selected_generation.value is not None:
        filter_clause.append(f"generation = {selected_generation.value}")
    if selected_agent.value is not None:
        filter_clause.append(f"agent_id = {selected_agent.value}")
    filter_clause = " AND ".join(filter_clause)
    return (filter_clause,)


@app.cell
def _(conn, filter_clause, h, mo, selected_listener):
    filtered_df = mo.sql(
        f"""
        SELECT {selected_listener.value}, variant, lineage_seed, generation, agent_id, time FROM ({h}) {filter_clause}
        """,
        output=False,
        engine=conn,
    )
    return (filtered_df,)


@app.cell
def _(alt, filtered_df, selected_listener):
    (
        alt.Chart(filtered_df)
        .mark_line()
        .encode(
            x=alt.X(
                "time",
                axis=alt.Axis(title="Time (s)", labelFlush=False),
            ),
            y=alt.Y(selected_listener.value),
        )
    )
    return


if __name__ == "__main__":
    app.run()
