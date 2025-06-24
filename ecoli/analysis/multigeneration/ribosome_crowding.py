import os
import pickle

from typing import Any

import altair as alt
import polars as pl
import numpy as np
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import (
    open_arbitrary_sim_data,
    field_metadata,
    named_idx,
    read_stacked_columns,
)

# Maximum number of overcrowded proteins to plot
MAX_NUMBER_OF_MONOMERS_TO_PLOT = 300


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
    Compare target vs actual translation probabilities for mRNAs
    whose translation probabilities were limited by ribosome crowding.
    """

    # 1. Load sim_data from arbitrary source (as in new_gene example)
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # 2. From sim_data, get monomer IDs and mappings to mRNA/gene
    mRNA_sim_array = sim_data.process.transcription.cistron_data.struct_array
    monomer_sim_array = sim_data.process.translation.monomer_data.struct_array
    monomer_ids: list[str] = monomer_sim_array["id"].tolist()

    # Build mapping: monomer_id -> mRNA_id -> gene_id
    monomer_to_mRNA: dict[str, str] = dict(
        zip(monomer_sim_array["id"], monomer_sim_array["cistron_id"])
    )
    mRNA_to_gene: dict[str, str] = dict(
        zip(mRNA_sim_array["id"], mRNA_sim_array["gene_id"])
    )

    # 3. Determine listener names / field names in vEcoli for target & actual probabilities.
    #    You need to replace the placeholders below with the real listener/field names
    #    in your DuckDB schema / parquet emitter config.
    #    Common pattern: listener name might be "ribosome_data" and fields like
    #    "target_prob_translation_per_transcript" and "actual_prob_translation_per_transcript".
    #
    #    For example, if in config_sql you have a listener named "ribosome_data" and
    #    parquet columns named "target_prob_translation_per_transcript_<monomerId>",
    #    then you may use field_metadata to get the list of column names and named_idx to select indices.
    #
    # 3.a. Fetch all column names for target probabilities
    # TODO: replace "listeners__ribosome_data__target_prob_translation_per_transcript"
    #       with the actual listener name used in your vEcoli config.
    try:
        target_columns: list[str] = field_metadata(
            conn,
            config_sql,
            "listeners__ribosome_data__target_prob_translation_per_transcript",
        )
        actual_columns: list[str] = field_metadata(
            conn,
            config_sql,
            "listeners__ribosome_data__actual_prob_translation_per_transcript",
        )
    except Exception:
        # If the above naming is incorrect, adjust to your listener naming convention.
        raise RuntimeError(
            "Failed to fetch field metadata for ribosome data. "
            "Please replace the listener names in field_metadata(...) with your actual names."
        )

    # 3.b. Build index dicts: column name -> index in the wide array
    #     We assume that field_metadata returns a list of column names in the same order
    #     as the underlying array dimension for translation probabilities.
    target_idx_dict = {col: i for i, col in enumerate(target_columns)}
    actual_idx_dict = {col: i for i, col in enumerate(actual_columns)}

    # 3.c. Determine indexes for all monomer_ids in these columns
    #     Here we assume that column names in target_columns/actual_columns directly match monomer_ids.
    #     If not, adjust the mapping logic accordingly.
    target_indexes: list[int] = []
    actual_indexes: list[int] = []
    missing_target = []
    missing_actual = []
    for mon in monomer_ids:
        if mon in target_idx_dict:
            target_indexes.append(target_idx_dict[mon])
        else:
            missing_target.append(mon)
        if mon in actual_idx_dict:
            actual_indexes.append(actual_idx_dict[mon])
        else:
            missing_actual.append(mon)
    if missing_target or missing_actual:
        # Warn user that some monomers are not present in the listener fields
        print(
            f"Warning: some monomer IDs not found in target/actual fields. "
            f"Missing in target: {missing_target[:5]}{'...' if len(missing_target) > 5 else ''}; "
            f"Missing in actual: {missing_actual[:5]}{'...' if len(missing_actual) > 5 else ''}."
        )
        # Continue with intersection of available monomers
    # Use only those present in both
    # Find intersection in order of monomer_ids:
    valid_monomer_ids = [
        mon for mon in monomer_ids if mon in target_idx_dict and mon in actual_idx_dict
    ]
    valid_target_indexes = [target_idx_dict[mon] for mon in valid_monomer_ids]
    valid_actual_indexes = [actual_idx_dict[mon] for mon in valid_monomer_ids]

    if not valid_monomer_ids:
        print(
            "No overlapping monomer IDs found in ribosome_data listeners; aborting plot."
        )
        return

    # 4. Read stacked columns: time + target + actual arrays.
    #    First read target data:
    target_named = named_idx(
        "listenersribosome_data_target_prob_translation_per_transcript",
        valid_monomer_ids,
        valid_target_indexes,
    )
    # Then read actual data:
    actual_named = named_idx(
        "listenersribosome_data_actual_prob_translation_per_transcript",
        valid_monomer_ids,
        valid_actual_indexes,
    )
    # Note: 上面 named_idx 的第一个参数需要替换为你项目中实际的 listener 名称前缀，例如
    # "listeners__ribosome_data__target_prob_translation_per_transcript" 或类似，确保与 field_metadata(...) 中使用的 listener 匹配。

    # Read time + these columns. read_stacked_columns 返回 dict-like，包含 "time" 字段和各 monomer 列。
    # 如果 time 字段命名不是 "time"，请调整。
    target_data = read_stacked_columns(history_sql, [target_named], conn=conn)
    actual_data = read_stacked_columns(history_sql, [actual_named], conn=conn)

    # Convert to Polars DataFrame
    df_target = pl.DataFrame(target_data)
    df_actual = pl.DataFrame(actual_data)

    # Assume both have a "time" column; drop duplicate time in actual
    if "time" in df_actual.columns:
        df_actual = df_actual.drop("time")

    # 5. Rename columns to distinguish target vs actual
    #    e.g., columns are monomer IDs; 重命名为 target_<monomer> / actual_<monomer>
    rename_target = {mon: f"target_{mon}" for mon in valid_monomer_ids}
    rename_actual = {mon: f"actual_{mon}" for mon in valid_monomer_ids}
    df_target = df_target.rename(rename_target)
    df_actual = df_actual.rename(rename_actual)

    # Merge horizontally on row order (time)
    df = pl.concat([df_target, df_actual], how="horizontal")
    # Create Time (min) column
    if "time" in df.columns:
        df = df.with_columns((pl.col("time") / 60).alias("Time (min)"))
    else:
        raise RuntimeError("No 'time' column found in ribosome data readout.")

    # Compute overcrowded monomer indices: where max(target - actual) > 0
    #    We'll convert to NumPy for efficient max along time axis.
    #    Build numpy arrays in matching order
    # n = len(valid_monomer_ids)
    # Stack arrays: shape (T, n)
    target_matrix = np.vstack(
        [df[f"target_{mon}"].to_numpy() for mon in valid_monomer_ids]
    ).T
    actual_matrix = np.vstack(
        [df[f"actual_{mon}"].to_numpy() for mon in valid_monomer_ids]
    ).T
    diff = target_matrix - actual_matrix
    # max over time for each monomer
    max_diff = diff.max(axis=0)
    overcrowded_mask = max_diff > 0
    overcrowded_indices = np.where(overcrowded_mask)[0].tolist()
    n_overcrowded = len(overcrowded_indices)

    if n_overcrowded == 0:
        print("No overcrowded mRNAs detected in this simulation; nothing to plot.")
        return

    # Limit number to plot
    n_to_plot = min(n_overcrowded, MAX_NUMBER_OF_MONOMERS_TO_PLOT)
    if n_overcrowded > MAX_NUMBER_OF_MONOMERS_TO_PLOT:
        print(
            f"Total overcrowded proteins: {n_overcrowded}. "
            f"Plotting first {MAX_NUMBER_OF_MONOMERS_TO_PLOT} only."
        )

    # For each overcrowded monomer, get gene ID and build a small Altair chart.
    charts = []
    for idx_in_list in overcrowded_indices[:n_to_plot]:
        monomer = valid_monomer_ids[idx_in_list]
        gene = mRNA_to_gene.get(monomer_to_mRNA.get(monomer, ""), "unknown")
        # Build a pandas DataFrame for plotting: columns Time (min), target, actual
        # Use Polars to pandas conversion for this single monomer:
        pd_df = (
            df.select(["Time (min)", f"target_{monomer}", f"actual_{monomer}"])
            .rename(
                {
                    "Time (min)": "Time (min)",
                    f"target_{monomer}": "target",
                    f"actual_{monomer}": "actual",
                }
            )
            .to_pandas()
        )
        # Melt to long form
        pd_long = pd_df.melt(
            id_vars=["Time (min)"],
            value_vars=["target", "actual"],
            var_name="Type",
            value_name="Probability",
        )
        # Create line chart
        chart = (
            alt.Chart(pd_long)
            .mark_line()
            .encode(
                x=alt.X("Time (min)", title="Time (min)"),
                y=alt.Y("Probability", title="Translation Probability"),
                color="Type",
            )
            .properties(title=f"{gene} (monomer {monomer})")
            .interactive()
        )
        charts.append(chart)

    # Vertically concatenate all charts
    combined = (
        alt.vconcat(*charts)
        .configure_axis(labelFontSize=10, titleFontSize=12)
        .configure_title(fontSize=14)
    )

    # Save to HTML
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "ribosome_crowding.html")
    combined.save(outpath)
    print(f"Saved ribosome crowding plot to {outpath}")
