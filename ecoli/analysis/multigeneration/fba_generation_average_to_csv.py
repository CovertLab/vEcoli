"""
Export FBA reaction net fluxes to CSV format with generation-wise averages.

For each specified BioCyc_ID, creates a CSV where:
- Each row represents a BioCyc reaction ID
- Each column represents the average flux value for generation i
- Final column contains the overall average across all time steps

Usage parameters:
    "fba_flux_csv": {
        # Required: specify BioCyc reaction IDs to export
        "BioCyc_ID": ["Name1", "Name2", ...],
        # Optional: specify generations to include
        # If not specified, all generations will be used
        "generation": [1, 2, ...],
        # Optional: output filename (default: "fba_generation_average_summary.csv")
        "output_filename": "custom_filename.csv"
    }
"""

import os
import pandas as pd
import polars as pl
from typing import Any
from duckdb import DuckDBPyConnection

from ecoli.library.parquet_emitter import field_metadata
from ecoli.analysis.utils import (
    create_base_to_extended_mapping,
    build_flux_calculation_sql,
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
    """Export FBA reaction net fluxes to CSV with generation-wise averages."""

    # Get parameters
    biocyc_ids = params.get("BioCyc_ID", [])
    if not biocyc_ids:
        print(
            "[ERROR] No BioCyc_ID found in params. Please specify reaction IDs to export."
        )
        return None

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

    output_filename = params.get(
        "output_filename", "fba_generation_average_summary.csv"
    )
    target_generations = params.get("generation", None)

    print(f"[INFO] Exporting net fluxes for {len(biocyc_ids)} reactions: {biocyc_ids}")
    if target_generations:
        print(f"[INFO] Filtering for generations: {target_generations}")

    # Create base to extended reaction mapping
    base_to_extended_mapping = create_base_to_extended_mapping(sim_data_dict)
    if not base_to_extended_mapping:
        print("[ERROR] Could not create base to extended reaction mapping")
        return None

    # Load reaction IDs from config
    try:
        all_reaction_ids = field_metadata(
            conn, config_sql, "listeners__fba_results__reaction_fluxes"
        )
        print(f"[INFO] Total reactions in sim_data: {len(all_reaction_ids)}")
    except Exception as e:
        print(f"[ERROR] Error loading reaction IDs: {e}")
        return None

    # Build SQL query for efficient flux calculation
    flux_calculation_sql, valid_biocyc_ids = build_flux_calculation_sql(
        biocyc_ids, base_to_extended_mapping, all_reaction_ids, history_sql
    )

    if not flux_calculation_sql or not valid_biocyc_ids:
        print("[ERROR] Could not build flux calculation SQL")
        return None

    print(f"[INFO] Processing {len(valid_biocyc_ids)} valid BioCyc IDs")

    # Execute the optimized SQL query
    try:
        df = conn.sql(flux_calculation_sql).pl()
        print(f"[INFO] Loaded data with {df.height} time steps")
    except Exception as e:
        print(f"[ERROR] Error executing flux calculation SQL: {e}")
        return None

    if df.is_empty():
        print("[ERROR] No data found")
        return None

    # Filter by specified generations if provided
    if target_generations is not None:
        print(f"[INFO] Filtering for generations: {target_generations}")
        df = df.filter(pl.col("generation").is_in(target_generations))

    # Get unique generations and sort them
    unique_generations = sorted(df["generation"].unique().to_list())
    print(f"[INFO] Found {len(unique_generations)} generations: {unique_generations}")

    # Calculate averages for each BioCyc ID
    csv_data = calculate_generation_averages(df, valid_biocyc_ids, unique_generations)

    if csv_data is None or csv_data.empty:
        print("[ERROR] No valid data to export")
        return None

    # Save to CSV
    output_path = os.path.join(outdir, output_filename)
    csv_data.to_csv(output_path, index=False)
    print(f"[INFO] Successfully exported flux data to: {output_path}")
    print(
        f"[INFO] CSV contains {len(csv_data)} rows and {len(csv_data.columns)} columns"
    )

    return csv_data


def calculate_generation_averages(df, valid_biocyc_ids, unique_generations):
    """
    Calculate average flux values for each generation and overall average.

    Returns a pandas DataFrame with:
    - BioCyc_ID column
    - One column per generation (Gen_1_Avg, Gen_2_Avg, etc.)
    - Overall_Average column
    """

    results = []

    for biocyc_id in valid_biocyc_ids:
        net_flux_col = f"{biocyc_id}_net_flux"

        if net_flux_col not in df.columns:
            print(f"[WARNING] Column {net_flux_col} not found in dataframe")
            continue

        # Initialize row data
        row_data = {"BioCyc_ID": biocyc_id}

        # Calculate average for each generation
        generation_averages = []
        for gen in unique_generations:
            gen_data = df.filter(pl.col("generation") == gen)
            if gen_data.height > 0:
                gen_avg = gen_data[net_flux_col].mean()
                if gen_avg is not None:
                    row_data[f"Gen_{gen}_Avg"] = gen_avg
                    generation_averages.append(gen_avg)
                else:
                    row_data[f"Gen_{gen}_Avg"] = 0.0
            else:
                row_data[f"Gen_{gen}_Avg"] = 0.0

        # Calculate overall average across all time steps
        overall_avg = df[net_flux_col].mean()
        row_data["Overall_Average"] = overall_avg if overall_avg is not None else 0.0

        results.append(row_data)

    if not results:
        print("[ERROR] No valid results calculated")
        return None

    # Convert to pandas DataFrame for easy CSV export
    csv_data = pd.DataFrame(results)

    # Reorder columns: BioCyc_ID first, then generation columns in order, then Overall_Average
    column_order = ["BioCyc_ID"]
    for gen in unique_generations:
        column_order.append(f"Gen_{gen}_Avg")
    column_order.append("Overall_Average")

    csv_data = csv_data[column_order]

    return csv_data


def validate_parameters(params):
    """Validate input parameters."""

    if not isinstance(params, dict):
        return False, "Parameters must be a dictionary"

    biocyc_ids = params.get("BioCyc_ID", [])
    if not biocyc_ids:
        return False, "BioCyc_ID parameter is required"

    if isinstance(biocyc_ids, str):
        biocyc_ids = [biocyc_ids]

    if not isinstance(biocyc_ids, list):
        return False, "BioCyc_ID must be a string or list of strings"

    target_generations = params.get("generation", None)
    if target_generations is not None:
        if not isinstance(target_generations, list):
            return False, "generation parameter must be a list of integers"
        if not all(isinstance(x, int) for x in target_generations):
            return False, "All generation values must be integers"

    output_filename = params.get("output_filename", "fba_flux_summary.csv")
    if not isinstance(output_filename, str):
        return False, "output_filename must be a string"

    if not output_filename.endswith(".csv"):
        return False, "output_filename must end with .csv"

    return True, "Parameters are valid"
