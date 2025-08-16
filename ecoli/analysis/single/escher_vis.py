"""
Utilize Escher API for visualizing fluxes in E. coli models.
This single-analysis method provides functionality to calculate generational average fluxes,
and then visualize them using Escher with BiGG ID mapping.

You can specify the Escher map and the mapping CSV file in params:
    "escher_vis": {
        # Required: path to the CSV file with BiGG and BioCyc IDs
        "csv_file_path": "path/to/bigg_biocyc_mapping.csv",
        # Optional: specify Escher map name to visualize
        # If not specified, defaults to 'e_coli_core.Core metabolism'
        "map_name": "e_coli_core.Core metabolism",
        }
"""

import pandas as pd
import numpy as np
from duckdb import DuckDBPyConnection
from escher import Builder
import os
from typing import Dict, Any

from ecoli.library.parquet_emitter import field_metadata
from ecoli.analysis.utils import (
    create_base_to_extended_mapping,
    build_flux_calculation_sql,
)


class EscherFluxVisualizer:
    """
    A class to calculate generational average fluxes from BioCyc IDs
    and visualize them using Escher with BiGG ID mapping.
    """

    def __init__(self, csv_file_path: str):
        """
        Initialize the visualizer with the CSV mapping file.

        Args:
            csv_file_path (str): Path to CSV file with columns:
            Original_Reaction_ID, BiGG_ID, BioCyc_ID
        """
        self.csv_file_path = csv_file_path
        self.mapping_df = None
        self.average_fluxes = {}

        # Load the CSV mapping file
        self._load_mapping_file()

    def _load_mapping_file(self):
        """Load and validate the CSV mapping file."""
        try:
            self.mapping_df = pd.read_csv(self.csv_file_path)
            print(f"[INFO] Loaded mapping file with {len(self.mapping_df)} reactions")

            # Validate required columns
            required_cols = ["Original_Reaction_ID", "BiGG_ID", "BioCyc_ID"]
            missing_cols = [
                col for col in required_cols if col not in self.mapping_df.columns
            ]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Filter out rows with empty BioCyc_ID
            initial_count = len(self.mapping_df)
            self.mapping_df = self.mapping_df.dropna(subset=["BioCyc_ID"])
            self.mapping_df = self.mapping_df[
                self.mapping_df["BioCyc_ID"].str.strip() != ""
            ]
            filtered_count = len(self.mapping_df)

            print(
                f"[INFO] Filtered to {filtered_count} reactions with valid BioCyc_ID "
                f"(removed {initial_count - filtered_count} empty entries)"
            )

        except Exception as e:
            print(f"[ERROR] Failed to load mapping file: {e}")
            raise

    def calculate_generational_average_flux(
        self,
        conn: DuckDBPyConnection,
        history_sql: str,
        config_sql: str,
        sim_data_dict: dict[str, dict[int, str]],
    ) -> Dict[str, float]:
        """
        Calculate generational average flux for all BioCyc IDs using optimized SQL approach.

        Args:
            conn: DuckDB connection
            history_sql: SQL for historical data
            config_sql: SQL for configuration data
            sim_data_dict: Simulation data dictionary for creating base-to-extended mapping

        Returns:
            Dict[str, float]: Mapping of BioCyc_ID to average flux
        """
        print("[INFO] Calculating generational average flux...")

        # Get unique BioCyc IDs from the mapping file
        biocyc_ids = self.mapping_df["BioCyc_ID"].unique().tolist()
        print(f"[INFO] Found {len(biocyc_ids)} unique BioCyc IDs to process")

        # Create base to extended reaction mapping
        base_to_extended_mapping = create_base_to_extended_mapping(sim_data_dict)
        if not base_to_extended_mapping:
            print("[ERROR] Could not create base to extended reaction mapping")
            return {}

        # Load reaction IDs from config
        try:
            all_reaction_ids = field_metadata(
                conn, config_sql, "listeners__fba_results__reaction_fluxes"
            )
            print(f"[INFO] Total reactions in sim_data: {len(all_reaction_ids)}")
        except Exception as e:
            print(f"[ERROR] Error loading reaction IDs: {e}")
            return {}

        # Build SQL query for efficient flux calculation
        flux_calculation_sql, valid_biocyc_ids = build_flux_calculation_sql(
            biocyc_ids, base_to_extended_mapping, all_reaction_ids, history_sql
        )

        if not flux_calculation_sql or not valid_biocyc_ids:
            print("[ERROR] Could not build flux calculation SQL")
            return {}

        print(f"[INFO] Processing {len(valid_biocyc_ids)} valid BioCyc IDs")

        # Execute the optimized SQL query
        try:
            df = conn.sql(flux_calculation_sql).pl()
            print(f"[INFO] Loaded flux data with {df.height} time steps")
        except Exception as e:
            print(f"[ERROR] Error executing flux calculation SQL: {e}")
            return {}

        if df.height == 0:
            print("[ERROR] No data found")
            return {}

        # Calculate average flux for each BioCyc ID
        average_fluxes = {}

        for biocyc_id in valid_biocyc_ids:
            net_flux_col = f"{biocyc_id}_net_flux"
            if net_flux_col in df.columns:
                # Calculate average net flux using Polars
                avg_net_flux = df[net_flux_col].mean()
                average_fluxes[biocyc_id] = avg_net_flux

                print(
                    f"[INFO] Average net flux for {biocyc_id}: {avg_net_flux:.6f} mmol/gDW/hr"
                )
            else:
                print(f"[WARNING] Column {net_flux_col} not found in results")

        self.average_fluxes = average_fluxes
        return average_fluxes

    def create_escher_flux_map(
        self,
        output_path: str = "flux_visualization.html",
        map_name: str = "e_coli_core.Core metabolism",
    ) -> Builder:
        """
        Create Escher flux visualization using BiGG IDs and calculated fluxes.

        Args:
            output_path (str): Path to save the HTML visualization
            map_name (str): Escher map name to use

        Returns:
            Builder: Escher Builder object
        """
        if not self.average_fluxes:
            print(
                "[ERROR] No average fluxes calculated. Run calculate_generational_average_flux first."
            )
            return None

        print(
            f"[INFO] Creating Escher visualization with {len(self.average_fluxes)} flux values..."
        )

        # Create flux dictionary using BiGG IDs with proper data cleaning
        bigg_flux_dict = {}
        mapped_count = 0

        for _, row in self.mapping_df.iterrows():
            biocyc_id = row["BioCyc_ID"]
            bigg_id = str(row["BiGG_ID"]).strip()  # Ensure string and strip whitespace

            if biocyc_id in self.average_fluxes:
                flux_value = float(self.average_fluxes[biocyc_id])  # Ensure float type

                # Skip invalid values
                if np.isnan(flux_value) or np.isinf(flux_value):
                    print(
                        f"[WARNING] Skipping invalid flux value for {biocyc_id}: {flux_value}"
                    )
                    continue

                bigg_flux_dict[bigg_id] = flux_value
                mapped_count += 1
                print(f"[INFO] Mapped {biocyc_id} -> {bigg_id}: {flux_value:.6f}")

        print(
            f"[INFO] Successfully mapped {mapped_count} reactions for Escher visualization"
        )

        if not bigg_flux_dict:
            print(
                "[ERROR] No flux values mapped to BiGG IDs. Cannot create visualization."
            )
            return None

        # Create Escher builder - using minimal initialization like original working version
        try:
            builder = Builder(
                map_name=map_name,
                reaction_data=bigg_flux_dict,
                reaction_scale=[
                    {"type": "min", "color": "#c8e6c9", "size": 12},
                    {"type": "median", "color": "#81c784", "size": 20},
                    {"type": "max", "color": "#388e3c", "size": 25},
                ],
                reaction_no_data_color="#ddd",
                reaction_no_data_size=8,
            )

            # Save to HTML file
            builder.save_html(output_path)
            print(f"[INFO] Escher visualization saved to: {output_path}")

            # Save as JSON for debugging
            json_path = output_path.replace(".html", "_data.json")
            import json

            with open(json_path, "w") as f:
                json.dump(bigg_flux_dict, f, indent=2)
            print(f"[DEBUG] Flux data also saved as JSON: {json_path}")

            return builder

        except Exception as e:
            print(f"[ERROR] Failed to create Escher visualization: {e}")
            return None

    def generate_flux_summary(self) -> pd.DataFrame:
        """
        Generate a summary DataFrame with BioCyc_ID, BiGG_ID, and calculated fluxes.

        Returns:
            pd.DataFrame: Summary of flux calculations
        """
        if not self.average_fluxes:
            print("[ERROR] No average fluxes calculated.")
            return pd.DataFrame()

        summary_data = []

        for _, row in self.mapping_df.iterrows():
            biocyc_id = row["BioCyc_ID"]
            bigg_id = row["BiGG_ID"]
            original_id = row["Original_Reaction_ID"]

            flux_value = self.average_fluxes.get(biocyc_id, np.nan)

            summary_data.append(
                {
                    "Original_Reaction_ID": original_id,
                    "BiGG_ID": bigg_id,
                    "BioCyc_ID": biocyc_id,
                    "Average_Flux": flux_value,
                    "Has_Flux_Data": not np.isnan(flux_value),
                }
            )

        summary_df = pd.DataFrame(summary_data)

        # Print summary statistics
        total_reactions = len(summary_df)
        reactions_with_data = summary_df["Has_Flux_Data"].sum()
        print(
            f"[INFO] Summary: {reactions_with_data}/{total_reactions} reactions have flux data"
        )

        return summary_df

    def save_flux_summary(self, output_path: str = "flux_summary.csv"):
        """Save flux summary to CSV file."""
        summary_df = self.generate_flux_summary()
        if not summary_df.empty:
            summary_df.to_csv(output_path, index=False)
            print(f"[INFO] Flux summary saved to: {output_path}")
        else:
            print("[WARNING] No summary data to save")


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
    """Complete pipeline to run flux visualization with Escher."""

    # Get parameters
    csv_file_path = params.get("csv_file_path")
    if not csv_file_path:
        print("[ERROR] csv_file_path parameter is required")
        return None

    escher_map_name = params.get("map_name", "e_coli_core.Core metabolism")
    print(f"[INFO] Using Escher map: {escher_map_name}")

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    try:
        # Initialize visualizer
        visualizer = EscherFluxVisualizer(csv_file_path)

        # Calculate generational average flux using the efficient SQL approach
        average_fluxes = visualizer.calculate_generational_average_flux(
            conn, history_sql, config_sql, sim_data_dict
        )

        if not average_fluxes:
            print("[ERROR] No fluxes calculated. Cannot proceed with visualization.")
            return None

        # Generate and save flux summary
        summary_path = os.path.join(outdir, "escher_flux_summary.csv")
        visualizer.save_flux_summary(summary_path)

        # Create Escher visualization
        escher_path = os.path.join(outdir, "escher_flux_visualization.html")
        builder = visualizer.create_escher_flux_map(escher_path, escher_map_name)

        if builder is None:
            print("[ERROR] Failed to create Escher visualization")
            return None

        print("[INFO] Escher flux visualization completed successfully!")
        return visualizer, builder

    except Exception as e:
        print(f"[ERROR] Failed to complete Escher visualization pipeline: {e}")
        return None
