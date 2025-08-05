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
from typing import Dict, List, Tuple, Any

from ecoli.library.parquet_emitter import field_metadata

# ----------------------------------- #


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
        self.reaction_mappings = {}
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

    def find_matching_reactions(
        self, reaction_ids: List[str], reaction_name: str, reverse_flag: bool = False
    ) -> List[Tuple[str, int]]:
        """
        Find all reaction IDs that match the given reaction name pattern.
        Adapted from the original helper function.

        Args:
            reaction_ids (List[str]): List of all reaction IDs in the model
            reaction_name (str): Root reaction name to search for
            reverse_flag (bool): If True, search for reverse reactions

        Returns:
            List[Tuple[str, int]]: List of (reaction_name, index) tuples
        """
        matching_reactions = []

        if reverse_flag:
            # For reverse reactions, look for reactions with "(reverse)" suffix
            reverse_name = reaction_name + " (reverse)"
            if reverse_name in reaction_ids:
                idx = reaction_ids.index(reverse_name)
                matching_reactions.append((reverse_name, idx))

            # Search for extended reverse names with delimiters
            delimiters = ["_", "[", "-", "/"]
            for delimiter in delimiters:
                extend_name = reaction_name + delimiter
                for idx, reaction_id in enumerate(reaction_ids):
                    if (
                        extend_name in reaction_id
                        and "(reverse)" in reaction_id
                        and reaction_id not in [r[0] for r in matching_reactions]
                    ):
                        matching_reactions.append((reaction_id, idx))
        else:
            # For forward reactions, look for reactions WITHOUT "(reverse)" suffix
            if reaction_name in reaction_ids and "(reverse)" not in reaction_name:
                idx = reaction_ids.index(reaction_name)
                matching_reactions.append((reaction_name, idx))

            # Search for extended forward names with delimiters
            delimiters = ["_", "[", "-", "/"]
            for delimiter in delimiters:
                extend_name = reaction_name + delimiter
                for idx, reaction_id in enumerate(reaction_ids):
                    if (
                        extend_name in reaction_id
                        and "(reverse)" not in reaction_id
                        and reaction_id not in [r[0] for r in matching_reactions]
                    ):
                        matching_reactions.append((reaction_id, idx))

        return matching_reactions

    def precompute_reaction_mappings(
        self, reaction_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Precompute all reaction mappings for forward and reverse reactions.
        Adapted from the original helper function.

        Args:
            reaction_ids (List[str]): List of all reaction IDs in the model

        Returns:
            Dict: Mapping of BioCyc ID to forward/reverse reaction indices
        """
        reaction_mappings = {}
        biocyc_ids = self.mapping_df["BioCyc_ID"].unique().tolist()

        for biocyc_id in biocyc_ids:
            print(f"[INFO] Preprocessing reaction mappings for {biocyc_id}...")

            # Find forward reactions
            forward_reactions = self.find_matching_reactions(
                reaction_ids, biocyc_id, reverse_flag=False
            )
            forward_indices = [idx for _, idx in forward_reactions]

            # Find reverse reactions
            reverse_reactions = self.find_matching_reactions(
                reaction_ids, biocyc_id, reverse_flag=True
            )
            reverse_indices = [idx for _, idx in reverse_reactions]

            reaction_mappings[biocyc_id] = {
                "forward_indices": forward_indices,
                "reverse_indices": reverse_indices,
                "forward_reactions": [name for name, _ in forward_reactions],
                "reverse_reactions": [name for name, _ in reverse_reactions],
            }

            print(
                f"[INFO] Found {len(forward_reactions)} forward and {len(reverse_reactions)} reverse reactions for {biocyc_id}"
            )

            if not forward_reactions and not reverse_reactions:
                print(f"[WARNING] No reactions found for {biocyc_id}")

        return reaction_mappings

    def calculate_generational_average_flux(
        self, conn: DuckDBPyConnection, history_sql: str, config_sql: str
    ) -> Dict[str, float]:
        """
        Calculate generational average flux for all BioCyc IDs.

        Args:
            conn: DuckDB connection
            history_sql: SQL for historical data
            config_sql: SQL for configuration data

        Returns:
            Dict[str, float]: Mapping of BioCyc_ID to average flux
        """
        print("[INFO] Calculating generational average flux...")

        # Build SQL query
        sql = f"""
        SELECT
            time,
            listeners__fba_results__reaction_fluxes
        FROM ({history_sql})
        ORDER BY time
        """

        try:
            df = conn.sql(sql).pl()
            print(f"[INFO] Loaded flux data with {df.height} time steps")
        except Exception as e:
            print(f"[ERROR] Error executing SQL query: {e}")
            return {}

        if df.height == 0:
            print("[ERROR] No data found")
            return {}

        # Load reaction IDs from config
        try:
            reaction_ids = field_metadata(
                conn, config_sql, "listeners__fba_results__reaction_fluxes"
            )
            print(f"[INFO] Total reactions in model: {len(reaction_ids)}")
        except Exception as e:
            print(f"[ERROR] Error loading reaction IDs: {e}")
            return {}

        # Precompute reaction mappings
        self.reaction_mappings = self.precompute_reaction_mappings(reaction_ids)

        # Calculate net flux using vectorized operations
        flux_matrix = df.select("listeners__fba_results__reaction_fluxes").to_numpy()
        flux_array = np.vstack([row[0] for row in flux_matrix])

        average_fluxes = {}

        for biocyc_id, mappings in self.reaction_mappings.items():
            forward_indices = mappings["forward_indices"]
            reverse_indices = mappings["reverse_indices"]

            # Skip if no reactions found
            if not forward_indices and not reverse_indices:
                print(f"[WARNING] No reactions found for {biocyc_id}, skipping...")
                continue

            # Calculate forward flux sum
            if forward_indices:
                forward_flux = flux_array[:, forward_indices].sum(axis=1)
            else:
                forward_flux = np.zeros(flux_array.shape[0])

            # Calculate reverse flux sum
            if reverse_indices:
                reverse_flux = flux_array[:, reverse_indices].sum(axis=1)
            else:
                reverse_flux = np.zeros(flux_array.shape[0])

            # Calculate net flux and average
            net_flux = forward_flux - reverse_flux
            avg_net_flux = net_flux.mean()
            average_fluxes[biocyc_id] = avg_net_flux

            print(
                f"[INFO] Average net flux for {biocyc_id}: {avg_net_flux:.6f} mmol/gDW/hr"
            )

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

        # Create flux dictionary using BiGG IDs
        bigg_flux_dict = {}

        for _, row in self.mapping_df.iterrows():
            biocyc_id = row["BioCyc_ID"]
            bigg_id = row["BiGG_ID"]

            if biocyc_id in self.average_fluxes:
                flux_value = self.average_fluxes[biocyc_id]
                bigg_flux_dict[bigg_id] = flux_value
                print(f"[INFO] Mapped {biocyc_id} -> {bigg_id}: {flux_value:.6f}")

        # Create Escher builder
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
        return summary_df

    def save_flux_summary(self, output_path: str = "flux_summary.csv"):
        """Save flux summary to CSV file."""
        summary_df = self.generate_flux_summary()
        summary_df.to_csv(output_path, index=False)
        print(f"[INFO] Flux summary saved to: {output_path}")


# ----------------------------------- #


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
    """Complete pipeline to run flux visualization."""

    # Create output directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    # Initialize visualizer
    csv_file_path = params.get("csv_file_path")
    visualizer = EscherFluxVisualizer(csv_file_path)

    # Calculate generational average flux
    average_fluxes = visualizer.calculate_generational_average_flux(
        conn, history_sql, config_sql
    )

    if not average_fluxes:
        print("[ERROR] No fluxes calculated. Cannot proceed with visualization.")
        return None

    # Generate and save flux summary
    summary_path = os.path.join(outdir, "escher_flux_summary.csv")
    visualizer.save_flux_summary(summary_path)

    # Create Escher visualization
    escher_path = os.path.join(outdir, "escher_flux_visualization.html")
    escher_map_name = params.get("map_name", "e_coli_core.Core metabolism")
    builder = visualizer.create_escher_flux_map(escher_path, escher_map_name)

    return visualizer, builder
