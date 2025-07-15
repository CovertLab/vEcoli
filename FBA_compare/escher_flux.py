import pandas as pd
import numpy as np
from escher import Builder
import os
from typing import Dict


class FluxDataVisualizer:
    """
    Flux data visualization tool using BiGG reaction IDs
    """

    def __init__(self):
        self.data_groups = ["data1", "data2", "data3", "data4", "data5"]
        self.output_dir = "flux_visualizations"

        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_flux_data(self, csv_file: str) -> pd.DataFrame:
        """
        Load flux data from CSV file
        """
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
            print(f"Available columns: {list(df.columns)}")
            return df
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            return None

    def filter_exchange_reactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out exchange reactions (containing 'exch')
        """
        # Filter out rows where flux column contains 'exch'
        if "flux" in df.columns:
            original_count = len(df)
            df_filtered = df[~df["flux"].str.contains("exch", case=False, na=False)]
            filtered_count = len(df_filtered)
            print(f"Filtered out {original_count - filtered_count} exchange reactions")
            print(f"Remaining reactions: {filtered_count}")
            return df_filtered
        else:
            print("Warning: 'flux' column not found")
            return df

    def extract_best_fit_data(
        self, df: pd.DataFrame, data_group: str
    ) -> Dict[str, float]:
        """
        Extract best fit data for a specific data group
        """
        best_fit_column = f"{data_group}_best_fit"
        bigg_id_column = "BiGG_id"

        if best_fit_column not in df.columns:
            print(f"Warning: Column '{best_fit_column}' not found")
            return {}

        if bigg_id_column not in df.columns:
            print(f"Warning: Column '{bigg_id_column}' not found")
            return {}

        flux_data = {}
        valid_count = 0

        for _, row in df.iterrows():
            bigg_id = row[bigg_id_column]
            flux_value = row[best_fit_column]

            # Check if both values are valid
            if pd.notna(bigg_id) and pd.notna(flux_value) and bigg_id != "":
                try:
                    flux_data[str(bigg_id)] = float(flux_value)
                    valid_count += 1
                except (ValueError, TypeError):
                    continue

        print(f"Extracted {valid_count} valid flux values for {data_group}")
        return flux_data

    def normalize_flux_data(self, flux_data: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize flux data for better visualization
        """
        if not flux_data:
            return {}

        max_abs_flux = max(abs(v) for v in flux_data.values())
        if max_abs_flux > 0:
            normalized_flux = {k: v / max_abs_flux for k, v in flux_data.items()}
        else:
            normalized_flux = flux_data

        return normalized_flux

    def create_escher_visualization(
        self,
        flux_data: Dict[str, float],
        data_group: str,
        map_name: str = "e_coli_core.Core metabolism",
    ) -> Builder:
        """
        Create Escher visualization for flux data
        """
        if not flux_data:
            print(f"No flux data available for {data_group}")
            return None

        # Normalize data
        normalized_flux = self.normalize_flux_data(flux_data)

        print(
            f"Creating visualization for {data_group} with {len(normalized_flux)} reactions"
        )

        # Create Escher Builder
        builder = Builder(
            map_name=map_name,
            reaction_data=normalized_flux,
            reaction_scale=[
                {"type": "min", "color": "#1e88e5", "size": 8},  # Blue for minimum
                {"type": "median", "color": "#ffc107", "size": 15},  # Yellow for median
                {"type": "max", "color": "#d32f2f", "size": 25},  # Red for maximum
            ],
            reaction_compare_style="log2_fold",
            hide_secondary_metabolites=True,
            show_gene_reaction_rules=False,
        )

        return builder

    def save_visualization(self, builder: Builder, data_group: str):
        """
        Save Escher visualization as HTML file
        """
        if builder is None:
            return

        filename = os.path.join(
            self.output_dir, f"{data_group}_flux_visualization.html"
        )
        builder.save_html(filename)
        print(f"Visualization saved to {filename}")

    def generate_flux_report(
        self, flux_data: Dict[str, float], data_group: str, df: pd.DataFrame
    ):
        """
        Generate detailed flux report for a data group
        """
        if not flux_data:
            print(f"No flux data available for report generation for {data_group}")
            return

        # Create report data
        report_data = []

        lb_column = f"{data_group}_LB95"
        ub_column = f"{data_group}_UB95"

        for bigg_id, flux_value in flux_data.items():
            # Find corresponding row in dataframe
            row = df[df["BiGG_id"] == bigg_id]
            if not row.empty:
                row = row.iloc[0]

                report_entry = {
                    "BiGG_ID": bigg_id,
                    "Flux_Value": flux_value,
                    "LB95": row.get(lb_column, "N/A"),
                    "UB95": row.get(ub_column, "N/A"),
                    "BioCyc_ID": row.get("BioCyc_id", "N/A"),
                    "Type": row.get("type", "N/A"),
                    "Type_Name": row.get("type_name", "N/A"),
                }
                report_data.append(report_entry)

        # Create DataFrame and save
        report_df = pd.DataFrame(report_data)

        # Sort by absolute flux value (descending)
        report_df["Abs_Flux"] = report_df["Flux_Value"].abs()
        report_df = report_df.sort_values("Abs_Flux", ascending=False).drop(
            "Abs_Flux", axis=1
        )

        # Save report
        report_filename = os.path.join(self.output_dir, f"{data_group}_flux_report.csv")
        report_df.to_csv(report_filename, index=False)
        print(f"Flux report saved to {report_filename}")

        # Print statistics
        self.print_flux_statistics(flux_data, data_group, report_df)

    def print_flux_statistics(
        self, flux_data: Dict[str, float], data_group: str, report_df: pd.DataFrame
    ):
        """
        Print flux statistics for a data group
        """
        flux_values = list(flux_data.values())

        print(f"\n=== {data_group.upper()} Statistics ===")
        print(f"Total reactions: {len(flux_values)}")
        print(f"Max flux: {max(flux_values):.4f}")
        print(f"Min flux: {min(flux_values):.4f}")
        print(f"Mean flux: {np.mean(flux_values):.4f}")
        print(f"Median flux: {np.median(flux_values):.4f}")
        print(f"Std flux: {np.std(flux_values):.4f}")

        # Top 5 highest flux reactions
        top_5 = report_df.head(5)
        print("\nTop 5 highest flux reactions:")
        for _, row in top_5.iterrows():
            print(f"  {row['BiGG_ID']}: {row['Flux_Value']:.4f}")

    def generate_summary_report(self, all_flux_data: Dict[str, Dict[str, float]]):
        """
        Generate summary report comparing all data groups
        """
        summary_data = []

        for data_group, flux_data in all_flux_data.items():
            if flux_data:
                flux_values = list(flux_data.values())
                summary_entry = {
                    "Data_Group": data_group,
                    "Total_Reactions": len(flux_values),
                    "Max_Flux": max(flux_values),
                    "Min_Flux": min(flux_values),
                    "Mean_Flux": np.mean(flux_values),
                    "Median_Flux": np.median(flux_values),
                    "Std_Flux": np.std(flux_values),
                }
                summary_data.append(summary_entry)

        summary_df = pd.DataFrame(summary_data)
        summary_filename = os.path.join(self.output_dir, "summary_comparison.csv")
        summary_df.to_csv(summary_filename, index=False)
        print(f"\nSummary comparison saved to {summary_filename}")

        return summary_df

    def process_all_data_groups(
        self, csv_file: str, map_name: str = "e_coli_core.Core metabolism"
    ):
        """
        Process all data groups and generate visualizations and reports
        """
        print("=== Starting Flux Data Processing ===")

        # Load data
        df = self.load_flux_data(csv_file)
        if df is None:
            return

        # Filter exchange reactions
        df_filtered = self.filter_exchange_reactions(df)

        # Process each data group
        all_flux_data = {}

        for data_group in self.data_groups:
            print(f"\n=== Processing {data_group.upper()} ===")

            # Extract flux data
            flux_data = self.extract_best_fit_data(df_filtered, data_group)

            if flux_data:
                # Store for summary
                all_flux_data[data_group] = flux_data

                # Create visualization
                builder = self.create_escher_visualization(
                    flux_data, data_group, map_name
                )

                # Save visualization
                self.save_visualization(builder, data_group)

                # Generate report
                self.generate_flux_report(flux_data, data_group, df_filtered)
            else:
                print(f"No valid flux data found for {data_group}")

        # Generate summary report
        if all_flux_data:
            print("\n=== Generating Summary Report ===")
            summary_df = self.generate_summary_report(all_flux_data)
            print(summary_df)

        print("\n=== Processing Complete ===")
        print(f"All outputs saved to '{self.output_dir}' directory")


# Usage function
def main():
    # Create visualizer
    visualizer = FluxDataVisualizer()

    # Process your CSV file
    csv_file = "merged_flux_data_complete.csv"

    # You can also try different maps if needed:
    # map_name = 'iJO1366.Central metabolism'  # For more comprehensive E. coli model
    # map_name = 'e_coli_core.Core metabolism'  # For core metabolism

    visualizer.process_all_data_groups(csv_file)


if __name__ == "__main__":
    main()
