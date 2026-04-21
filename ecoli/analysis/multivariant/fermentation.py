from duckdb import DuckDBPyConnection
import os
import numpy as np
import altair as alt
import pandas as pd
from typing import Any
from scipy.integrate import odeint

from ecoli.library.parquet_emitter import (
    open_arbitrary_sim_data,
    read_stacked_columns,
)
from reconstruction.ecoli.simulation_data import SimulationDataEcoli

from wholecell.utils import units

import pickle


def derivatives(y, t, r, K, flux_rates, mass):
    N = y[0]

    dNdt = r * N * (1 - N / K)
    dMetabolites = [flux_rate * N for flux_rate in flux_rates]

    return [dNdt] + dMetabolites


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
    """change these to match your experiment, also need to add starting concentrations"""

    init_OD = 0.02  # initial OD from Gingko fermenters
    max_OD = 6000000  # max OD from Gingko fermenters, M5 at 8 hours

    init_cells = init_OD * 8 * 10**8 * 10**3  # initial number of cells per mL
    max_cells = max_OD * 8 * 10**8 * 10**3  # carrying capacity cells per mL

    init_glucose = 20 * 10**15  # initial glucose in fg/L, 20 g/L converted

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data: "SimulationDataEcoli" = pickle.load(f)

    sample_data = conn.sql(f"SELECT * FROM ({history_sql}) LIMIT 1").df()

    # Find all external_exchange_fluxes columns
    flux_columns = [
        col
        for col in sample_data.columns
        if "listeners__fba_results__external_exchange_fluxes__" in col
    ]

    print(f"Found {len(flux_columns)} external exchange flux columns:")
    for col in flux_columns:
        print(f"  {col}")

    if not flux_columns:
        print("No external exchange flux columns found! Available columns:")
        for col in sample_data.columns:
            if "fba" in col.lower():
                print(f"  {col}")
        return

    base_columns = [
        "time",
        "variant",
        "generation",
        "agent_id",
        "listeners__mass__instantaneous_growth_rate",
        "listeners__mass__dry_mass",
        "listeners__mass__cell_mass",
    ]

    quoted_flux_columns = []
    for col in flux_columns:
        if not col.startswith('"'):
            quoted_flux_columns.append(f'"{col}"')
        else:
            quoted_flux_columns.append(col)

    all_columns = base_columns + quoted_flux_columns

    flux_subquery = read_stacked_columns(
        history_sql,
        all_columns,
        order_results=True,
    )

    # Build dynamic SQL for averaging all flux columns with safe aliases
    flux_avg_columns = []
    metabolite_name_mapping = {}

    for i, col in enumerate(flux_columns):
        # Extract metabolite name
        metabolite_name = col.replace(
            "listeners__fba_results__external_exchange_fluxes__", ""
        )

        # Create a safe SQL alias using index to avoid special characters
        alias = f"flux_{i}"

        # Ensure column is properly quoted
        quoted_col = f'"{col}"' if not col.startswith('"') else col
        flux_avg_columns.append(f"AVG({quoted_col}) AS {alias}")
        metabolite_name_mapping[alias] = metabolite_name

    flux_avg_sql = ",\n                ".join(flux_avg_columns)

    sql_query = f"""
        SELECT
            AVG(listeners__mass__instantaneous_growth_rate) as avg_growth_rate, 
            AVG(listeners__mass__dry_mass) AS avg_cell_mass,
            {flux_avg_sql}
        FROM ({flux_subquery})
        """

    print("Executing SQL query...")
    data = conn.sql(sql_query).pl()

    avg_growth_rate = data["avg_growth_rate"][0]
    avg_mass = data["avg_cell_mass"][0]

    print(f"Average growth rate: {avg_growth_rate}")
    print(f"Average cell mass: {avg_mass}")

    # Extract metabolite names and their average fluxes
    metabolite_data = {}

    for alias, metabolite_name in metabolite_name_mapping.items():
        try:
            avg_flux = data[alias][0]

            # Skip if flux is zero or very small
            if abs(avg_flux) < 1e-12:
                print(f"  Skipping {metabolite_name}: flux too small ({avg_flux})")
                continue

            # Get scaling factor
            try:
                flux_scaling_factor = (
                    sim_data.getter.get_mass(metabolite_name)
                    * (units.mmol / units.g / units.h)
                    * units.fg
                ).asNumber()
                scaled_flux = avg_flux * flux_scaling_factor / 3600
                print(
                    f"  {metabolite_name}: {avg_flux:.2e} -> {scaled_flux:.2e} (scaled)"
                )
            except Exception:
                # If mass not found, use a default scaling
                print(
                    f"  Warning: Could not find mass for {metabolite_name}, using raw flux"
                )
                scaled_flux = avg_flux / 3600
                print(f"  {metabolite_name}: {avg_flux:.2e} -> {scaled_flux:.2e} (raw)")

            metabolite_data[metabolite_name] = scaled_flux

        except Exception as e:
            print(f"Error processing {metabolite_name}: {e}")

    if not metabolite_data:
        print("No metabolite data found!")
        return

    print(f"\nProcessing {len(metabolite_data)} metabolites with non-zero fluxes")

    # Set up ODE
    t = np.linspace(0, 36000, 10000)
    N0 = init_cells * avg_mass
    K = max_cells * avg_mass

    # Initialize metabolite concentrations
    initial_concentrations = []
    metabolite_names = list(metabolite_data.keys())
    flux_rates = list(metabolite_data.values())
    # update this
    for name in metabolite_names:
        if "GLC[p]" in name:
            initial_concentrations.append(init_glucose)
        else:
            initial_concentrations.append(0)

    init = [N0] + initial_concentrations

    print(f"Solving ODE with {len(metabolite_names)} metabolites...")

    sol = odeint(
        derivatives,
        init,
        t,
        args=(avg_growth_rate, K, flux_rates, avg_mass),
    )

    N = sol[:, 0]
    metabolite_concentrations = sol[:, 1:]

    df_data = {
        "Time": t / 60,
        "OD_simulated": N / avg_mass / (8 * 10**8) / 1000,
    }

    for i, name in enumerate(metabolite_names):
        # Create safe column name
        clean_name = (
            name.replace("[", "_")
            .replace("]", "")
            .replace("(", "_")
            .replace(")", "")
            .replace("-", "_")
            .replace("+", "plus")
            .replace(" ", "_")
        )
        df_data[f"{clean_name}_simulated"] = metabolite_concentrations[:, i] / 10**15

    df = pd.DataFrame(df_data)

    # Find when glucose runs out (if glucose exists)
    glucose_cols = [col for col in df.columns if "GLC" in col and "simulated" in col]
    if glucose_cols:
        glucose_col = glucose_cols[0]
        if (df[glucose_col] <= 0).any():
            zero_glucose = df[df[glucose_col] <= 0].index[0]
            df = df.iloc[:zero_glucose]
            print(
                f"Truncating at time when glucose runs out: {df['Time'].iloc[-1]:.1f} min"
            )

    # Create visualization for all metabolites
    metabolite_sim_cols = [
        col
        for col in df.columns
        if col.endswith("_simulated") and col != "OD_simulated"
    ]

    print(
        f"Creating plots for metabolites: {[col.replace('_simulated', '') for col in metabolite_sim_cols]}"
    )

    df_sim = df.melt(
        id_vars="Time",
        value_vars=metabolite_sim_cols,
        var_name="Variable",
        value_name="Value",
    )
    df_sim["Variable"] = df_sim["Variable"].str.replace("_simulated", "")
    df_sim["Type"] = "Simulated"
    df_sim["Label"] = df_sim["Variable"] + " (" + df_sim["Type"] + ")"

    # Plot all metabolites
    metabolite_chart = (
        alt.Chart(df_sim)
        .mark_line()
        .encode(
            x=alt.X("Time:Q", axis=alt.Axis(title="Time (min)")),
            y=alt.Y("Value:Q", axis=alt.Axis(title="Concentration (g/L)")),
            color=alt.Color("Label:N", title="Metabolites"),
            tooltip=["Time:Q", "Value:Q", "Variable:N"],
        )
        .properties(
            width=750,
            height=420,
            title="All External Exchange Fluxes - dFBA Simulation",
        )
    )

    # OD chart
    df_od_sim = df[["Time", "OD_simulated"]].rename(columns={"OD_simulated": "Value"})
    df_od_sim["Variable"] = "OD"
    df_od_sim["Type"] = "Simulated"

    od_chart = (
        alt.Chart(df_od_sim)
        .mark_line(strokeDash=[5, 3], color="red")
        .encode(
            x=alt.X("Time:Q", axis=alt.Axis(title="Time (min)")),
            y=alt.Y("Value:Q", axis=alt.Axis(title="OD")),
            tooltip=["Time:Q", "Value:Q"],
        )
        .properties(width=750, height=200, title="Cell Growth (OD)")
    )

    combined = alt.vconcat(metabolite_chart, od_chart)

    combined.save(os.path.join(outdir, "dFBA_all_metabolites_plot.html"))
    print(f"Saved plot to {os.path.join(outdir, 'dFBA_all_metabolites_plot.html')}")

    # Also create log scale version
    EPS = 1e-12
    df_sim_log = df_sim.copy()
    df_sim_log["Value"] = np.log2(np.abs(df_sim_log["Value"]) + EPS)

    df_od_sim_log = df_od_sim.copy()
    df_od_sim_log["Value"] = np.log2(df_od_sim_log["Value"] + EPS)

    metabolite_chart_log = (
        alt.Chart(df_sim_log)
        .mark_line()
        .encode(
            x=alt.X("Time:Q", axis=alt.Axis(title="Time (min)")),
            y=alt.Y("Value:Q", axis=alt.Axis(title="log2(|Concentration|)")),
            color=alt.Color("Label:N", title="Metabolites"),
            tooltip=["Time:Q", "Value:Q", "Variable:N"],
        )
        .properties(
            width=750,
            height=420,
            title="All External Exchange Fluxes - dFBA Simulation (log2 scale)",
        )
    )

    od_chart_log = (
        alt.Chart(df_od_sim_log)
        .mark_line(strokeDash=[5, 3], color="red")
        .encode(
            x=alt.X("Time:Q", axis=alt.Axis(title="Time (min)")),
            y=alt.Y("Value:Q", axis=alt.Axis(title="log2(OD)")),
            tooltip=["Time:Q", "Value:Q"],
        )
        .properties(width=750, height=200, title="Cell Growth (OD) - log2 scale")
    )

    combined_log = alt.vconcat(metabolite_chart_log, od_chart_log)
    combined_log.save(os.path.join(outdir, "dFBA_all_metabolites_plot_log2.html"))
    print(
        f"Saved log plot to {os.path.join(outdir, 'dFBA_all_metabolites_plot_log2.html')}"
    )
