"""Helper functions for vEcoli analysis"""

from typing import List, Tuple, Dict, Optional
import pickle
from collections import defaultdict

from ecoli.library.parquet_emitter import open_arbitrary_sim_data


def categorize_reactions(extended_reactions: List[str]) -> Tuple[List[str], List[str]]:
    """
    Categorize extended reactions into forward and reverse based on naming convention.

    Args:
        extended_reactions: List of extended reaction names

    Returns:
        Tuple of (forward_reactions, reverse_reactions)
    """
    forward_reactions = []
    reverse_reactions = []

    for rxn in extended_reactions:
        if rxn.endswith(" (reverse)"):
            reverse_reactions.append(rxn)
        else:
            forward_reactions.append(rxn)

    return forward_reactions, reverse_reactions


def get_reaction_indices(
    reaction_names: List[str], all_reaction_ids: List[str]
) -> List[int]:
    """
    Get indices of reactions in the flux array.

    Args:
        reaction_names: List of reaction names to find
        all_reaction_ids: List of all reaction IDs from field_metadata

    Returns:
        List of indices (0-based for Python, will be converted to 1-based for SQL)
    """
    indices = []
    for rxn_name in reaction_names:
        try:
            idx = all_reaction_ids.index(rxn_name)
            indices.append(idx)
        except ValueError:
            print(f"[WARNING] Reaction {rxn_name} not found in flux array")

    return indices


def build_flux_calculation_sql(
    biocyc_ids: List[str],
    base_to_extended_mapping: Dict[str, List[str]],
    all_reaction_ids: List[str],
    history_sql: str,
) -> Tuple[Optional[str], List[str]]:
    """
    Build SQL query to calculate net fluxes directly in DuckDB for optimal performance.

    This function generates an optimized SQL query that calculates net flux
    (forward_flux - reverse_flux) for each specified BioCyc ID using streaming
    computation, avoiding the need to load large flux matrices into memory.

    Args:
        biocyc_ids: List of BioCyc IDs (base reaction IDs) to analyze
        base_to_extended_mapping: Mapping from base reaction ID to list of extended reaction names
        all_reaction_ids: List of all reaction IDs from field_metadata (defines flux array order)
        history_sql: SQL query string for accessing historical simulation data

    Returns:
        Tuple of (sql_query_string, valid_biocyc_ids_list)
        - sql_query_string: Complete SQL query for flux calculation, or None if no valid reactions
        - valid_biocyc_ids_list: List of BioCyc IDs that have valid reactions found

    Example:
        For a reaction with forward indices [3,4] and reverse indices [5,6], generates:
        ```
        (fluxes[4] + fluxes[5]) - (fluxes[6] + fluxes[7]) AS "REACTION-ID_net_flux"
        ```
        Note: Indices are converted from 0-based (Python) to 1-based (SQL)
    """
    flux_calculations = []
    valid_biocyc_ids = []

    for biocyc_id in biocyc_ids:
        extended_reactions = base_to_extended_mapping.get(biocyc_id, [])

        if not extended_reactions:
            print(f"[WARNING] No extended reactions found for BioCyc ID: {biocyc_id}")
            continue

        # Separate forward and reverse reactions
        forward_reactions, reverse_reactions = categorize_reactions(extended_reactions)
        forward_indices = get_reaction_indices(forward_reactions, all_reaction_ids)
        reverse_indices = get_reaction_indices(reverse_reactions, all_reaction_ids)

        if not forward_indices and not reverse_indices:
            print(
                f"[WARNING] No valid reaction indices found for BioCyc ID: {biocyc_id}"
            )
            continue

        print(
            f"[INFO] {biocyc_id}: {len(forward_reactions)} forward, {len(reverse_reactions)} reverse reactions"
        )

        # Build SQL expression for net flux calculation
        # Convert to 1-based indexing for SQL (DuckDB arrays are 1-indexed)
        forward_terms = []
        if forward_indices:
            forward_terms = [f"fluxes[{idx + 1}]" for idx in forward_indices]

        reverse_terms = []
        if reverse_indices:
            reverse_terms = [f"fluxes[{idx + 1}]" for idx in reverse_indices]

        # Construct the net flux calculation expression
        flux_expr_parts = []

        # Add forward flux terms
        if forward_terms:
            if len(forward_terms) == 1:
                flux_expr_parts.append(forward_terms[0])
            else:
                flux_expr_parts.append(f"({' + '.join(forward_terms)})")
        else:
            flux_expr_parts.append("0")

        # Subtract reverse flux terms
        if reverse_terms:
            if len(reverse_terms) == 1:
                flux_expr_parts.append(f" - {reverse_terms[0]}")
            else:
                flux_expr_parts.append(f" - ({' + '.join(reverse_terms)})")

        net_flux_expr = "".join(flux_expr_parts)

        # Escape column name with quotes to handle special characters like hyphens
        safe_column_name = f'"{biocyc_id}_net_flux"'
        flux_calculations.append(f"{net_flux_expr} AS {safe_column_name}")
        valid_biocyc_ids.append(biocyc_id)

    if not flux_calculations:
        print("[ERROR] No valid flux calculations could be built")
        return None, []

    # Build complete SQL query with CTE for better readability and performance
    sql = f"""
    WITH renamed AS (
        SELECT time, generation, variant, listeners__fba_results__reaction_fluxes AS fluxes 
        FROM ({history_sql})
    )
    SELECT 
        time,
        generation,
        variant,
        time / 60.0 AS time_min,
        {", ".join(flux_calculations)}
    FROM renamed
    ORDER BY generation, time
    """

    return sql, valid_biocyc_ids


def create_base_to_extended_mapping(sim_data_dict):
    """
    Create reverse mapping from base reaction ID to extended fab reactions.

    Args:
        sim_data_dict: Dictionary containing sim_data information

    Returns:
        dict: Mapping from base reaction ID to list of extended reaction names
    """
    # Load sim_data
    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)
    reaction_ids = sim_data.process.metabolism.reaction_id_to_base_reaction_id

    if not reaction_ids:
        print("[WARNING] Could not find reaction_id_to_base_reaction_id in sim_data")
        return {}

    # Create reverse mapping
    base_to_extended_mapping = defaultdict(list)
    for extended_rxn, base_rxn_id in reaction_ids.items():
        base_to_extended_mapping[base_rxn_id].append(extended_rxn)

    return dict(base_to_extended_mapping)
