"""
Plot one value per index via heatmap for
new_gene_expression_and_translation_efficiency variant.

Possible Plots:
- Percent of sims that successfully reached a given generation number
- Average doubling time
- Average cell volume, mass, dry cell mass, mRNA mass, protein mass
- Average translation efficiency, weighted by cistron count
- Average mRNA count, monomer count, mRNA mass fraction, protein mass fraction,
    RNAP portion, and ribosome portion for a capacity gene to measure burden on
    overall host expression
- Average new gene copy number
- Average new gene mRNA count
- Average new gene mRNA mass fraction
- Average new gene mRNA counts fraction
- Average new gene NTP mass fraction
- Average new gene protein count
- Average new gene protein mass fraction
- Average new gene protein counts fraction
- Average new gene initialization rate for RNAP and ribosomes
- Average new gene initialization probabilities for RNAP and ribosomes
- Average count and portion of new gene ribosome initialization events per time
    step
- Average number and proportion of RNAP on new genes at a given time step
- Average number and proportion of ribosomes on new gene mRNAs at a given time
    step
- Average number and proportion of RNAP making rRNAs at a given time step
- Average number and proportion of RNAP and ribosomes making RNAP subunits at
    a given time step
- Average number and proportion of RNAP and ribosomes making ribosomal proteins
    at a given time step
- Average fraction of time new gene is overcrowded by RNAP and Ribosomes
- Average overcrowding probability ratio for new gene RNA synthesis and
    polypeptide initiation
- Average max_p probabilities for RNA synthesis and polypeptide initiation
- Average number of overcrowded genes for RNAP and Ribosomes
- Average number of total, active, and free ribosomes
- Average number of ribosomes initialized at each time step
- Average number of total active, and free RNA polymerases
- Average ppGpp concentration
- Average rate of glucose consumption
- Average new gene monomer yields - per hour and per fg of glucose
"""

from duckdb import DuckDBPyConnection
import numpy as np
from matplotlib import pyplot as plt
import math
import pyarrow as pa
from unum.units import fg
from typing import Any, Callable, Optional

from ecoli.variants.new_gene_internal_shift import (
    get_new_gene_ids_and_indices)
from ecoli.library.parquet_emitter import (get_field_metadata,
    ndlist_to_ndarray, read_stacked_columns)
from ecoli.library.schema import bulk_name_to_idx
from wholecell.utils.plotting_tools import heatmap
from wholecell.utils import units

import pickle

FONT_SIZE=9

"""
Dashboard Flag
0: Separate Only (Each plot is its own file)
1: Dashboard Only (One file with all plots)
2: Both Dashboard and Separate
"""
DASHBOARD_FLAG = 2

"""
Standard Deviations Flag
True: Plot an additional copy of all plots with standard deviation displayed
    insted of the average
False: Plot no additional plots
"""
STD_DEV_FLAG = True

"""
Count number of sims that reach this generation (remember index 7 
corresponds to generation 8)
"""
COUNT_INDEX = 23
# COUNT_INDEX = 2 ### TODO: revert back after developing plot locally

"""
Plot data from generations [MIN_CELL_INDEX, MAX_CELL_INDEX)
Note that early generations may not be representative of dynamics 
due to how they are initialized
"""
MIN_CELL_INDEX = 16
# # MIN_CELL_INDEX = 1 ### TODO: revert back after developing plot locally
# MIN_CELL_INDEX = 0
MAX_CELL_INDEX = 24

"""
Specify which subset of heatmaps should be made
Completed_gens heatmap is always made, because it is used to
create the other heatmaps, and should not be included here.
The order listed here will be the order of the heatmaps in the
dashboard plot.
"""
HEATMAPS_TO_MAKE_LIST = [
        "completed_gens_heatmap",
        "doubling_times_heatmap",
        "cell_mass_heatmap",
        "cell_dry_mass_heatmap",
        "cell_volume_heatmap",
        "ppgpp_concentration_heatmap",
        # # "rnap_crowding_heatmap",
        # # "ribosome_crowding_heatmap",
        "cell_mRNA_mass_heatmap",
        "cell_protein_mass_heatmap",
        "rnap_counts_heatmap",
        "ribosome_counts_heatmap",
        "new_gene_mRNA_counts_heatmap",
        "new_gene_monomer_counts_heatmap",
        "new_gene_copy_number_heatmap",
        "new_gene_rnap_init_rate_heatmap",
        "new_gene_ribosome_init_rate_heatmap",
        "new_gene_mRNA_mass_fraction_heatmap",
        "new_gene_monomer_mass_fraction_heatmap",
        "new_gene_rnap_time_overcrowded_heatmap",
        "new_gene_ribosome_time_overcrowded_heatmap",
        "new_gene_mRNA_counts_fraction_heatmap",
        "new_gene_monomer_counts_fraction_heatmap",
        "active_rnap_counts_heatmap",
        "active_ribosome_counts_heatmap",
        "new_gene_rnap_counts_heatmap",
        "new_gene_rnap_portion_heatmap",
        "rrna_rnap_counts_heatmap",
        "rrna_rnap_portion_heatmap",
        "rnap_subunit_rnap_counts_heatmap",
        "rnap_subunit_rnap_portion_heatmap",
        "rnap_subunit_ribosome_counts_heatmap",
        "rnap_subunit_ribosome_portion_heatmap",
        "ribosomal_protein_rnap_counts_heatmap",
        "ribosomal_protein_rnap_portion_heatmap",
        "ribosomal_protein_ribosome_counts_heatmap",
        "ribosomal_protein_ribosome_portion_heatmap",
        "new_gene_ribosome_counts_heatmap",
        "new_gene_ribosome_portion_heatmap",
        # # "weighted_avg_translation_efficiency_heatmap",
        "protein_init_prob_max_p_heatmap",
        "new_gene_protein_init_prob_max_p_target_ratio_heatmap",
        "new_gene_target_protein_init_prob_heatmap",
        "new_gene_actual_protein_init_prob_heatmap",
        "rna_synth_prob_max_p_heatmap",
        "new_gene_rna_synth_prob_max_p_target_ratio_heatmap",
        "new_gene_target_rna_synth_prob_heatmap",
        "new_gene_actual_rna_synth_prob_heatmap",
        "capacity_gene_mRNA_counts_heatmap",
        "capacity_gene_monomer_counts_heatmap",
        "capacity_gene_rnap_portion_heatmap",
        "capacity_gene_ribosome_portion_heatmap",
        "capacity_gene_mRNA_mass_fraction_heatmap",
        "capacity_gene_monomer_mass_fraction_heatmap",
        "capacity_gene_mRNA_counts_fraction_heatmap",
        "capacity_gene_monomer_counts_fraction_heatmap",
        "free_rnap_counts_heatmap",
        "free_ribosome_counts_heatmap",
        "rnap_ribosome_counts_ratio_heatmap",
        "ribosome_init_events_heatmap",
        "new_gene_ribosome_init_events_heatmap",
        "new_gene_ribosome_init_events_portion_heatmap",
        "new_gene_yield_per_glucose",
        "new_gene_yield_per_hour",
        "glucose_consumption_rate",
        "new_gene_mRNA_NTP_fraction_heatmap",
    ]

### TODO map id to common name, don't hardcode, add error checking?
capacity_gene_monomer_id = "EG10544-MONOMER[m]"
capacity_gene_common_name = "lpp"
# capacity_gene_monomer_id = "EG11036-MONOMER[c]"
# capacity_gene_common_name = "tufA"


def data_to_mapping(
    conn: DuckDBPyConnection,
    variant_mapping: dict[int, tuple[int, int]],
    variant_matrix_shape: tuple[int, int],
    history_sql: str,
    columns: list[str],
    projections: Optional[list[str]] = None,
    remove_first: bool = False,
    func: Optional[Callable] = None,
    order_results: bool = False,
    custom_sql: Optional[str] = None,
    post_func: Optional[Callable] = None,
    num_digits_rounding: Optional[int] = None,
    default_value: Optional[Any] = None,
) -> dict[tuple[float, float], Any]:
    """
    Reads one or more columns and calculates some aggregate value for each
    variant. If no custom SQL query is provided, this defaults to averaging
    per cell, then calculating the averages and standard deviations of all
    cells per variant.

    Args:
        conn: DuckDB connection
        variant_mapping: Mapping of variant IDs to row and column in matrix
            of new gene translation efficiency and expression factor variants
        variant_matrix_shape: Number of rows and columns in variant matrix
        history_sql: SQL subquery from :py:func:`ecoli.library.parquet_emitter.get_dataset_sql`
        columns, projections, remove_first, func, order_results: See
            :py:func:`ecoli.library.parquet_emitter.read_stacked_columns`
        custom_sql: SQL string containing a placeholder with name ``subquery``
            where the result of read_stacked_columns will be placed. Final query
            result must only have two columns in order: ``variant`` and a value
            for each variant. If not provided, defaults to average of averages
        post_func: Function that is called on PyArrow table resulting from query.
            Should return a PyArrow table with exactly two columns: one named
            ``variant`` that contains the variant indices, and the other a
            numerical column with any name.
        num_digits_rounding: Number of decimal places to round to
        default_value: 
    
    Returns:
        Mapping of ``(new gene translation efficiency, new gene expression)``
        to value calculated for that variant
    """
    subquery = read_stacked_columns(
        history_sql=history_sql,
        columns=columns,
        projections=projections,
        remove_first=remove_first,
        func=func,
        return_sql=True,
        order_results=order_results
    )
    if custom_sql is None:
        if len(columns) > 1:
            raise RuntimeError("Must provide custom SQL expression to handle "
                               "multiple columns at once.")
        custom_sql = f"""
        WITH avg_per_cell AS (
            SELECT avg({columns[0]}) AS avg_col_per_cell, experiment_id, variant
            FROM ({subquery})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        )
        SELECT variant, avg(avg_col_per_cell) AS mean,
            stddev(avg_col_per_cell) AS std
        FROM avg_per_cell
        GROUP BY experiment_id, variant
        """
    if post_func is None:
        data = conn.sql(custom_sql.format(subquery=subquery)).arrow()
    else:
        data = conn.sql(custom_sql.format(subquery=subquery)).arrow()
        data = post_func(data)
    if set(data.column_names) != {"variant", "mean", "std"}:
        raise RuntimeError("post_func should return a PyArrow table with "
            "exactly three columns named `variant`, `mean`, and `std`")
    data = [(i["variant"], i["mean"], i["std"]) for i in data.to_pylist()]
    mean_matrix = [
        [default_value for _ in range(len(variant_matrix_shape[1]))]
        for _ in range(len(variant_matrix_shape[0]))
    ]
    std_matrix = [
        [default_value for _ in range(len(variant_matrix_shape[1]))]
        for _ in range(len(variant_matrix_shape[0]))
    ]
    for variant, mean, std in data:
        variant_row, variant_col = variant_mapping[variant]
        if num_digits_rounding is not None:
            mean = np.round(mean, num_digits_rounding)
            std = np.round(std, num_digits_rounding)
        mean_matrix[variant_row, variant_col] = mean
        std_matrix[variant_row, variant_col] = std
    return np.array(mean_matrix), np.array(std_matrix)


def get_mRNA_ids_from_monomer_ids(sim_data, target_monomer_ids):
    """
    Map monomer ids back to the mRNA ids that they were translated from.

    Args:
        target_monomer_ids: ids of the monomers to map to mRNA ids

    Returns: set of mRNA ids
    """
    # Map protein ids to cistron ids
    monomer_ids = sim_data.process.translation.monomer_data['id']
    cistron_ids = sim_data.process.translation.monomer_data[
        'cistron_id']
    monomer_to_cistron_id_dict = {
        monomer_id: cistron_ids[i] for i, monomer_id in
        enumerate(monomer_ids)}
    target_cistron_ids = [
        monomer_to_cistron_id_dict.get(RNAP_monomer_id) for
        RNAP_monomer_id in target_monomer_ids]
    # Map cistron ids to RNA indexes
    target_RNA_indexes = [
        sim_data.process.transcription.cistron_id_to_rna_indexes(
            RNAP_cistron_id) for RNAP_cistron_id in
        target_cistron_ids]
    # Map RNA indexes to RNA ids
    RNA_ids = sim_data.process.transcription.rna_data['id']
    target_RNA_ids = set()
    for i in range(len(target_RNA_indexes)):
        for index in target_RNA_indexes[i]:
            target_RNA_ids.add(RNA_ids[index])
    return target_RNA_ids


def get_indexes(
    conn, config_sql, sim_data, index_type, ids
):
    """
    Retrieve indices of a given type for a set of ids.

    Args:
        all_cells: Paths to all cells to read data from (directories should
            contain a simOut/ subdirectory), typically the return from
            AnalysisPaths.get_cells()
        index_type: Type of indexes to extract, currently supported options
            are 'cistron', 'RNA', 'mRNA', and 'monomer'
        capacity_gene_monomer_ids: monomer ids of capacity gene we need
            indexes for

    Returns:
        List of requested indexes
    """
    if index_type == 'cistron':
        # Extract cistron indexes for each new gene
        cistron_idx_dict = {
            cis: i for i, cis in enumerate(get_field_metadata(
                conn, config_sql,
                "listeners__rnap_data__rna_init_event_per_cistron"))}
        indices = [cistron_idx_dict.get(mRNA_id) for mRNA_id in ids]
    elif index_type == 'RNA':
        # Extract RNA indexes for each new gene
        RNA_idx_dict = {
            rna[:-3]: i for i, rna in enumerate(get_field_metadata(
                conn, config_sql,
                "listeners__rna_synth_prob__target_rna_synth_prob"))}
        indices = [RNA_idx_dict.get(mRNA_id) for mRNA_id in ids]
    elif index_type == 'mRNA':
        # Extract mRNA indexes for each new gene
        mRNA_idx_dict = {
            rna[:-3]: i for i, rna in enumerate(get_field_metadata(
                conn, config_sql, "listeners__rna_counts__mRNA_counts"))}
        indices = [mRNA_idx_dict.get(mRNA_id[:-3]) for mRNA_id in ids]
    elif index_type == 'monomer':
        # Extract protein indexes for each new gene
        monomer_idx_dict = {
            monomer: i for i, monomer in enumerate(get_field_metadata(
                conn, config_sql, "listeners__monomer_counts"))}
        indices = [monomer_idx_dict.get(monomer_id) for monomer_id in ids]
    else:
        raise Exception(
            "Index type " + index_type +
            " has no instructions for data extraction.")

    return indices


GENE_COUNTS_SQL = """
    WITH unnested_counts AS (
        SELECT unnest(gene_counts) AS gene_counts,
            generate_subscripts(gene_counts, 1)
            AS gene_idx, experiment_id, variant, lineage_seed, generation,
            agent_id
        FROM ({subquery})
    ),
    avg_per_cell AS (
        SELECT avg(gene_counts) AS avg_count,
            experiment_id, variant, gene_idx
        FROM unnested_counts
        GROUP BY experiment_id, variant, lineage_seed,
            generation, agent_id, gene_idx
    ),
    avg_per_variant AS (
        SELECT avg(log10(avg_count + 1)) AS avg_count,
            stddev(log10(avg_count + 1)) AS std_count,
            experiment_id, variant, gene_idx
        FROM avg_per_cell
        GROUP BY experiment_id, variant, gene_idx
    )
    SELECT variant, list(avg_count ORDER BY gene_idx) AS mean,
        list(std_count ORDER BY gene_idx) AS std,
    FROM avg_per_variant
    GROUP BY experiment_id, variant
    """


def get_gene_mass_fraction_func(sim_data, new_gene_ids):
    # Get mass for gene (mRNAs or monomer)
    new_gene_masses = np.array([
        (sim_data.getter.get_mass(gene_id) / sim_data.constants.n_avogadro).asNumber(fg)
        for gene_id in new_gene_ids
    ])

    def gene_mass_fraction(variant_agg):
        avg_count_over_mass = ndlist_to_ndarray(variant_agg["avg_ratio"])
        std_count_over_mass = ndlist_to_ndarray(variant_agg["std_ratio"])
        # Count / total mass * mass / count = mass fraction
        avg_mass_fractions = avg_count_over_mass * new_gene_masses
        std_mass_fractions = std_count_over_mass * new_gene_masses
        return pa.table({"variant": variant_agg["variant"],
                         "mean": avg_mass_fractions,
                         "std": std_mass_fractions})

    return gene_mass_fraction


def get_gene_count_fraction_sql(gene_indices: list[int], column: str):
    return f"""
        WITH unnested_counts AS (
            SELECT unnest({column}) AS gene_counts,
                generate_subscripts({column}, 1) AS gene_idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
        ),
        -- Materialize in memory so data only needs to be read from disk once
        avg_per_cell AS MATERIALIZED (
            SELECT avg(gene_counts) AS avg_count,
                experiment_id, variant, gene_idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
        ),
        total_avg_per_cell AS (
            SELECT sum(avg_count) as total_avg, experiment_id, variant,
                lineage_seed, generation, agent_id
            FROM avg_per_cell
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id
        ),
        new_gene_avg_per_cell AS (
            SELECT avg_count as new_gene_avg, experiment_id, variant,
                lineage_seed, generation, agent_id, gene_idx
            FROM avg_per_cell
            WHERE gene_idx IN {tuple(gene_indices)}
        ),
        ratio_avg_per_cell AS (
            SELECT capacity_avg / total_avg AS ratio_avg, experiment_id,
                variant, gene_idx
            FROM new_gene_avg_per_cell
            LEFT JOIN total_avg_per_cell USING (experiment_id, variant,
                lineage_seed, generation, agent_id)
        ),
        ratio_avg_per_variant AS (
            SELECT experiment_id, variant, avg(ratio_avg)
                AS ratio_avg, stddev(ratio_avg) AS ratio_std, gene_idx
            FROM ratio_avg_per_cell
            GROUP BY experiment_id, variant, gene_idx
        )
        SELECT variant, list(ratio_avg ORDER BY gene_idx) AS mean,
            list(ratio_std ORDER BY gene_idx) AS std,
        FROM ratio_avg_per_variant
        GROUP BY experiment_id, variant
        """


def get_new_gene_mRNA_NTP_fraction_sql(sim_data, new_gene_mRNA_ids, ntp_ids):
    """
    Special function to handle extraction and saving of new gene mRNA and
    NTP fraction heatmap data.
    """
    # Determine number of NTPs per new gene mRNA and for all mRNAs
    all_rna_counts_ACGU = sim_data.process.transcription.rna_data[
        'counts_ACGU'].asNumber()
    rna_ids = sim_data.process.transcription.rna_data['id']
    rna_id_to_index_mapping = {rna[:-3]: i for i, rna in enumerate(rna_ids)}
    new_gene_mRNA_idx = np.array([
        rna_id_to_index_mapping[rna_id] for rna_id in new_gene_mRNA_ids])
    new_gene_mRNA_ACGU = all_rna_counts_ACGU[new_gene_mRNA_idx]
    all_gene_mRNA_ACGU = (
        all_rna_counts_ACGU[sim_data.process.transcription.rna_data["is_mRNA"]])
    
    # Use DuckDB list comprehension to calculate ratio between each new gene
    # average mRNA count and the total number of an NTP used by all mRNAs. Each
    # NTP gets its own list column in the final result. 
    ntp_projections = ", ".join(["[new_gene_avg / list_dot_product("
        f"all_gene_avg_count, {all_gene_mRNA_ACGU[:, ntp_idx].tolist()}) "
        f"for new_gene_avg in new_gene_avg_count] AS avg_count_over_{ntp_id}"
        for ntp_idx, ntp_id in enumerate(ntp_ids)])
    # Use DuckDB list comprehension to multiply (avg count / total NTP) from
    # above by (NTP / count) to get NTP fraction for each new gene mRNA
    frac_projections = ", ".join([
        f"[i[0] * i[1] for i in list_zip(avg_count_over_{ntp_id}, "
        f"{new_gene_mRNA_ACGU[:, ntp_idx].tolist()})] AS avg_{ntp_id}_frac"
        for ntp_idx, ntp_id in enumerate(ntp_ids)])
    # Unnest NTP fraction list columns so we can average across variants
    unnested_frac_projections = ", ".join([
        f"unnest(avg_{ntp_id}_frac) AS unnested_{ntp_id}_frac"
        for ntp_id in ntp_ids
    ]) + f", generate_subscripts(avg_{ntp_ids[0]}, 1) AS gene_idx"
    avg_frac_projections = ", ".join([
        f"avg(unnested_{ntp_id}_frac) AS avg_{ntp_id}_frac, "
        f"stddev(unnested_{ntp_id}_frac) AS std_{ntp_id}_frac, "
        for ntp_id in ntp_ids
    ])
    # Compile new gene average NTP fractions into list columns
    list_avg_frac_projections = ", ".join([
        f"list(avg_{ntp_id}_frac ORDER BY gene_idx) AS mean, "
        f"list(std_{ntp_id}_frac ORDER BY gene_idx) AS std"
        for ntp_id in ntp_ids
    ])

    return f"""
        WITH unnested_counts AS (
            SELECT unnest(listeners__rna_counts__mRNA_counts) AS gene_counts,
                generate_subscripts(listeners__rna_counts__mRNA_counts, 1)
                AS gene_idx, experiment_id, variant, lineage_seed, generation,
                agent_id
            FROM ({{subquery}})
        ),
        avg_per_cell AS (
            SELECT avg(gene_counts) AS avg_count,
                experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
        ),
        avg_per_cell_lists AS (
            -- Collect all average gene counts in one list column and new gene
            -- average counts in another list column
            SELECT list(avg_count ORDER BY gene_idx) AS all_gene_avg_count,
                list(avg_count ORDER BY gene_idx) FILTER (
                    gene_idx IN {tuple((new_gene_mRNA_idx + 1).tolist())})
                    AS new_gene_avg_count,
                experiment_id, variant
            FROM avg_per_cell
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id
        ),
        avg_count_over_ntp_per_cell AS (
            SELECT experiment_id, variant, {ntp_projections}
            FROM avg_per_cell_lists
        ),
        avg_frac_per_cell AS (
            SELECT experiment_id, variant, {frac_projections}
            FROM avg_count_over_ntp_per_cell
        ),
        unnested_frac_per_cell AS (
            SELECT experiment_id, variant, {unnested_frac_projections}
            FROM avg_frac_per_cell
        ),
        avg_frac_per_variant AS (
            SELECT experiment_id, variant, gene_idx, {avg_frac_projections}
            FROM unnested_frac_per_cell
            GROUP BY experiment_id, variant, gene_idx
        )
        SELECT variant, {list_avg_frac_projections}
        FROM avg_frac_per_variant
        GROUP BY experiment_id, variant
        """


def avg_ratio_of_1d_arrays_sql(numerator, denominator):
    return f"""
        WITH unnested_data AS (
            SELECT unnest({denominator}) AS denominator,
                unnest({numerator}) AS numerator,
                generate_subscripts({numerator}, 1) AS list_idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
        ),
        ratio_avg_per_cell AS (
            SELECT avg(numerator) / avg(denominator)
                AS ratio_avg, experiment_id, variant, list_idx
            FROM unnested_data
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id, list_idx
        ),
        ratio_avg_per_variant AS (
            SELECT avg(ratio_avg) AS ratio_avg, stddev(ratio_avg) AS ratio_std,
                list_idx, experiment_id, variant
            FROM ratio_avg_per_cell
            GROUP BY experiment_id, variant, list_idx
        )
        SELECT variant, list(ratio_avg ORDER BY list_idx) AS mean,
            list(ratio_std ORDER BY list_idx) AS std,
        FROM ratio_avg_per_variant
        GROUP BY experiment_id, variant
        """


def avg_1d_array_sql(column: str):
    return f"""
        WITH unnested_counts AS (
            SELECT unnest({column}) AS array_col,
                generate_subscripts({column}, 1) AS array_idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
        ),
        avg_per_cell AS (
            SELECT avg(array_col) AS avg_array_col,
                experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
        ),
        avg_per_variant AS (
            SELECT avg(avg_array_col) AS avg_array_col,
                stddev(avg_array_col) AS std_array_col,
                experiment_id, variant, gene_idx
            FROM avg_per_cell
            GROUP BY experiment_id, variant, gene_idx
        )
        SELECT variant, list(avg_array_col ORDER BY gene_idx) AS mean,
            list(std_array_col ORDER BY gene_idx) AS std
        FROM avg_per_variant
        GROUP BY experiment_id, variant
        """


def sum_avg_1d_array_sql(column: str):
    return f"""
        WITH unnested_counts AS (
            SELECT unnest({column}) AS array_col,
                generate_subscripts({column}, 1) AS array_idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
        ),
        avg_per_cell AS (
            SELECT avg(array_col) AS avg_array_col,
                experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
        ),
        sum_avg_per_cell AS (
            SELECT sum(avg_array_col) AS avg_array_col,
                experiment_id, variant
            FROM avg_per_cell
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        )
        SELECT variant, avg(avg_array_col) AS mean,
            stddev(avg_array_col) AS std
        FROM sum_avg_per_cell
        GROUP BY experiment_id, variant
        """


def avg_1d_array_over_scalar_sql(array_column: str, scalar_column: str):
    return f"""
        WITH unnested_counts AS (
            SELECT unnest({array_column}) AS array_col,
                generate_subscripts({array_column}, 1) AS array_idx,
                {scalar_column} AS scalar_col,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
        ),
        avg_ratio_per_cell AS (
            SELECT avg(array_col) / avg(scalar_col) AS avg_ratio,
                experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
        ),
        avg_per_variant AS (
            SELECT avg(avg_ratio) AS avg_ratio, stddev(avg_ratio) AS std_ratio,
                experiment_id, variant, gene_idx
            FROM avg_per_cell
            GROUP BY experiment_id, variant, gene_idx
        )
        SELECT variant, list(avg_ratio ORDER BY gene_idx) AS mean,
            list(std_ratio ORDER BY gene_idx) AS std
        FROM avg_per_variant
        GROUP BY experiment_id, variant
        """


def sum_avg_1d_array_over_scalar_sql(array_column: str, scalar_column: str):
    return f"""
        WITH unnested_counts AS (
            SELECT unnest({array_column}) AS array_col,
                generate_subscripts({array_column}, 1) AS array_idx,
                {scalar_column} AS scalar_col,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
        ),
        avg_ratio_per_cell AS (
            SELECT avg(array_col) / avg(scalar_col) AS avg_ratio,
                experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
        ),
        sum_avg_ratio_per_cell AS (
            SELECT sum(avg_ratio) AS avg_ratio, experiment_id, variant
            FROM avg_ratio_per_cell
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        )
        SELECT variant, avg(avg_ratio) AS mean, stddev(avg_ratio) AS std
        FROM sum_avg_ratio_per_cell
        GROUP BY experiment_id, variant
        """


NEW_GENE_TIME_OVERCROWDED_SQL = """
    -- Boolean values must be explicitly cast to numeric for aggregation
    WITH unnested_overcrowded AS (
        SELECT unnest(overcrowded::TINYINT) AS unnested_data,
            generate_subscripts(overcrowded, 1) AS list_idx,
            experiment_id, variant, lineage_seed, generation, agent_id
        FROM ({{subquery}})
    ),
    avg_per_cell AS (
        SELECT avg(unnested_data) AS frac_time_overcrowded,
            experiment_id, variant, list_idx
        FROM unnested_overcrowded
        GROUP BY experiment_id, variant, lineage_seed, generation,
            agent_id, list_idx
    ),
    avg_per_variant AS (
        SELECT avg(frac_time_overcrowded) AS avg_frac_time_overcrowded,
            stddev(frac_time_overcrowded) AS std_frac_time_overcrowded,
            experiment_id, variant, list_idx
        FROM avg_per_cell
        GROUP BY experiment_id, variant, list_idx
    )
    SELECT variant, list(avg_frac_time_overcrowded ORDER BY list_idx) AS mean,
        list(std_frac_time_overcrowded ORDER BY list_idx) AS std
    FROM avg_per_variant
    GROUP BY experiment_id, variant
    """


def get_rnap_counts_projection(sim_data, bulk_ids):
    """
    Return SQL projection to selectively read bulk inactive RNAP count.
    """
    rnap_idx = bulk_name_to_idx(
        [sim_data.molecule_ids.full_RNAP], bulk_ids)
    return f"bulk[{rnap_idx + 1}] AS bulk"


def get_ribosome_counts_projection(sim_data, bulk_ids):
    """
    Return SQL projection to selectively read bulk inactive ribosome count
    (defined as minimum of free 30S and 50S subunits at any given moment)
    """
    ribosome_idx = bulk_name_to_idx(
        [sim_data.molecule_ids.s30_full_complex,
            sim_data.molecule_ids.s50_full_complex], bulk_ids)
    return f"least(bulk[{ribosome_idx[0] + 1}], bulk[{ribosome_idx[1] + 1}]) AS bulk"


def get_overcrowding_sql(
    target_col: str,
    actual_col: str,
):
    """
    Returns mapping from tuples (translation efficiency, expression) to the
    number of genes (averaged over all cells per variant) for which the
    target transcription probability was higher than the actual (averaged
    over all timesteps per cell in that variant).
    """
    return f"""
        WITH unnested_probs AS (
            SELECT unnest({actual_col}) AS actual, experiment_id, variant,
                lineage_seed, generation, unnest({target_col}) AS target,
                agent_id, generate_subscripts({target_col}, 1) AS list_idx
            FROM ({{subquery}})
        ),
        overcrowded_genes AS (
            SELECT avg(actual) > avg(target) AS overcrowded,
                experiment_id, variant, lineage_seed, generation, agent_id,
            FROM unnested_probs
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id,
                list_idx
        ),
        num_overcrowded_per_cell AS (
            SELECT sum(overcrowded::BIGINT) AS overcrowded_per_cell,
                experiment_id, variant
            FROM overcrowded_genes
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        )
        SELECT variant, avg(num_overcrowded_per_cell) AS mean,
            stddev(num_overcrowded_per_cell) AS std,
        FROM num_overcrowded_per_cell
        GROUP BY experiment_id, variant
        """


def get_trl_eff_weighted_avg_post_func(sim_data, cistron_ids):
    # Get normalized translation efficiency for all mRNAs
    trl_effs = sim_data.process.translation.translation_efficiencies_by_monomer
    trl_eff_ids = sim_data.process.translation.monomer_data['cistron_id']
    mRNA_cistron_idx_dict = {rna: i for i, rna in enumerate(cistron_ids)}
    trl_eff_id_mapping = np.array([
        mRNA_cistron_idx_dict[id] for id in trl_eff_ids])

    def trl_eff_weighted_avg(variant_agg):
        avg_counts = ndlist_to_ndarray(variant_agg["mean"])
        total_avg_counts = np.expand_dims(
            avg_counts.sum(axis=1), axis=1
        )
        # Compute average translation efficiency, weighted by mRNA counts
        weighted_avg_trl_eff = np.dot(avg_counts / total_avg_counts,
            trl_effs[np.argsort(trl_eff_id_mapping)])
        return pa.table({"variant": variant_agg["variant"],
                         "mean": weighted_avg_trl_eff})

    return trl_eff_weighted_avg


def get_new_gene_mass_yield_func(sim_data, new_gene_ids):
    # Get mass for each new gene
    new_gene_masses = np.array([
        (sim_data.getter.get_mass(gene_id) /
         sim_data.constants.n_avogadro).asNumber(fg)
        for gene_id in new_gene_ids])

    def new_gene_mass_yield(variant_agg):
        avg_array = ndlist_to_ndarray(variant_agg["mean"])
        std_array = ndlist_to_ndarray(variant_agg["std"])
        return pa.table({"variant": variant_agg["variant"],
                         "mean": avg_array * new_gene_masses,
                         "std": std_array * new_gene_masses})

    return new_gene_mass_yield


"""
Plotting Functions
"""
def plot_heatmaps(
    self, is_dashboard, variant_mask, heatmap_x_label, heatmap_y_label,
    new_gene_expression_factors, new_gene_translation_efficiency_values,
    summary_statistic, figsize_x, figsize_y, plotOutDir, plot_suffix):
    """
    Plots all heatmaps in order given by HEATMAPS_TO_MAKE_LIST.

    Args:
        is_dashboard: Boolean flag for whether we are creating a dashboard
            of heatmaps or a number of individual heatmaps
        variant_mask: np.array of dimension
            (len(new_gene_translation_efficiency_values),
            len(new_gene_expression_factors)) with entries set to True if
            variant was run, False otherwise.
        heatmap_x_label: Label for x axis of heatmap
        heatmap_y_label: Label for y axis of heatmap
        new_gene_expression_factors: New gene expression factors used in
            these variants
        new_gene_translation_efficiency_values: New gene translation
            efficiency values used in these variants
        summary_statistic: Specifies whether average ('mean') or
            standard deviation ('std_dev') should be displayed on the
            heatmaps
        figsize_x: Horizontal size of each heatmap
        figsize_y: Vertical size of each heatmap
        plotOutDir: Output directory for plots
        plot_suffix: Suffix to add to plot file names, usually specifying
            which generations were plotted
    """
    if summary_statistic == 'std_dev':
        plot_suffix = plot_suffix + "_std_dev"
    elif summary_statistic != 'mean':
        raise Exception(
            "'mean' and 'std_dev' are the only currently supported"
            " summary statistics")

    if is_dashboard:
        # Determine dashboard layout
        if self.total_heatmaps_to_make > 3:
            dashboard_ncols = 4
            dashboard_nrows = math.ceil((self.total_heatmaps_to_make + 1) / dashboard_ncols)
        else:
            dashboard_ncols = self.total_heatmaps_to_make + 1
            dashboard_nrows = 1
        fig, axs = plt.subplots(nrows=dashboard_nrows,
            ncols=dashboard_ncols,
            figsize=(figsize_y * dashboard_ncols,figsize_x * dashboard_nrows),
            layout='constrained'
            )
        if dashboard_nrows == 1:
            axs = np.reshape(axs, (1, dashboard_ncols))

        # Percent Completion Heatmap
        heatmap(
            self, axs[0,0], variant_mask,
            self.heatmap_data["completed_gens_heatmap"][0,:,:],
            self.heatmap_data["completed_gens_heatmap"][0,:,:],
            new_gene_expression_factors,
            new_gene_translation_efficiency_values,
            heatmap_x_label,
            heatmap_y_label,
            f"Percentage of Sims That Reached Generation {COUNT_INDEX + 1}")
        row_ax = 0
        col_ax = 1

        for h in HEATMAPS_TO_MAKE_LIST:
            if not self.heatmap_details[h]["is_nonstandard_plot"]:
                stop_index = 1
                title_addition = ""
                if self.heatmap_details[h]["is_new_gene_heatmap"]:
                    stop_index = len(self.new_gene_mRNA_ids)
                for i in range(stop_index):
                    if self.heatmap_details[h]["is_new_gene_heatmap"]:
                        title_addition = f": {self.new_gene_mRNA_ids[i][:-4]}"
                    self.make_single_heatmap(
                        h, axs[row_ax, col_ax], variant_mask,
                        heatmap_x_label, heatmap_y_label, i,
                        new_gene_expression_factors,
                        new_gene_translation_efficiency_values,
                        summary_statistic, title_addition)
                    col_ax += 1
                    if (col_ax == dashboard_ncols):
                        col_ax = 0
                        row_ax += 1
            elif h == "new_gene_mRNA_NTP_fraction_heatmap":
                for i in range(len(self.new_gene_mRNA_ids)):
                    for ntp_id in self.ntp_ids:
                        self.make_new_gene_mRNA_NTP_fraction_heatmap(
                            h, axs[row_ax, col_ax], variant_mask,
                            heatmap_x_label, heatmap_y_label, i,
                            new_gene_expression_factors,
                            new_gene_translation_efficiency_values,
                            summary_statistic,
                            ntp_id)
                        fig.tight_layout()
                        plt.show()
                        exportFigure(
                            plt, plotOutDir,
                            f'new_gene_mRNA_{ntp_id[:-3]}_fraction_heatmap'
                            f'_{self.new_gene_mRNA_ids[i][:-4]}'
                            f'{plot_suffix}')
                        col_ax += 1
                        if (col_ax == dashboard_ncols):
                            col_ax = 0
                            row_ax += 1
            else:
                raise Exception(
                    f"Heatmap {h} is neither a standard plot nor a"
                    f" nonstandard plot that has specific instructions for"
                    f" plotting.")
        fig.tight_layout()
        exportFigure(plt, plotOutDir,
            f"00_new_gene_exp_trl_eff_dashboard{plot_suffix}") ## TODO: Revert back after running new plots on Sherlock sims
        plt.close("all")

    else: # individual plots
        # Plot percent completion heatmap
        fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
        heatmap(
            self, ax, variant_mask,
            self.heatmap_data["completed_gens_heatmap"][0, :, :],
            self.heatmap_data["completed_gens_heatmap"][0, :, :],
            new_gene_expression_factors,
            new_gene_translation_efficiency_values,
            heatmap_x_label,
            heatmap_y_label,
            f"Percentage of Sims that Reached Generation {COUNT_INDEX + 1}")
        fig.tight_layout()
        plt.show()
        exportFigure(plt, plotOutDir, 'completed_gens_heatmap')

        for h in HEATMAPS_TO_MAKE_LIST:
            if not self.heatmap_details[h]["is_nonstandard_plot"]:
                stop_index = 1
                title_addition = ""
                filename_addition = ""
                if self.heatmap_details[h]["is_new_gene_heatmap"]:
                    stop_index = len(self.new_gene_mRNA_ids)
                for i in range(stop_index):
                    if self.heatmap_details[h]["is_new_gene_heatmap"]:
                        title_addition = f": {self.new_gene_mRNA_ids[i][:-4]}"
                        filename_addition = f"_{self.new_gene_mRNA_ids[i][:-4]}"
                    fig, ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
                    self.make_single_heatmap(
                        h, ax, variant_mask, heatmap_x_label, heatmap_y_label,
                        i, new_gene_expression_factors,
                        new_gene_translation_efficiency_values,
                        summary_statistic, title_addition)
                    fig.tight_layout()
                    plt.show()
                    exportFigure(plt, plotOutDir, h + filename_addition +
                        plot_suffix)
                    plt.close()
            elif h == "new_gene_mRNA_NTP_fraction_heatmap":
                for i in range(len(self.new_gene_mRNA_ids)):
                    for ntp_id in self.ntp_ids:
                        fig, ax = plt.subplots(1, 1, figsize=(figsize_x,
                            figsize_y))
                        self.make_new_gene_mRNA_NTP_fraction_heatmap(
                            h, ax, variant_mask, heatmap_x_label,
                            heatmap_y_label, i, new_gene_expression_factors,
                            new_gene_translation_efficiency_values,
                            summary_statistic, ntp_id)
                        fig.tight_layout()
                        plt.show()
                        exportFigure(
                            plt, plotOutDir,
                            f'new_gene_mRNA_{ntp_id[:-3]}_fraction_heatmap'
                            f'_{self.new_gene_mRNA_ids[i][:-4]}{plot_suffix}')
            else:
                raise Exception(
                    f"Heatmap {h} is neither a standard plot nor a"
                    f" nonstandard plot that has specific instructions for"
                    f" plotting.")


def make_single_heatmap(
        self, h, ax, variant_mask, heatmap_x_label, heatmap_y_label,
        initial_index, new_gene_expression_factors,
        new_gene_translation_efficiency_values,
        summary_statistic, title_addition):
    """
    Creates a heatmap for h.

    Args:
        h: Heatmap identifier
        ax: Axes to plot on
        variant_mask: np.array of dimension
            (len(new_gene_translation_efficiency_values),
            len(new_gene_expression_factors)) with entries set to True if
            variant was run, False otherwise.
        heatmap_x_label: Label for x axis of heatmap
        heatmap_y_label: Label for y axis of heatmap
        initial_index: 0 for non new gene heatmaps, otherwise the relative
            index of the new gene
        new_gene_expression_factors: New gene expression factors used in
            these variants
        new_gene_translation_efficiency_values: New gene translation
            efficiency values used in these variants
        summary_statistic: Specifies whether average ('mean') or
            standard deviation ('std_dev') should be displayed on the
            heatmaps
        title_addition: Any string that needs to be added to the title of
            the heatmap, e.g. a new gene id
    """
    title = self.heatmap_details[h]['plot_title'] + title_addition
    if summary_statistic == "std_dev":
        title = f"Std Dev: {title}"

    heatmap(
        self, ax, variant_mask,
        self.heatmap_data[h][summary_statistic][initial_index, :, :],
        self.heatmap_data["completed_gens_heatmap"][0, :, :],
        new_gene_expression_factors, new_gene_translation_efficiency_values,
        heatmap_x_label, heatmap_y_label,
        title,
        self.heatmap_details[h]['box_text_size'])


def make_new_gene_mRNA_NTP_fraction_heatmap(
        self, h, ax, variant_mask, heatmap_x_label, heatmap_y_label,
        initial_index, new_gene_expression_factors,
        new_gene_translation_efficiency_values, summary_statistic, ntp_id):
    """
    Special function that creates a new gene mRNA NTP fraction heatmap for
    one new gene and one NTP.

    Args:
        h: Heatmap identifier
        ax: Axes to plot on
        variant_mask: np.array of dimension
            (len(new_gene_translation_efficiency_values),
            len(new_gene_expression_factors)) with entries set to True if
            variant was run, False otherwise.
        heatmap_x_label: Label for x axis of heatmap
        heatmap_y_label: Label for y axis of heatmap
        initial_index: 0 for non new gene heatmaps, otherwise the relative
            index of the new gene
        new_gene_expression_factors: New gene expression factors used in
            these variants
        new_gene_translation_efficiency_values: New gene translation
            efficiency values used in these variants
        summary_statistic: Specifies whether average ('mean') or
            standard deviation ('std_dev') should be displayed on the
            heatmaps
        ntp_id: Id of NTP to plot
    """
    title = (f"{self.heatmap_details[h]['plot_title']} {ntp_id[:-3]}"
                f" Fraction: {self.new_gene_mRNA_ids[initial_index][:-4]}")
    if summary_statistic == 'std_dev':
        title = f"Std Dev: {title}"

    heatmap(
        self, ax, variant_mask,
        self.heatmap_data[h][summary_statistic][ntp_id][initial_index, :, :],
        self.heatmap_data["completed_gens_heatmap"][0, :, :],
        new_gene_expression_factors, new_gene_translation_efficiency_values,
        heatmap_x_label, heatmap_y_label,
        title,
        self.heatmap_details[h]['box_text_size'])


def plot(
    params: dict[str, Any],
    conn: DuckDBPyConnection,
    history_sql: str,
    config_sql: str,
    sim_data_paths: dict[int, list[str]],
    validation_data_paths: list[str],
    outdir: str,
    variant_metadata: dict[int, Any]
):
    """
    Create either a single multi-heatmap plot or 1+ separate heatmaps of data
    for a grid of new gene variant simulations with varying expression and
    translation efficiencies. 
    """
    with open(sim_data_paths[0], 'rb') as f:
        sim_data = pickle.load(f)

    # Determine new gene mRNA and monomer ids
    (new_gene_mRNA_ids, new_gene_indices, new_gene_monomer_ids,
        new_gene_monomer_indices) = get_new_gene_ids_and_indices(sim_data)

    # Assuming we ran a workflow with `n` new gene expression factors
    # and `m` new gene translation efficiency values, create an `n * m`
    # grid sorted in ascending order along both axes and calculate
    # mapping from variant numbers to row and column indices in grid
    assert ("exp_trl_eff" in variant_metadata["0"]), (
        "This plot is intended to be run on simulations where the"
        " new gene expression-translation efficiency variant was "
        "enabled, but no parameters for this variant were found.")
    new_gene_expression_factors = sorted(set(
        variant_params["exp_trl_eff"]["exp"]
        for variant_params in variant_metadata.values()))
    new_gene_translation_efficiency_values = sorted(set(
        variant_params["exp_trl_eff"]["trl_eff"]
        for variant_params in variant_metadata.values()))
    variant_to_row_col = {
        variant: (
                new_gene_translation_efficiency_values.index(
                    variant_params["exp_trl_eff"]["trl_eff"]),
                new_gene_expression_factors.index(
                    variant_params["exp_trl_eff"]["exp"]),
            )
        for variant, variant_params in variant_metadata.items()
    }

    bulk_ids = get_field_metadata(conn, config_sql, "bulk")
    ntp_ids = list(sim_data.ntp_code_to_id_ordered.values())
    rnap_subunit_mRNA_ids = get_mRNA_ids_from_monomer_ids(
        sim_data, sim_data.molecule_groups.RNAP_subunits)
    rnap_subunit_mRNA_indexes = get_indexes(
        conn, config_sql, sim_data, "mRNA", rnap_subunit_mRNA_ids
    )
    rnap_subunit_monomer_indexes = get_indexes(
        conn, config_sql, sim_data, "monomer",
        sim_data.molecule_groups.RNAP_subunits
    )
    ribosomal_mRNA_ids = get_mRNA_ids_from_monomer_ids(
        sim_data, sim_data.molecule_groups.ribosomal_proteins)
    ribosomal_mRNA_indexes = get_indexes(
        conn, config_sql, sim_data, "mRNA", ribosomal_mRNA_ids
    )
    ribosomal_monomer_indexes = get_indexes(
        conn, config_sql, sim_data, "monomer",
        sim_data.molecule_groups.ribosomal_proteins
    )
    cistron_ids = get_field_metadata(conn, config_sql,
        "listeners__rna_counts__full_mRNA_cistron_counts")
    capacity_gene_mRNA_ids = list(get_mRNA_ids_from_monomer_ids(
        sim_data, [capacity_gene_monomer_id]))
    capacity_gene_monomer_ids = [capacity_gene_monomer_id]
    capacity_gene_indexes = {
        index_type: get_indexes(conn, config_sql, sim_data, index_type, capacity_gene_mRNA_ids)
        for index_type in ["mRNA", "cistron", "RNA"]
    }
    capacity_gene_indexes["monomer"] = get_indexes(conn, config_sql, sim_data, "monomer", capacity_gene_monomer_ids)
    new_gene_indexes = {
        index_type: get_indexes(conn, config_sql, sim_data, index_type, new_gene_mRNA_ids)
        for index_type in ["mRNA", "cistron", "RNA"]
    }
    new_gene_indexes["monomer"] = get_indexes(conn, config_sql, sim_data, "monomer", new_gene_monomer_ids)

    FLUX_UNITS = units.mmol / units.g / units.h
    MASS_UNITS = units.fg

    # Determine glucose index in exchange fluxes
    external_molecule_ids = np.array(get_field_metadata(
        conn, config_sql, "listeners__fba_results__external_exchange_fluxes"
    ))
    if "GLC[p]" not in external_molecule_ids:
        print("This plot only runs when glucose is the carbon source.")
        return
    glucose_idx = np.where(external_molecule_ids == "GLC[p]")[0][0]
    flux_scaling_factor = FLUX_UNITS * MASS_UNITS * sim_data.getter.get_mass("GLC[p]")

    # Get normalized translation efficiency for all mRNAs
    trl_effs = sim_data.process.translation.translation_efficiencies_by_monomer
    trl_eff_ids = sim_data.process.translation.monomer_data['cistron_id']
    mRNA_cistron_idx_dict = {rna: i for i, rna in enumerate(cistron_ids)}
    trl_eff_id_mapping = np.array([
        mRNA_cistron_idx_dict[id] for id in trl_eff_ids])
    ordered_trl_effs = trl_effs[np.argsort(trl_eff_id_mapping)]
    """
    Details needed to create all possible heatmaps
        key (string): Heatmap identifier, will also be used in file name if
            plots are saved separately
        is_new_gene_heatmap (bool): If True, one heatmap will be made 
            for each new gene
        is_nonstandard_plotting (bool): False if only one plot (or one plot
            per new gene) needs to be made. True in all other cases.
        data_table (string): Table to get data from.
        data_column (string): Column in table to get data from.
        default_value (int): Value to use in heatmap if no data is 
            extracted for this parameter combination.
        remove_first (bool): If True, removes the first column of data 
            from each cell (which might be set to a default value in 
            some cases)
        num_digits_rounding (int): Specifies how to round the number 
            displayed in each sequare of the heatmap
        box_text_size (string): Specifies font size of number displayed 
            in each square of the heatmap
        plot_title (string): Title of heatmap to display
    """
    # Defaults - unless otherwise specified, these values will be
    # used for plotting
    default_is_nonstandard_plot = False
    default_value = -1
    default_remove_first = False
    default_num_digits_rounding = 2
    default_box_text_size = 'medium'
    # Specify unique fields and non-default values here
    heatmap_details = {
        "completed_gens_heatmap": {
            "columns": ["generation"],
            "custom_sql": f"""
                WITH max_gen_per_seed AS (
                    SELECT max(generation) AS max_generation, experiment_id, variant
                    FROM ({{subquery}})
                    GROUP BY experiment_id, variant, lineage_seed
                )
                -- Boolean values must be explicitly cast to numeric for aggregation
                SELECT variant, avg((max_generation > {COUNT_INDEX})::BIGINT)
                FROM max_gen_per_seed
                GROUP BY experiment_id, variant
                """,
            "plot_title": f"Percentage of Sims That Reached Generation {COUNT_INDEX}",
        },
        "doubling_times_heatmap": {
            "columns": ["time"],
            "custom_sql": """
                WITH doubling_times AS (
                    SELECT (max(time) - min(time)) / 60 AS doubling_time,
                        experiment_id, variant, lineage_seed, generation, agent_id
                    FROM ({subquery})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(doubling_time)
                FROM doubling_times
                GROUP BY experiment_id, variant
                """,
            "plot_title": "Doubling Time (minutes)",
        },
        "cell_volume_heatmap": {
            "columns": ["listeners__mass__volume"],
            "plot_title": "Cell Volume (fL)",
        },
        "cell_mass_heatmap": {
            "columns": ["listeners__mass__cell_mass"],
            "plot_title": "Cell Mass (fg)",
        },
        "cell_dry_mass_heatmap": {
            "columns": ["listeners__mass__dry_mass"],
            "plot_title": "Dry Cell Mass (fg)",
        },
        "cell_mRNA_mass_heatmap": {
            "columns": ["listeners__mass__mRna_mass"],
            "plot_title": "Total mRNA Mass (fg)",
        },
        "cell_protein_mass_heatmap": {
            "columns": ["listeners__mass__protein_mass"],
            "plot_title": "Total Protein Mass (fg)",
            "box_text_size": "x-small",
            "num_digits_rounding": 0,
        },
        "ppgpp_concentration_heatmap": {
            "columns": ["listeners__growth_limits__ppgpp_conc"],
            "plot_title": "ppGpp Concentration (uM)",
            "remove_first": True,
            "num_digits_rounding": 1,
        },
        "ribosome_init_events_heatmap": {
            "columns": ["listeners__ribosome_data__did_initialize"],
            "plot_title": "Ribosome Init Events Per Time Step",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
        },
        "rnap_counts_heatmap": {
            "columns": ["bulk", "listeners__unique_molecule_counts__active_RNAP"],
            "plot_title": "RNA Polymerase (RNAP) Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projections": [get_rnap_counts_projection(sim_data, bulk_ids), None],
            "custom_sql": """
                WITH total_counts AS (
                    SELECT avg(bulk +
                        listeners__unique_molecule_counts__active_RNAP) AS rnap_counts,
                        experiment_id, variant
                    FROM ({subquery})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(rnap_counts)
                FROM total_counts
                GROUP BY experiment_id, variant
                """
        },
        "ribosome_counts_heatmap": {
            "columns": ["bulk", "listeners__unique_molecule_counts__active_ribosome"],
            "plot_title": "Active Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projections": [get_ribosome_counts_projection(sim_data, bulk_ids), None],
            "custom_sql": """
                WITH total_counts AS (
                    SELECT avg(bulk +
                        listeners__unique_molecule_counts__active_ribosome)
                        AS ribosome_counts,
                        experiment_id, variant
                    FROM ({subquery})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(ribosome_counts)
                FROM inactive_counts
                GROUP BY experiment_id, variant
                """
        },
        "active_ribosome_counts_heatmap": {
            "columns": ["listeners__unique_molecule_counts__active_ribosome"],
            "plot_title": "Active Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
        },
        "active_rnap_counts_heatmap": {
            "columns": ["listeners__unique_molecule_counts__active_RNAP"],
            "plot_title": "Active RNA Polymerase (RNAP) Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
        },
        "free_ribosome_counts_heatmap": {
            "columns": ["bulk"],
            "plot_title": "Free Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projections": [get_ribosome_counts_projection(sim_data, bulk_ids)],
        },
        "free_rnap_counts_heatmap": {
            "columns": ["bulk"],
            "plot_title": "Free RNA Polymerase (RNAP) Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projections": [get_rnap_counts_projection(sim_data, bulk_ids)],
        },
        "rnap_ribosome_counts_ratio_heatmap": {
            "columns": ["bulk", "listeners__unique_molecule_counts__active_RNAP",
                        "listeners__unique_molecule_counts__active_RNAP"],
            "plot_title": "RNAP Counts / Ribosome Counts",
            "num_digits_rounding": 4,
            "box_text_size": "x-small",
            "projections": [get_rnap_counts_projection(sim_data, bulk_ids) +
                "_rnap, " + get_ribosome_counts_projection(sim_data, bulk_ids) +
                "_ribosome", None, None],
            "custom_sql": """
                WITH total_counts AS (
                    SELECT avg(bulk_ribosome +
                        listeners__unique_molecule_counts__active_ribosome)
                        AS ribosome_counts,
                        avg(bulk_rnap +
                        listeners__unique_molecule_counts__active_RNAP) AS rnap_counts,
                        experiment_id, variant
                    FROM ({{subquery}})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(rnap_counts) / avg(ribosome_counts)
                FROM total_counts
                GROUP BY experiment_id, variant
                """
        },
        "rnap_crowding_heatmap": {
            "columns": ["listeners__rna_synth_prob__target_rna_synth_prob",
                        "listeners__rna_synth_prob__actual_rna_synth_prob"],
            "plot_title": "RNAP Crowding: # of TUs",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "custom_sql": get_overcrowding_sql(
                "listeners__rna_synth_prob__target_rna_synth_prob",
                "listeners__rna_synth_prob__actual_rna_synth_prob")
        },
        "ribosome_crowding_heatmap": {
            "columns": [
                "listeners__ribosome_data__target_prob_translation_per_transcript",
                "listeners__ribosome_data__actual_prob_translation_per_transcript"],
            "plot_title": "Ribosome Crowding: # of Monomers",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "custom_sql": get_overcrowding_sql(
                "listeners__ribosome_data__target_prob_translation_per_transcript",
                "listeners__ribosome_data__actual_prob_translation_per_transcript")
        },
        "rna_synth_prob_max_p_heatmap": {
            "columns": ["listeners__rna_synth_prob__max_p"],
            "plot_title": "RNA Synth Max Prob",
            "num_digits_rounding": 4,
        },
        "protein_init_prob_max_p_heatmap": {
            "columns": ["listeners__ribosome_data__max_p"],
            "plot_title": "Protein Init Max Prob",
            "num_digits_rounding": 4,
        },
        "weighted_avg_translation_efficiency_heatmap": {
            "columns": ["listeners__rna_counts__full_mRNA_cistron_counts"],
            "plot_title": "Translation Efficiency (Weighted Average)",
            "num_digits_rounding": 3,
            "custom_sql": f"""
                WITH unnested_counts AS (
                    SELECT unnest(listeners__rna_counts__full_mRNA_cistron_counts) AS array_col,
                        generate_subscripts(listeners__rna_counts__full_mRNA_cistron_counts, 1) AS array_idx,
                        unnest({ordered_trl_effs.tolist()}) AS trl_effs,
                        experiment_id, variant, lineage_seed, generation, agent_id
                    FROM ({{subquery}})
                ),
                avg_per_cell AS (
                    SELECT avg(array_col * trl_effs) / sum(array_col) OVER
                        (PARTITION BY experiment_id, variant, lineage_seed,
                        generation, agent_id) AS avg_array_col,
                        experiment_id, variant, lineage_seed,
                        generation, agent_id, gene_idx
                    FROM unnested_counts
                    GROUP BY experiment_id, variant, lineage_seed,
                        generation, agent_id, gene_idx
                ),
                avg_per_variant AS (
                    SELECT avg(avg_array_col) AS avg_array_col,
                        stddev(avg_array_col) AS std_array_col,
                        experiment_id, variant, gene_idx
                    FROM avg_per_cell
                    GROUP BY experiment_id, variant, gene_idx
                )
                SELECT variant, list(avg_array_col ORDER BY gene_idx) AS mean,
                    list(std_array_col ORDER BY gene_idx) AS std
                FROM avg_per_variant
                GROUP BY experiment_id, variant
                """,
            "post_func": get_trl_eff_weighted_avg_post_func(
                sim_data, cistron_ids),
        },
        "capacity_gene_mRNA_counts_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts"],
            "plot_title": "Log(Capacity Gene mRNA Counts+1): "
                + capacity_gene_common_name,
            "projection": ["list_select(listeners__rna_counts__mRNA_counts, "
                            f"{capacity_gene_indexes['mRNA']}) AS gene_counts"],
            "custom_sql": GENE_COUNTS_SQL,
        },
        "capacity_gene_monomer_counts_heatmap": {
            "columns": ["listeners__monomer_counts"],
            "plot_title": "Log(Capacity Gene Protein Counts+1): "
                + capacity_gene_common_name,
            "projection": ["list_select(listeners__monomer_counts, "
                            f"{capacity_gene_indexes['monomer']}) AS gene_counts"],
            "custom_sql": GENE_COUNTS_SQL,
        },
        "capacity_gene_mRNA_mass_fraction_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts",
                        "listeners__mass__mRna_mass"],
            "plot_title": "Capacity Gene mRNA Mass Fraction: "
                + capacity_gene_common_name,
            "num_digits_rounding": 3,
            "projection": ["list_select(listeners__rna_counts__mRNA_counts, "
                            f"{capacity_gene_indexes['mRNA']}) AS gene_counts",
                            "listeners__mass__mRna_mass AS mass"],
            "custom_sql": avg_1d_array_over_scalar_sql("gene_counts", "mass"),
            "post_func": get_gene_mass_fraction_func(
                sim_data, capacity_gene_mRNA_ids),
        },
        "capacity_gene_monomer_mass_fraction_heatmap": {
            "columns": ["listeners__monomer_counts",
                        "listeners__mass__protein_mass"],
            "plot_title": "Capacity Gene Protein Mass Fraction: "
                + capacity_gene_common_name,
            "num_digits_rounding": 3,
            "projection": ["list_select(listeners__monomer_counts, "
                            f"{capacity_gene_indexes['monomer']}) AS gene_counts",
                            "listeners__mass__protein_mass AS mass"],
            "custom_sql": avg_1d_array_over_scalar_sql("gene_counts", "mass"),
            "post_func": get_gene_mass_fraction_func(
                sim_data, capacity_gene_monomer_ids),
        },
        "capacity_gene_mRNA_counts_fraction_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts"],
            "plot_title": "Capacity Gene mRNA Counts Fraction: "
                + capacity_gene_common_name,
            "num_digits_rounding": 3,
            "custom_sql": get_gene_count_fraction_sql(
                capacity_gene_indexes["mRNA"],
                "listeners__rna_counts__mRNA_counts"),
        },
        "capacity_gene_monomer_counts_fraction_heatmap": {
            "columns": ["listeners__monomer_counts"],
            "plot_title": "Capacity Gene Protein Counts Fraction: "
                + capacity_gene_common_name,
            "num_digits_rounding": 3,
            "custom_sql": get_gene_count_fraction_sql(
                capacity_gene_indexes["monomer"],
                "listeners__monomer_counts"),
        },
        "new_gene_copy_number_heatmap": {
            "columns": ["listeners__rna_synth_prob__gene_copy_number"],
            "plot_title": "New Gene Copy Number",
            "projections": ["list_select(listeners__rna_synth_prob__gene_copy_number, "
                            f"{new_gene_indexes['cistron']}) AS gene_counts"],
            "num_digits_rounding": 3,
            "custom_sql": avg_1d_array_sql("listeners__rna_synth_prob__gene_copy_number"),
        },
        "new_gene_mRNA_counts_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts"],
            "plot_title": "Log(New Gene mRNA Counts+1)",
            "projections": ["list_select(listeners__rna_counts__mRNA_counts, "
                            f"{new_gene_indexes['mRNA']}) AS gene_counts"],
            "custom_sql": GENE_COUNTS_SQL,
        },
        "new_gene_monomer_counts_heatmap": {
            "columns": ["listeners__monomer_counts"],
            "plot_title": "Log(New Gene Protein Counts+1)",
            "projections": ["list_select(listeners__monomer_counts, "
                            f"{new_gene_indexes['monomer']}) AS gene_counts"],
            "custom_sql": GENE_COUNTS_SQL,
        },
        "new_gene_mRNA_mass_fraction_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts",
                        "listeners__mass__mRna_mass"],
            "plot_title": "New Gene mRNA Mass Fraction",
            "projection": ["list_select(listeners__rna_counts__mRNA_counts, "
                            f"{new_gene_indexes['mRNA']}) AS gene_counts",
                            "listeners__mass__mRna_mass AS mass"],
            "custom_sql": avg_1d_array_over_scalar_sql("gene_counts", "mass"),
            "post_func": get_gene_mass_fraction_func(
                sim_data, new_gene_mRNA_ids),
        },
        "new_gene_mRNA_counts_fraction_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts"],
            "plot_title": "New Gene mRNA Counts Fraction",
            "custom_sql": get_gene_count_fraction_sql(
                new_gene_indexes["mRNA"],
                "listeners__rna_counts__mRNA_counts"),
        },
        "new_gene_mRNA_NTP_fraction_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts"],
            "plot_title": "New Gene",
            "num_digits_rounding": 4,
            "box_text_size": "x-small",
            "custom_sql": get_new_gene_mRNA_NTP_fraction_sql(
                sim_data, new_gene_mRNA_ids, ntp_ids),
            "is_nonstandard_plot": True,
        },
        "new_gene_monomer_mass_fraction_heatmap": {
            "columns": ["listeners__monomer_counts",
                        "listeners__mass__protein_mass"],
            "plot_title": "New Gene Protein Mass Fraction",
            "projection": ["list_select(listeners__monomer_counts, "
                            f"{new_gene_indexes['monomer']}) AS gene_counts",
                            "listeners__mass__protein_mass AS mass"],
            "custom_sql": avg_1d_array_over_scalar_sql("gene_counts", "mass"),
            "post_func": get_gene_mass_fraction_func(
                sim_data, new_gene_monomer_ids),
        },
        "new_gene_monomer_counts_fraction_heatmap": {
            "columns": ["listeners__monomer_counts"],
            "plot_title": "New Gene Protein Counts Fraction",
            "custom_sql": get_gene_count_fraction_sql(
                new_gene_indexes["monomer"],
                "listeners__monomer_counts"),
        },
        "new_gene_rnap_init_rate_heatmap": {
            "columns": ["listeners__rna_synth_prob__gene_copy_number",
                        "listeners__rnap_data__rna_init_event_per_cistron"],
            "plot_title": "New Gene RNAP Initialization Rate",
            "projection": ["list_select(listeners__rna_synth_prob__gene_copy_number, "
                    f"{new_gene_indexes['cistron']}) AS gene_copy_number",
                "list_select(listeners__rnap_data__rna_init_event_per_cistron, "
                    f"{new_gene_indexes['cistron']}) AS rna_init_event_per_cistron"],
            "custom_sql": avg_ratio_of_1d_arrays_sql(
                "rna_init_event_per_cistron", "gene_copy_number"),
        },
        "new_gene_ribosome_init_rate_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts",
                        "listeners__ribosome_data__ribosome_init_event_per_monomer"],
            "plot_title": "New Gene Ribosome Initalization Rate",
            "projection": ["list_select(listeners__rna_counts__mRNA_counts, "
                    f"{new_gene_indexes['mRNA']}) AS mRNA_counts",
                "list_select(listeners__ribosome_data__ribosome_init_event_per_monomer, "
                    f"{new_gene_indexes['monomer']}) AS ribosome_init_event_per_monomer"],
            "custom_sql": avg_ratio_of_1d_arrays_sql(
                "ribosome_init_event_per_monomer", "mRNA_counts"),
        },
        "new_gene_rnap_time_overcrowded_heatmap": {
            "columns": ["listeners__rna_synth_prob__tu_is_overcrowded"],
            "plot_title": "Fraction of Time RNAP Overcrowded New Gene",
            "projection": ["list_select(listeners__rna_synth_prob__tu_is_overcrowded, "
                    f"{new_gene_indexes['RNA']}) AS overcrowded"],
            "custom_sql": NEW_GENE_TIME_OVERCROWDED_SQL,
        },
        "new_gene_ribosome_time_overcrowded_heatmap": {
            "columns": ["listeners__ribosome_data__mRNA_is_overcrowded"],
            "plot_title": "Fraction of Time Ribosome Overcrowded New Gene",
            "projection": ["list_select(listeners__ribosome_data__mRNA_is_overcrowded, "
                    f"{new_gene_indexes['monomer']}) AS overcrowded"],
            "custom_sql": NEW_GENE_TIME_OVERCROWDED_SQL,
        },
        "new_gene_actual_protein_init_prob_heatmap": {
            "columns": ["listeners__ribosome_data__actual_prob_translation_per_transcript"],
            "plot_title": "New Gene Actual Protein Init Prob",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__ribosome_data__actual_prob_translation_per_transcript, "
                    f"{new_gene_indexes['monomer']}) AS init_prob"],
            "custom_sql": avg_1d_array_sql("init_prob"),
        },
        "new_gene_target_protein_init_prob_heatmap": {
            "columns": ["listeners__ribosome_data__target_prob_translation_per_transcript"],
            "plot_title": "New Gene Target Protein Init Prob",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__ribosome_data__target_prob_translation_per_transcript, "
                    f"{new_gene_indexes['monomer']}) AS init_prob"],
            "custom_sql": avg_1d_array_sql("init_prob"),
        },
        "new_gene_protein_init_prob_max_p_target_ratio_heatmap": {
            "columns": ["listeners__ribosome_data__target_prob_translation_per_transcript",
                        "listeners__ribosome_data__max_p"],
            "plot_title": "New Gene Protein Max Prob / Target Prob Ratio",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__ribosome_data__target_prob_translation_per_transcript, "
                    f"{new_gene_indexes['monomer']}) AS target_prob",
                "list_select(listeners__ribosome_data__max_p, "
                    f"{new_gene_indexes['monomer']}) AS max_p"
                ],
            "custom_sql": avg_ratio_of_1d_arrays_sql("target_prob", "max_p"),
        },
        "new_gene_rna_synth_prob_max_p_target_ratio_heatmap": {
            "columns": ["listeners__rna_synth_prob__target_rna_synth_prob",
                        "listeners__rna_synth_prob__max_p"],
            "plot_title": "New Gene Protein Max Prob / Target Prob Ratio",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__rna_synth_prob__target_rna_synth_prob, "
                    f"{new_gene_indexes['monomer']}) AS target_prob",
                "listeners__rna_synth_prob__max_p AS max_p"],
            "custom_sql": avg_1d_array_over_scalar_sql("target_prob", "max_p"),
        },
        "new_gene_ribosome_init_events_heatmap": {
            "columns": ["listeners__ribosome_data__ribosome_init_event_per_monomer"],
            "plot_title": "New Gene Ribosome Init Events Per Time Step",
            "num_digits_rounding": 0,
            "box_test_size": "x-small",
            "projection": [
                "list_select(listeners__ribosome_data__ribosome_init_event_per_monomer, "
                    f"{new_gene_indexes['monomer']}) AS init_events"],
            "custom_sql": avg_1d_array_sql("init_events"),
        },
        "new_gene_ribosome_init_events_portion_heatmap": {
            "columns": ["listeners__ribosome_data__ribosome_init_event_per_monomer",
                        "listeners__ribosome_data__did_initialize"],
            "plot_title": "New Gene Portion of Initiated Ribosomes",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__ribosome_data__ribosome_init_event_per_monomer, "
                    f"{new_gene_indexes['monomer']}) AS init_events",
                "listeners__ribosome_data__did_initialize AS did_initialize"],
            "custom_sql": avg_1d_array_over_scalar_sql("init_events", "did_initialize"),
        },
        "new_gene_actual_rna_synth_prob_heatmap": {
            "columns": ["listeners__rna_synth_prob__actual_rna_synth_prob"],
            "plot_title": "New Gene Actual RNA Synth Prob",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__rna_synth_prob__actual_rna_synth_prob, "
                    f"{new_gene_indexes['RNA']}) AS synth_probs"],
            "custom_sql": avg_1d_array_sql("synth_probs"),
        },
        "new_gene_target_rna_synth_prob_heatmap": {
            "columns": ["listeners__rna_synth_prob__target_rna_synth_prob"],
            "plot_title": "New Gene Target RNA Synth Prob",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__rna_synth_prob__target_rna_synth_prob, "
                    f"{new_gene_indexes['RNA']}) AS synth_probs"],
            "custom_sql": avg_1d_array_sql("synth_probs"),
        },
        "new_gene_rnap_counts_heatmap": {
            "columns": ["listeners__rna_counts__partial_mRNA_counts"],
            "plot_title": "New Gene RNAP Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projection": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                    f"{new_gene_indexes['mRNA']}) AS rnap_counts"],
            "custom_sql": avg_1d_array_sql("rnap_counts"),
        },
        "new_gene_rnap_portion_heatmap": {
            "columns": ["listeners__rna_counts__partial_mRNA_counts",
                        "listeners__unique_molecule_counts__active_RNAP"],
            "plot_title": "New Gene RNAP Portion",
            "num_digits_rounding": 3,
            "projection": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                    f"{new_gene_indexes['mRNA']}) AS rnap_counts",
                "listeners__unique_molecule_counts__active_RNAP AS active_counts"],
            "custom_sql": avg_1d_array_over_scalar_sql("rnap_counts", "active_counts"),
        },
        "rrna_rnap_counts_heatmap": {
            "columns": ["listeners__rna_counts__partial_rRNA_counts"],
            "plot_title": "rRNA RNAP Counts",
            "num_digits_rounding": 0,
            "projection": [
                "listeners__rna_counts__partial_mRNA_counts AS rnap_counts"],
            "custom_sql": sum_avg_1d_array_sql("rnap_counts"),
        },
        "rrna_rnap_portion_heatmap": {
            "columns": ["listeners__rna_counts__partial_rRNA_counts",
                        "listeners__unique_molecule_counts__active_RNAP"],
            "plot_title": "rRNA RNAP Portion",
            "num_digits_rounding": 3,
            "projection": [
                "listeners__rna_counts__partial_mRNA_counts AS rnap_counts",
                "listeners__unique_molecule_counts__active_RNAP AS active_counts"],
            "custom_sql": sum_avg_1d_array_over_scalar_sql(
                "rnap_counts", "active_counts"),
        },
        "rnap_subunit_rnap_counts_heatmap": {
            "columns": ["listeners__rna_counts__partial_mRNA_counts"],
            "plot_title": "RNAP Subunit RNAP Counts",
            "num_digits_rounding": 0,
            "projection": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                    f"{rnap_subunit_mRNA_indexes}) AS rnap_counts"],
            "custom_sql": sum_avg_1d_array_sql("rnap_counts"),
        },
        "rnap_subunit_rnap_portion_heatmap": {
            "columns": ["listeners__rna_counts__partial_mRNA_counts",
                        "listeners__unique_molecule_counts__active_RNAP"],
            "plot_title": "RNAP Subunit RNAP Portion",
            "num_digits_rounding": 3,
            "projection": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                    f"{rnap_subunit_mRNA_indexes}) AS rnap_counts",
                "listeners__unique_molecule_counts__active_RNAP AS active_counts"],
            "custom_sql": sum_avg_1d_array_over_scalar_sql(
                "rnap_counts", "active_counts"),
        },
        "rnap_subunit_ribosome_counts_heatmap": {
            "columns": ["listeners__ribosome_data__n_ribosomes_per_transcript"],
            "plot_title": "RNAP Subunit Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projection": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                    f"{rnap_subunit_monomer_indexes}) AS ribosome_counts"],
            "custom_sql": sum_avg_1d_array_sql("ribosome_counts"),
        },
        "rnap_subunit_ribosome_portion_heatmap": {
            "columns": ["listeners__ribosome_data__n_ribosomes_per_transcript",
                        "listeners__unique_molecule_counts__active_ribosome"],
            "plot_title": "RNAP Subunit Ribosome Portion",
            "num_digits_rounding": 3,
            "projection": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                    f"{rnap_subunit_monomer_indexes}) AS ribosome_counts",
                "listeners__unique_molecule_counts__active_ribosome AS active_counts"],
            "custom_sql": sum_avg_1d_array_over_scalar_sql(
                "ribosome_counts", "active_counts"),
        },
        "ribosomal_protein_rnap_counts_heatmap": {
            "columns": ["listeners__rna_counts__partial_mRNA_counts"],
            "plot_title": "Ribosomal Protein RNAP Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projection": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                    f"{ribosomal_mRNA_indexes}) AS rnap_counts"],
            "custom_sql": sum_avg_1d_array_sql("rnap_counts"),
        },
        "ribosomal_protein_rnap_portion_heatmap": {
            "columns": ["listeners__rna_counts__partial_mRNA_counts",
                        "listeners__unique_molecule_counts__active_RNAP"],
            "plot_title": "Ribosomal Protein RNAP Portion",
            "num_digits_rounding": 3,
            "projection": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                    f"{ribosomal_mRNA_indexes}) AS rnap_counts",
                "listeners__unique_molecule_counts__active_RNAP AS active_counts"],
            "custom_sql": sum_avg_1d_array_over_scalar_sql(
                "rnap_counts", "active_counts"),
        },
        "ribosomal_protein_ribosome_counts_heatmap": {
            "columns": ["listeners__ribosome_data__n_ribosomes_per_transcript"],
            "plot_title": "Ribosomal Protein Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projection": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                    f"{ribosomal_monomer_indexes}) AS ribosome_counts"],
            "custom_sql": sum_avg_1d_array_sql("ribosome_counts"),
        },
        "ribosomal_protein_ribosome_portion_heatmap": {
            "columns": ["listeners__ribosome_data__n_ribosomes_per_transcript",
                        "listeners__unique_molecule_counts__active_ribosome"],
            "plot_title": "Ribosomal Protein Ribosome Portion",
            "num_digits_rounding": 3,
            "projection": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                    f"{ribosomal_monomer_indexes}) AS ribosome_counts",
                "listeners__unique_molecule_counts__active_ribosome AS active_counts"],
            "custom_sql": sum_avg_1d_array_over_scalar_sql(
                "ribosome_counts", "active_counts"),
        },
        # Note: These four were changed from wcEcoli implementation to sum average
        # ribosome count or portion for all new or capacity genes per cell instead
        # of averaging them per cell (both then average variant)
        "new_gene_ribosome_counts_heatmap": {
            "columns": ["listeners__ribosome_data__n_ribosomes_per_transcript"],
            "plot_title": "New Gene Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "projection": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                    f"{new_gene_indexes['monomer']}) AS ribosome_counts"],
            "custom_sql": sum_avg_1d_array_sql("ribosome_counts"),
        },
        "new_gene_ribosome_portion_heatmap": {
            "columns": ["listeners__ribosome_data__n_ribosomes_per_transcript",
                        "listeners__unique_molecule_counts__active_ribosome"],
            "plot_title": "New Gene Ribosome Portion",
            "num_digits_rounding": 3,
            "projection": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                    f"{new_gene_indexes['monomer']}) AS ribosome_counts",
                "listeners__unique_molecule_counts__active_ribosome AS active_counts"],
            "custom_sql": sum_avg_1d_array_over_scalar_sql(
                "ribosome_counts", "active_counts"),
        },
        "capacity_gene_rnap_portion_heatmap": {
            "columns": ["listeners__rna_counts__partial_mRNA_counts"
                        "listeners__unique_molecule_counts__active_RNAP"],
            "plot_title": f"Capacity Gene RNAP Portion: {capacity_gene_common_name}",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                    f"{capacity_gene_indexes['mRNA']}) AS rnap_counts",
                "listeners__unique_molecule_counts__active_RNAP AS active_counts"],
            "custom_sql": sum_avg_1d_array_over_scalar_sql(
                "rnap_counts", "active_counts"),
        },
        "capacity_gene_ribosome_portion_heatmap": {
            "columns": ["listeners__ribosome_data__n_ribosomes_per_transcript",
                        "listeners__unique_molecule_counts__active_ribosome"],
            "plot_title": f"Capacity Gene Ribosome Portion: {capacity_gene_common_name}",
            "num_digits_rounding": 4,
            "projection": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                    f"{capacity_gene_indexes['monomer']}) AS ribosome_counts",
                "listeners__unique_molecule_counts__active_ribosome AS active_counts"],
            "custom_sql": sum_avg_1d_array_over_scalar_sql(
                "ribosome_counts", "active_counts"),
        },
        "glucose_consumption_rate": {
            "columns": ["listeners__fba_results__external_exchange_fluxes",
                        "listeners__mass__dry_mass"],
            "plot_title": "Average Glucose Consumption Rate (fg/hr)",
            "num_digits_rounding": 1,
            "projection": [
                "listeners__fba_results__external_exchange_fluxes["
                    f"{glucose_idx}]) AS glucose_flux",
                "listeners__mass__dry_mass AS dry_mass"],
            "remove_first": True,
            "custom_sql": f"""
                WITH avg_per_cell AS (
                    SELECT avg(glucose_flux) * avg(dry_mass) * {flux_scaling_factor}
                        AS avg_fg_glc_per_hr, experiment_id, variant
                    FROM ({{subquery}})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(avg_fg_glc_per_hr) AS avg_fg_glc_per_hr
                FROM avg_per_cell
                GROUP BY experiment_id, variant
                """,
        },
        "new_gene_yield_per_glucose": {
            "columns": ["listeners__fba_results__external_exchange_fluxes",
                "listeners__mass__dry_mass", "time", "listeners__monomer_counts"],
            "plot_title": "New Gene fg Protein Yield per fg Glucose",
            "num_digits_rounding": 3,
            "projection": [
                "listeners__fba_results__external_exchange_fluxes["
                    f"{glucose_idx}]) AS glucose_flux",
                "listeners__mass__dry_mass AS dry_mass", None,
                "list_select(listeners__monomer_counts, "
                    f"{new_gene_indexes['monomer']}) AS monomer_counts"],
            "remove_first": True,
            "custom_sql": f"""
                WITH unnested_counts AS (
                    SELECT unnest(monomer_counts) AS monomer_counts,
                        generate_subscripts(monomer_counts, 1) AS monomer_idx,
                        glucose_flux, dry_mass, time
                    FROM ({{subquery}})
                ),
                avg_per_cell AS (
                    SELECT avg(glucose_flux) * avg(dry_mass) * {flux_scaling_factor}
                        AS avg_fg_glc_per_hr, (max(time) - min(time)) / 60
                        AS doubling_time, last(monomer_counts ORDER BY time) -
                        first(monomer_counts ORDER BY time) AS monomer_delta,
                        experiment_id, variant, monomer_idx
                    FROM unnested_counts
                    GROUP BY experiment_id, variant, lineage_seed, generation,
                        agent_id, monomer_idx
                ),
                avg_per_variant AS (
                    SELECT experiment_id, variant, avg(monomer_delta / (
                        avg_fg_glc_per_hr * doubling_time))
                        AS avg_monomer_per_fg_glc, monomer_idx
                    FROM avg_per_cell
                    GROUP BY experiment_id, variant, monomer_idx
                )
                SELECT variant, list(avg_monomer_per_fg_glc ORDER BY monomer_idx)
                    AS variant_avg
                FROM avg_per_variant
                GROUP BY experiment_id, variant
                """,
            "post_func": get_new_gene_mass_yield_func(
                sim_data, new_gene_monomer_ids),
        },
        "new_gene_yield_per_hour": {
            "columns": ["time", "listeners__monomer_counts"],
            "plot_title": "New Gene fg Protein Yield per Hour",
            "num_digits_rounding": 2,
            "projection": [
                None, "list_select(listeners__monomer_counts, "
                    f"{new_gene_indexes['monomer']}) AS monomer_counts"],
            "remove_first": True,
            "custom_sql": f"""
                WITH unnested_counts AS (
                    SELECT unnest(monomer_counts) AS monomer_counts, time,
                        generate_subscripts(monomer_counts, 1) AS monomer_idx
                    FROM ({{subquery}})
                ),
                avg_per_cell AS (
                    SELECT (max(time) - min(time)) / 60 AS doubling_time,
                        last(monomer_counts ORDER BY time) - first(
                            monomer_counts ORDER BY time) AS monomer_delta,
                        experiment_id, variant, monomer_idx
                    FROM unnested_counts
                    GROUP BY experiment_id, variant, lineage_seed, generation,
                        agent_id, monomer_idx
                ),
                avg_per_variant AS (
                    SELECT experiment_id, variant, avg(monomer_delta /
                        doubling_time) AS avg_monomer_per_hr, monomer_idx
                    FROM avg_per_cell
                    GROUP BY experiment_id, variant, monomer_idx
                )
                SELECT variant, list(avg_monomer_per_hr ORDER BY monomer_idx)
                    AS variant_avg
                FROM avg_per_variant
                GROUP BY experiment_id, variant
                """,
            "post_func": get_new_gene_mass_yield_func(
                sim_data, new_gene_monomer_ids),
        },
    }

    # Check validity of requested heatmaps and fill in default values where needed
    heatmaps_to_make = set(HEATMAPS_TO_MAKE_LIST)
    total_heatmaps_to_make = 0
    for h in heatmaps_to_make:
        assert h in heatmap_details, "Heatmap " + h + " is not an option"
        heatmap_details[h]['is_new_gene_heatmap'] = h.startswith("new_gene_")
        heatmap_details[h].setdefault(
            'is_nonstandard_plot',default_is_nonstandard_plot)
        heatmap_details[h].setdefault(
            'default_value', default_value)
        heatmap_details[h].setdefault(
            'remove_first', default_remove_first)
        heatmap_details[h].setdefault(
            'num_digits_rounding', default_num_digits_rounding)
        heatmap_details[h].setdefault(
            'box_text_size', default_box_text_size)
        if not h.startswith("new_gene_"):
            total_heatmaps_to_make += 1
        elif h == "new_gene_mRNA_NTP_fraction_heatmap":
            total_heatmaps_to_make += len(ntp_ids)
        else:
            total_heatmaps_to_make += len(new_gene_mRNA_ids)

    # Create data structures to use for the heatmaps
    heatmap_data = {}
    heatmap_data["completed_gens_heatmap"] = np.zeros((
        1, len(new_gene_translation_efficiency_values),
        len(new_gene_expression_factors)))
    for h in heatmaps_to_make:
        if not heatmap_details[h]['is_new_gene_heatmap']:
            heatmap_data[h] = {}
            heatmap_data[h]["mean"] = np.zeros((
                1, len(new_gene_translation_efficiency_values),
                len(new_gene_expression_factors))
                ) + heatmap_details[h]['default_value']
            heatmap_data[h]["std_dev"] = np.zeros((
                1, len(new_gene_translation_efficiency_values),
                len(new_gene_expression_factors))
            ) + heatmap_details[h]['default_value']
        else:
            if h == "new_gene_mRNA_NTP_fraction_heatmap":
                heatmap_data[h] = {}
                heatmap_data[
                    "new_gene_mRNA_NTP_fraction_heatmap"]["mean"] = {}
                heatmap_data[
                    "new_gene_mRNA_NTP_fraction_heatmap"]["std_dev"] = {}
                for ntp_id in ntp_ids:
                    heatmap_data[
                        "new_gene_mRNA_NTP_fraction_heatmap"][
                        "mean"][ntp_id] = np.zeros(
                        (len(new_gene_mRNA_ids),
                            len(new_gene_translation_efficiency_values),
                            len(new_gene_expression_factors))
                        ) + heatmap_details[h]['default_value']
                    heatmap_data[
                        "new_gene_mRNA_NTP_fraction_heatmap"][
                        "std_dev"][ntp_id] = np.zeros(
                        (len(new_gene_mRNA_ids),
                            len(new_gene_translation_efficiency_values),
                            len(new_gene_expression_factors))
                        ) + heatmap_details[h]['default_value']
            else:
                heatmap_data[h] = {}
                heatmap_data[h]["mean"] = np.zeros((
                    len(new_gene_mRNA_ids),
                    len(new_gene_translation_efficiency_values),
                    len(new_gene_expression_factors))
                    ) + heatmap_details[h]['default_value']
                heatmap_data[h]["std_dev"] = np.zeros((
                    len(new_gene_mRNA_ids),
                    len(new_gene_translation_efficiency_values),
                    len(new_gene_expression_factors))
                ) + heatmap_details[h]['default_value']

    # Data extraction
    print("---Data Extraction---")
    for h in heatmaps_to_make:
        h_details = heatmap_details[h]
        mean_matrix, std_matrix = data_to_mapping(conn, variant_metadata, history_sql,
            h_details["columns"], h_details["projections"],
            h_details["remove_first"], None, h_details["order_results"],
            h_details["custom_sql"], h_details["post_func"],
            h_details["num_digits_rounding"], h_details["default_value"])
        heatmap_data[h]["mean"] = mean_matrix
        heatmap_data[h]["std_dev"] = std_matrix

    # Plotting
    print("---Plotting---")
    plot_suffix = "_gens_" + str(MIN_CELL_INDEX) + "_through_" + str(MAX_CELL_INDEX)
    heatmap_x_label = "Expression Variant"
    heatmap_y_label = "Translation Efficiency Value"
    figsize_x =  2 + 2*len(new_gene_expression_factors)/3
    figsize_y = 2*len(new_gene_translation_efficiency_values)/2

    # Create dashboard plot
    if DASHBOARD_FLAG == 1 or DASHBOARD_FLAG == 2:
        self.plot_heatmaps(
            True, variant_mask, heatmap_x_label, heatmap_y_label,
            new_gene_expression_factors,
            new_gene_translation_efficiency_values, 'mean', figsize_x,
            figsize_y, plotOutDir, plot_suffix)

        if STD_DEV_FLAG:
            self.plot_heatmaps(
                True, variant_mask, heatmap_x_label, heatmap_y_label,
                new_gene_expression_factors,
                new_gene_translation_efficiency_values, 'std_dev',
                figsize_x, figsize_y, plotOutDir, plot_suffix)

    # Create separate plots
    if DASHBOARD_FLAG == 0 or DASHBOARD_FLAG == 2:
        self.plot_heatmaps(
            False, variant_mask, heatmap_x_label,heatmap_y_label,
            new_gene_expression_factors,
            new_gene_translation_efficiency_values, 'mean', figsize_x,
            figsize_y, plotOutDir, plot_suffix)

        if STD_DEV_FLAG:
            self.plot_heatmaps(
                False, variant_mask, heatmap_x_label, heatmap_y_label,
                new_gene_expression_factors,
                new_gene_translation_efficiency_values, 'std_dev',
                figsize_x, figsize_y, plotOutDir, plot_suffix)
