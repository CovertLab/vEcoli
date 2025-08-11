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

import itertools

from duckdb import DuckDBPyConnection
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
import math
import polars as pl
from unum.units import fg
from typing import Any, Callable, cast, Optional, TYPE_CHECKING
from tqdm import tqdm

from ecoli.variants.new_gene_internal_shift import get_new_gene_ids_and_indices
from ecoli.library.parquet_emitter import (
    field_metadata,
    ndlist_to_ndarray,
    open_arbitrary_sim_data,
    read_stacked_columns,
)
from ecoli.library.schema import bulk_name_to_idx
from wholecell.utils.plotting_tools import export_figure, heatmap
from wholecell.utils import units

import pickle

if TYPE_CHECKING:
    from reconstruction.ecoli.fit_sim_data_1 import SimulationDataEcoli

FONT_SIZE = 9

"""
Dashboard Flag

- 0: Separate Only (Each plot is its own file)
- 1: Dashboard Only (One file with all plots)
- 2: Both Dashboard and Separate
"""
DASHBOARD_FLAG = 2

"""
Standard Deviations Flag

- True: Plot an additional copy of all plots with standard deviation displayed
  insted of the average
- False: Plot no additional plots
"""
STD_DEV_FLAG = True

"""
Count number of sims that reach this generation (remember index 7 
corresponds to generation 8)
"""
COUNT_INDEX = 32
# COUNT_INDEX = 2 ### TODO: revert back after developing plot locally

"""
Plot data from generations [MIN_CELL_INDEX, MAX_CELL_INDEX)
Note that early generations may not be representative of dynamics 
due to how they are initialized
"""
MIN_CELL_INDEX = 16
# # MIN_CELL_INDEX = 1 ### TODO: revert back after developing plot locally
# MIN_CELL_INDEX = 0
MAX_CELL_INDEX = 33

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
    "rnap_crowding_heatmap",
    "ribosome_crowding_heatmap",
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
    "weighted_avg_translation_efficiency_heatmap",
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

capacity_gene_monomer_ids = ["EG10544-MONOMER[m]"]
# capacity_gene_monomer_id = ["EG11036-MONOMER[c]"]


def get_mean_and_std_matrices(
    conn: DuckDBPyConnection,
    variant_mapping: dict[int, tuple[int, int]],
    variant_matrix_shape: tuple[int, int],
    history_sql: str,
    columns: list[str],
    remove_first: bool = False,
    func: Optional[Callable] = None,
    order_results: bool = False,
    success_sql: Optional[str] = None,
    custom_sql: Optional[str] = None,
    post_func: Optional[Callable] = None,
    num_digits_rounding: Optional[int] = None,
    default_value: Optional[Any] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads one or more columns and calculates mean and std. dev. for each
    variant. If no custom SQL query is provided, this defaults to averaging
    per cell, then calculating the averages and standard deviations of all
    cells per variant.

    Args:
        conn: DuckDB connection
        variant_mapping: Mapping of variant IDs to row and column in matrix
            of new gene translation efficiency and expression factor variants
        variant_matrix_shape: Number of rows and columns in variant matrix
        history_sql: SQL subquery from :py:func:`ecoli.library.parquet_emitter.dataset_sql`
        columns, remove_first, func, order_results, success_sql: See
            :py:func:`ecoli.library.parquet_emitter.read_stacked_columns`
        custom_sql: SQL string containing a placeholder with name ``subquery``
            where the result of read_stacked_columns will be placed. Final query
            result must only have two columns in order: ``variant`` and a value
            for each variant. If not provided, defaults to average of averages
        post_func: Function that is called on Polars DataFrame resulting from query.
            Should return a Polars DataFrame with exactly three columns: ``variant``
            for the variant IDs, ``mean`` for some mean aggregate value (can be
            N-D list column), and ``std`` for some standard deviation aggregate.
        num_digits_rounding: Number of decimal places to round to
        default_value: Default value to put in output variant matrices if
            variant ID not included in query result (e.g. if variant failed in
            first generation and had no completed sims)
        new_gene_NTP_fraction: Set to True for NTP fraction heatmap so query output
            is properly handled

    Returns:
        Tuple of Numpy matrices with first two dimensions ``variant_matrix_shape``.
        Each cell in first matrix has the mean for that variant. Each cell in the
        second matrix has the std. dev. for that variant. These values can be Numpy
        arrays instead of scalar values (e.g. when calculating aggregates for many
        genes at once), in which case the matrices have shapes
        ``variant_matrix_shape + (num_genes,)``
    """
    subquery = read_stacked_columns(
        history_sql=history_sql,
        columns=columns,
        remove_first=remove_first,
        func=func,
        order_results=order_results,
        success_sql=success_sql,
    )
    if custom_sql is None:
        if len(columns) > 1:
            raise RuntimeError(
                "Must provide custom SQL expression to handle multiple columns at once."
            )
        custom_sql = f"""
            WITH avg_per_cell AS (
                SELECT avg({columns[0]}) AS avg_col, experiment_id, variant
                FROM ({{subquery}})
                GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
            )
            SELECT variant, avg(avg_col) AS mean, stddev(avg_col) AS std
            FROM avg_per_cell
            GROUP BY experiment_id, variant
            """
    if post_func is None:
        data = conn.sql(custom_sql.format(subquery=subquery)).pl()
    else:
        data = conn.sql(custom_sql.format(subquery=subquery)).pl()
        data = post_func(data)
    if set(data.columns) != {"variant", "mean", "std"}:
        raise RuntimeError(
            "post_func should return a Polars DataFrame with "
            "exactly three columns named `variant`, `mean`, and `std`"
        )
    data = [(i["variant"], i["mean"], i["std"]) for i in data.rows(named=True)]  # type: ignore[assignment]
    mean_matrix = [
        [default_value for _ in range(variant_matrix_shape[1])]
        for _ in range(variant_matrix_shape[0])
    ]
    std_matrix = [
        [default_value for _ in range(variant_matrix_shape[1])]
        for _ in range(variant_matrix_shape[0])
    ]
    for variant, mean, std in data:
        variant_row, variant_col = variant_mapping[variant]
        if mean is not None:
            if num_digits_rounding is not None:
                mean = np.round(mean, num_digits_rounding)
            mean_matrix[variant_row][variant_col] = mean
        if std is not None:
            if num_digits_rounding is not None:
                std = np.round(std, num_digits_rounding)
            std_matrix[variant_row][variant_col] = std
    return np.array(mean_matrix), np.array(std_matrix)


def get_mRNA_ids_from_monomer_ids(
    sim_data: "SimulationDataEcoli", target_monomer_ids: list[str]
) -> list[list[str]]:
    """
    Map monomer IDs back to the mRNA IDs that they were translated from.

    Args:
        target_monomer_ids: IDs of the monomers to map to mRNA IDs

    Returns:
        List of mRNA ID lists, one for each monomer ID
    """
    # Map protein ids to cistron ids
    monomer_ids = sim_data.process.translation.monomer_data["id"]
    cistron_ids = sim_data.process.translation.monomer_data["cistron_id"]
    monomer_to_cistron_id_dict = {
        monomer_id: cistron_id
        for monomer_id, cistron_id in zip(monomer_ids, cistron_ids)
    }
    target_cistron_ids = [
        monomer_to_cistron_id_dict.get(monomer_id) for monomer_id in target_monomer_ids
    ]
    RNA_ids = sim_data.process.transcription.rna_data["id"]
    target_RNA_ids = []
    # Map cistron ids to RNA ids
    for RNAP_cistron_id in target_cistron_ids:
        target_RNA_idx = sim_data.process.transcription.cistron_id_to_rna_indexes(
            RNAP_cistron_id
        )
        target_RNA_ids.append(RNA_ids[target_RNA_idx].tolist())
    return target_RNA_ids


def get_indexes(
    conn: DuckDBPyConnection,
    config_sql: str,
    index_type: str,
    ids: list[str] | list[list[str]],
) -> list[int | None] | list[list[int | None]]:
    """
    Retrieve DuckDB indices of a given type for a set of IDs. Note that
    DuckDB lists are 1-indexed.

    Args:
        conn: DuckDB database connection
        config_sql: DuckDB SQL query for sim config data (see
            :py:func:`~ecoli.library.parquet_emitter.dataset_sql`)
        index_type: Type of indices to return (one of ``cistron``,
            ``RNA``, ``mRNA``, or ``monomer``)
        ids: List of IDs to get indices for (must be monomer IDs
            if ``index_type`` is ``monomer``, else mRNA IDs)

    Returns:
        List of requested indexes
    """
    if index_type == "cistron":
        # Extract cistron indexes for each new gene
        cistron_idx_dict = {
            cis: i + 1
            for i, cis in enumerate(
                field_metadata(
                    conn, config_sql, "listeners__rnap_data__rna_init_event_per_cistron"
                )
            )
        }
        return [cistron_idx_dict.get(cistron) for cistron in ids]
    elif index_type == "RNA":
        # Extract RNA indexes for each new gene
        RNA_idx_dict = {
            rna: i + 1
            for i, rna in enumerate(
                field_metadata(
                    conn, config_sql, "listeners__rna_synth_prob__target_rna_synth_prob"
                )
            )
        }
        return [[RNA_idx_dict.get(rna_id) for rna_id in rna_ids] for rna_ids in ids]
    elif index_type == "mRNA":
        # Extract mRNA indexes for each new gene
        mRNA_idx_dict = {
            rna: i + 1
            for i, rna in enumerate(
                field_metadata(conn, config_sql, "listeners__rna_counts__mRNA_counts")
            )
        }
        return [[mRNA_idx_dict.get(rna_id) for rna_id in rna_ids] for rna_ids in ids]
    elif index_type == "monomer":
        # Extract protein indexes for each new gene
        monomer_idx_dict = {
            monomer: i + 1
            for i, monomer in enumerate(
                field_metadata(conn, config_sql, "listeners__monomer_counts")
            )
        }
        return [monomer_idx_dict.get(monomer_id) for monomer_id in ids]
    else:
        raise Exception(
            "Index type " + index_type + " has no instructions for data extraction."
        )


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
        SELECT log10(avg(avg_count) + 1) AS avg_count,
            log10(stddev(avg_count) + 1) AS std_count,
            experiment_id, variant, gene_idx
        FROM avg_per_cell
        GROUP BY experiment_id, variant, gene_idx
    )
    SELECT variant, list(avg_count ORDER BY gene_idx) AS mean,
        list(std_count ORDER BY gene_idx) AS std,
    FROM avg_per_variant
    GROUP BY experiment_id, variant
    """
"""
Generic SQL query for calculating average of a 1D-array column
per cell, aggregates that per variant into ``log10(mean + 1)`` and
``log10(std + 1)`` columns.
"""


def get_gene_mass_prod_func(
    sim_data: "SimulationDataEcoli",
    index_type: str,
    gene_ids: list[str] | list[list[str]],
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """
    Create a function to be passed as the ``post_func`` argument to
    :py:func:`~.get_mean_and_std_matrices` which multiplies the
    average and standard deviation 1D array columns by the mass of
    the gene ID for each element.

    Args:
        sim_data: Simulation data
        index_type: Either ``mRNA`` or ``monomer``. If ``mRNA``,
            ``gene_ids`` is list of lists of mRNA IDs, where inner
            lists correspond to mRNAs for each gene. Therefore,
            we sum the masses for the mRNAs of each inner list
            and multiply the input mean and std by this sum per gene.
        gene_ids: IDs of genes in the order they appear in the
            1D arrays of the query result
    """
    # Get mass for gene
    if index_type == "monomer":
        gene_masses = np.array(
            [
                (
                    sim_data.getter.get_mass(gene_id) / sim_data.constants.n_avogadro
                ).asNumber(fg)
                for gene_id in gene_ids
            ]
        )
    elif index_type == "mRNA":
        gene_masses = np.array(
            [
                np.sum(
                    [
                        (
                            sim_data.getter.get_mass(gene_id)
                            / sim_data.constants.n_avogadro
                        ).asNumber(fg)
                        for gene_id in one_gene_ids
                    ]
                )
                for one_gene_ids in gene_ids
            ]
        )
    else:
        raise RuntimeError("index_type must be monomer or mRNA.")

    def gene_mass_prod(variant_agg):
        avg_arr = ndlist_to_ndarray(variant_agg["mean"])
        std_arr = ndlist_to_ndarray(variant_agg["std"])
        return pl.DataFrame(
            {
                "variant": variant_agg["variant"],
                "mean": pl.Series(avg_arr * gene_masses),
                "std": pl.Series(std_arr * gene_masses),
            }
        )

    return gene_mass_prod


def get_gene_count_fraction_sql(
    gene_indices: list[int] | list[list[int]], column: str, index_type: str
) -> str:
    """
    Construct generic SQL query that gets the average per cell of a select
    set of indices from a 1D list column divided by the total of all elements
    per row of that list column, and aggregates those ratios per variant
    into mean and std columns.

    Args:
        gene_indices: Indices to extract from 1D list column to get ratios for
        column: Name of 1D list column
        index_type: Can either be ``monomer`` or ``mRNA``. For ``monomer``,
            function works exactly as described above. For ``mRNA``,
            ``gene_indices`` will be a list of lists of mRNA indices. This is
            because one gene can have to multiple mRNAs (transcription units).
            Therefore, we sum the elements corresponding to each gene before
            proceeding (see :py:func:`~.get_rnas_combined_as_genes_projection`).
    """
    if index_type == "monomer":
        list_to_unnest = f"list_select({column}, {gene_indices})"
    else:
        list_to_unnest = (
            "["
            + ", ".join(
                [
                    f"list_sum(list_select({column}, {idx_for_one_gene}))"
                    for idx_for_one_gene in gene_indices
                ]
            )
            + "]"
        )
    return f"""
        WITH list_counts AS (
            SELECT {list_to_unnest} AS selected_counts, list_sum({column})
                AS total_counts, experiment_id, variant, lineage_seed,
                generation, agent_id
            FROM ({{subquery}})
        ),
        unnested_fracs AS (
            SELECT unnest(selected_counts) / total_counts AS gene_fracs,
                generate_subscripts(selected_counts, 1) AS gene_idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM list_counts
        ),
        avg_per_cell AS (
            SELECT avg(gene_fracs) AS avg_frac,
                experiment_id, variant, gene_idx
            FROM unnested_fracs
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
        ),
        avg_per_variant AS (
            SELECT experiment_id, variant, avg(avg_frac)
                AS avg_frac, stddev(avg_frac) AS std_frac, gene_idx
            FROM avg_per_cell
            GROUP BY experiment_id, variant, gene_idx
        )
        SELECT variant, list(avg_frac ORDER BY gene_idx) AS mean,
            list(std_frac ORDER BY gene_idx) AS std,
        FROM avg_per_variant
        GROUP BY experiment_id, variant
        """


def get_new_gene_mRNA_NTP_fraction_sql(
    sim_data: "SimulationDataEcoli",
    new_gene_mRNA_idx: list[list[int]],
    ntp_ids: list[str],
) -> str:
    """
    Construct SQL query that gets, for each NTP, the fraction used by
    the mRNAs of each new gene, averages that per cell, and aggregate
    those fractions per variant into mean and std columns where each
    row is a 2D list with shape ``(# NTPs, # new genes)``.

    Args:
        sim_data: Simulation data
        new_gene_mRNA_idx: List of lists of mRNA indices for each new gene
        ntp_ids: IDs for NTPs in same order that they appear in
            ``sim_data.process.transcription.rna_data["counts_ACGU"]``
    """
    # Determine number of NTPs per new gene mRNA and for all mRNAs
    all_gene_mRNA_ACGU = sim_data.process.transcription.rna_data[
        "counts_ACGU"
    ].asNumber()[sim_data.process.transcription.rna_data["is_mRNA"]]

    # Strip location tags from NTP IDs so they can be used in SQL
    # identifiers without quoting
    ntp_ids = [ntp[:-3] for ntp in ntp_ids]

    # DuckDB list comprehension to calculate # of each NTP used by each mRNA
    ntp_projections = ", ".join(
        f"""
        [count_ntp[1] * count_ntp[2] for count_ntp in
            list_zip(listeners__rna_counts__mRNA_counts,
                {all_gene_mRNA_ACGU[:, ntp_idx].tolist()})] AS count_{ntp_id}
        """
        for ntp_idx, ntp_id in enumerate(ntp_ids)
    )
    # For each NTP, calculate fraction used in mRNAs for each new gene
    ntp_frac_projections = ", ".join(
        "["
        + ", ".join(
            f"list_sum(list_select(count_{ntp_id}, {one_gene_idx})) / "
            f"list_sum(count_{ntp_id})"
            for one_gene_idx in new_gene_mRNA_idx
        )
        + f"] AS frac_{ntp_id}"
        for ntp_id in ntp_ids
    )
    # Unnest average NTP fraction per mRNA
    unnested_frac_projections = (
        ", ".join(
            [f"unnest(frac_{ntp_id}) AS unnested_{ntp_id}_frac" for ntp_id in ntp_ids]
        )
        + f", generate_subscripts(frac_{ntp_ids[0]}, 1) AS gene_idx"
    )
    # Average NTP fraction per new gene first per cell, then per variant
    cell_avg_frac_projections = ", ".join(
        [f"avg(unnested_{ntp_id}_frac) AS avg_{ntp_id}_frac" for ntp_id in ntp_ids]
    )
    variant_avg_frac_projections = ", ".join(
        [
            f"avg(avg_{ntp_id}_frac) AS avg_{ntp_id}_frac, "
            f"stddev(avg_{ntp_id}_frac) AS std_{ntp_id}_frac"
            for ntp_id in ntp_ids
        ]
    )
    # Compile new gene average NTP fractions into 2D list columns
    # with shape (# of NTPs, # of new genes)
    list_avg_frac_projections = (
        "["
        + ", ".join(
            [f"list(avg_{ntp_id}_frac ORDER BY gene_idx)" for ntp_id in ntp_ids]
        )
        + "] AS mean"
    )
    list_std_frac_projections = (
        "["
        + ", ".join(
            [f"list(std_{ntp_id}_frac ORDER BY gene_idx)" for ntp_id in ntp_ids]
        )
        + "] AS std"
    )

    return f"""
        WITH ntp_counts AS (
            SELECT {ntp_projections}, experiment_id, variant, lineage_seed,
                generation, agent_id
            FROM ({{subquery}})
        ),
        ntp_frac AS (
            SELECT {ntp_frac_projections}, experiment_id, variant, lineage_seed,
                generation, agent_id
            FROM ntp_counts
        ),
        unnested_frac AS (
            SELECT {unnested_frac_projections}, experiment_id, variant, lineage_seed,
                generation, agent_id
            FROM ntp_frac
        ),
        avg_per_cell AS (
            SELECT {cell_avg_frac_projections},
                experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
            FROM unnested_frac
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, gene_idx
        ),
        avg_per_variant AS (
            SELECT experiment_id, variant, gene_idx, {variant_avg_frac_projections}
            FROM avg_per_cell
            GROUP BY experiment_id, variant, gene_idx
        )
        SELECT variant, {list_avg_frac_projections}, {list_std_frac_projections}
        FROM avg_per_variant
        GROUP BY experiment_id, variant
        """


def avg_ratio_of_1d_arrays_sql(numerator: str, denominator: str) -> str:
    """
    Create generic SQL query that calculates the average per cell of each
    element in two 1D list columns divided elementwise and aggregates those
    ratios per variant into mean and std columns.

    .. note::
        Time steps with 0 in the denominator are assigned a ratio of 0.

    Args:
        numerator: Name of 1D list column that will be numerator in ratio
        denominator: Name of 1D list column that will be denominator in ratio
    """
    return f"""
        WITH unnested_data AS (
            SELECT unnest({denominator}) AS denominator,
                unnest({numerator}) AS numerator,
                generate_subscripts({numerator}, 1) AS list_idx,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
        ),
        ratio_avg_per_cell AS (
            SELECT avg(
                CASE
                    WHEN denominator = 0 THEN 0
                    ELSE numerator / denominator
                END
            ) AS ratio_avg, experiment_id, variant, list_idx
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


def avg_1d_array_sql(column: str) -> str:
    """
    Create generic SQL query that calculates the average per cell of
    each element in a 1D array column and aggregates that per variant
    into mean and std columns.

    Args:
        column: Name of 1D list column to aggregate
    """
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
                generation, agent_id, array_idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, array_idx
        ),
        avg_per_variant AS (
            SELECT avg(avg_array_col) AS avg_array_col,
                stddev(avg_array_col) AS std_array_col,
                experiment_id, variant, array_idx
            FROM avg_per_cell
            GROUP BY experiment_id, variant, array_idx
        )
        SELECT variant, list(avg_array_col ORDER BY array_idx) AS mean,
            list(std_array_col ORDER BY array_idx) AS std
        FROM avg_per_variant
        GROUP BY experiment_id, variant
        """


def avg_sum_1d_array_sql(column: str) -> str:
    """
    Create generic SQL query that calculates the average per cell of
    the sum of elements in a 1D array column and
    aggregates that per variant into mean and std columns.

    Args:
        column: Name of 1D list column to aggregate
    """
    return f"""
        WITH avg_per_cell AS (
            SELECT avg(list_sum({column})) AS avg_sum,
                experiment_id, variant, lineage_seed,
                generation, agent_id
            FROM ({{subquery}})
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id
        )
        SELECT variant, avg(avg_sum) AS mean,
            stddev(avg_sum) AS std
        FROM avg_per_cell
        GROUP BY experiment_id, variant
        """


def avg_1d_array_over_scalar_sql(array_column: str, scalar_column: str) -> str:
    """
    Create generic SQL query that calculates the average per cell of
    each element in a 1D array column divided by a scalar column, and
    aggregates those ratios per variant into mean and std columns.

    .. note::
        Time steps with 0 in the scalar column are assigned a ratio of 0.

    Args:
        array_column: Name of 1D list column to aggregate
        scalar_column: Name of scalar column to divide ``array_column``
            cell averages by
    """
    return f"""
        WITH unnested_counts AS (
            SELECT unnest({array_column}) AS array_col,
                generate_subscripts({array_column}, 1) AS array_idx,
                {scalar_column} AS scalar_col,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
        ),
        avg_per_cell AS (
            SELECT avg(
                CASE
                    WHEN scalar_col = 0 THEN 0
                    ELSE array_col / scalar_col
                END) AS avg_ratio,
                experiment_id, variant, lineage_seed,
                generation, agent_id, array_idx
            FROM unnested_counts
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id, array_idx
        ),
        avg_per_variant AS (
            SELECT avg(avg_ratio) AS avg_ratio, stddev(avg_ratio) AS std_ratio,
                experiment_id, variant, array_idx
            FROM avg_per_cell
            GROUP BY experiment_id, variant, array_idx
        )
        SELECT variant, list(avg_ratio ORDER BY array_idx) AS mean,
            list(std_ratio ORDER BY array_idx) AS std
        FROM avg_per_variant
        GROUP BY experiment_id, variant
        """


def avg_sum_1d_array_over_scalar_sql(array_column: str, scalar_column: str) -> str:
    """
    Create generic SQL query that calculates the average per cell of
    the sum of elements in a 1D array column divided by a scalar column, and
    aggregates those ratios per variant as mean and std columns.

    .. note::
        Time steps with 0 in the scalar column are assigned a ratio of 0.

    Args:
        array_column: Name of 1D list column to aggregate
        scalar_column: Name of scalar column to divide ``array_column``
            cell averages by
    """
    return f"""
        WITH avg_per_cell AS (
            SELECT avg(
                CASE
                    WHEN {scalar_column} = 0 THEN 0
                    ELSE list_sum({array_column}) / {scalar_column}
                END) AS avg_ratio,
                experiment_id, variant, lineage_seed,
                generation, agent_id
            FROM ({{subquery}})
            GROUP BY experiment_id, variant, lineage_seed,
                generation, agent_id
        )
        SELECT variant, avg(avg_ratio) AS mean, stddev(avg_ratio) AS std
        FROM avg_per_cell
        GROUP BY experiment_id, variant
        """


def get_rnap_counts_projection(
    sim_data: "SimulationDataEcoli", bulk_ids: list[str]
) -> str:
    """
    Return SQL projection to selectively read bulk inactive RNAP count.

    Args:
        sim_data: Simulation data
        bulk_ids: List of all bulk IDs in order
    """
    rnap_idx = bulk_name_to_idx(sim_data.molecule_ids.full_RNAP, bulk_ids)
    return f"bulk[{rnap_idx + 1}] AS bulk"


def get_ribosome_counts_projection(
    sim_data: "SimulationDataEcoli", bulk_ids: list[str]
) -> str:
    """
    Return SQL projection to selectively read bulk inactive ribosome count
    (defined as minimum of free 30S and 50S subunits at any given moment)

    Args:
        sim_data: Simulation data
        bulk_ids: List of all bulk IDs in order
    """
    ribosome_idx = cast(
        np.ndarray,
        bulk_name_to_idx(
            [
                sim_data.molecule_ids.s30_full_complex,
                sim_data.molecule_ids.s50_full_complex,
            ],
            bulk_ids,
        ),
    )
    return f"least(bulk[{ribosome_idx[0] + 1}], bulk[{ribosome_idx[1] + 1}]) AS bulk"


def get_overcrowding_sql(target_col: str, actual_col: str) -> str:
    """
    Create generic SQL query that calculates for average number of genes
    that are overcrowded per time step for each cell, then aggregates that
    per variant into mean and std columns.

    At every time step, if the element in ``target_col`` is greater than the
    corresponding element in ``actual_col``, we say that the gene for that
    element is overcrowded. We average the number of overcrowded genes over
    all the time steps for each cell. Then, we average the per-cell averages
    over all cells in each variant.

    Args:
        target_col: Name of 1D list column with target values
        actual_col: Name of 1D list column with actual values.
    """
    return f"""
        WITH avg_per_cell AS (
            SELECT avg(list_sum([(i[1] > i[2])::UTINYINT
                for i in list_zip({target_col}, {actual_col})])) AS overcrowded,
                experiment_id, variant, lineage_seed, generation, agent_id
            FROM ({{subquery}})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
        )
        SELECT variant, avg(overcrowded) AS mean, stddev(overcrowded) AS std,
        FROM avg_per_cell
        GROUP BY experiment_id, variant
        """


def get_rnas_combined_as_genes_projection(
    column: str, rna_idx: list[list[int]], name: str, cast_type: Optional[str] = None
):
    """
    Create generic SQL projection that evaluates to a list column
    where each element is the sum of a subset of elements from the
    original list column. This is mainly used to sum up all RNA
    data that corresponds to a single gene / cistron / monomer.
    """
    if cast_type is not None:
        cast_type = f"::{cast_type}"
    else:
        cast_type = ""
    sum_projections = ", ".join(
        [
            f"list_sum(list_select({column}, {rna_idx_for_one_gene}){cast_type})"
            for rna_idx_for_one_gene in rna_idx
        ]
    )
    return f"[{sum_projections}] AS {name}"


def get_variant_mask(
    conn: DuckDBPyConnection,
    config_sql: str,
    variant_to_row_col: dict[int, tuple[int, int]],
    variant_matrix_shape: tuple[int, int],
) -> npt.NDArray[np.bool_]:
    """
    Get a boolean matrix where the rows represent the different
    translation efficiencies and the columns represent the different
    expression factors that were used to create variants. The matrix
    is True for each combination that was actually simulated and
    False otherwise.
    """
    variants = conn.sql(
        f"SELECT DISTINCT variant AS variant FROM ({config_sql})"
    ).fetchnumpy()["variant"]
    variant_mask = np.zeros(variant_matrix_shape, np.bool_)
    for variant in variants:
        variant_mask[*variant_to_row_col[variant]] = True
    return variant_mask


def plot_heatmaps(
    heatmap_data,
    heatmap_details,
    new_gene_cistron_ids,
    ntp_ids,
    capacity_gene_common_names,
    total_heatmaps_to_make,
    is_dashboard,
    variant_mask,
    heatmap_x_label,
    heatmap_y_label,
    new_gene_expression_factors,
    new_gene_translation_efficiency_values,
    summary_statistic,
    figsize_x,
    figsize_y,
    plotOutDir,
    plot_suffix,
):
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
        summary_statistic: Specifies whether average (mean) or
            standard deviation (std_dev) should be displayed on the
            heatmaps
        figsize_x: Horizontal size of each heatmap
        figsize_y: Vertical size of each heatmap
        plotOutDir: Output directory for plots
        plot_suffix: Suffix to add to plot file names, usually specifying
            which generations were plotted
    """
    if summary_statistic == "std_dev":
        plot_suffix = plot_suffix + "_std_dev"
    elif summary_statistic != "mean":
        raise Exception(
            "mean and std_dev are the only currently supported summary statistics"
        )

    if is_dashboard:
        # Determine dashboard layout
        if total_heatmaps_to_make > 3:
            dashboard_ncols = 4
            dashboard_nrows = math.ceil((total_heatmaps_to_make + 1) / dashboard_ncols)
        else:
            dashboard_ncols = total_heatmaps_to_make + 1
            dashboard_nrows = 1
        fig, axs = plt.subplots(
            nrows=dashboard_nrows,
            ncols=dashboard_ncols,
            figsize=(figsize_y * dashboard_ncols, figsize_x * dashboard_nrows),
            layout="constrained",
        )
        if dashboard_nrows == 1:
            axs = np.reshape(axs, (1, dashboard_ncols))
        row_ax = 0
        col_ax = 0

    for h in HEATMAPS_TO_MAKE_LIST:
        if not heatmap_details[h]["is_nonstandard_plot"]:
            stop_index = 1
            title_addition = ""
            filename_addition = ""
            if heatmap_details[h]["is_new_gene_heatmap"]:
                stop_index = len(new_gene_cistron_ids)
            elif heatmap_details[h]["is_capacity_gene_heatmap"]:
                stop_index = len(capacity_gene_common_names)
            for i in range(stop_index):
                if heatmap_details[h]["is_new_gene_heatmap"]:
                    title_addition = f": {new_gene_cistron_ids[i][:-4]}"
                    filename_addition = f"_{new_gene_cistron_ids[i][:-4]}"
                    curr_heatmap_data = heatmap_data[h][summary_statistic][:, :, i]
                elif heatmap_details[h]["is_capacity_gene_heatmap"]:
                    title_addition = f": {capacity_gene_common_names[i]}"
                    filename_addition = f"_{capacity_gene_common_names[i]}"
                    curr_heatmap_data = heatmap_data[h][summary_statistic][:, :, i]
                else:
                    curr_heatmap_data = heatmap_data[h][summary_statistic]
                title = heatmap_details[h]["plot_title"] + title_addition
                if summary_statistic == "std_dev":
                    title = f"Std Dev: {title}"
                if is_dashboard:
                    curr_ax = axs[row_ax][col_ax]
                else:
                    fig, curr_ax = plt.subplots(1, 1, figsize=(figsize_x, figsize_y))
                heatmap(
                    curr_ax,
                    variant_mask,
                    curr_heatmap_data,
                    heatmap_data["completed_gens_heatmap"]["mean"],
                    new_gene_expression_factors,
                    new_gene_translation_efficiency_values,
                    heatmap_x_label,
                    heatmap_y_label,
                    title,
                    heatmap_details[h]["box_text_size"],
                )
                if not is_dashboard:
                    fig.tight_layout()
                    export_figure(plt, plotOutDir, h + filename_addition + plot_suffix)
                    plt.close(fig)
                else:
                    col_ax += 1
                    if col_ax == dashboard_ncols:
                        col_ax = 0
                        row_ax += 1
        elif h == "new_gene_mRNA_NTP_fraction_heatmap":
            for cistron_idx, cistron_id in enumerate(new_gene_cistron_ids):
                for ntp_idx, ntp_id in enumerate(ntp_ids):
                    title = (
                        f"{heatmap_details[h]['plot_title']} {ntp_id[:-3]}"
                        f" Fraction: {cistron_id[:-4]}"
                    )
                    if summary_statistic == "std_dev":
                        title = f"Std Dev: {title}"
                    if is_dashboard:
                        curr_ax = axs[row_ax][col_ax]
                    else:
                        fig, curr_ax = plt.subplots(
                            1, 1, figsize=(figsize_x, figsize_y)
                        )
                    heatmap(
                        curr_ax,
                        variant_mask,
                        heatmap_data[h][summary_statistic][:, :, ntp_idx, cistron_idx],
                        heatmap_data["completed_gens_heatmap"]["mean"],
                        new_gene_expression_factors,
                        new_gene_translation_efficiency_values,
                        heatmap_x_label,
                        heatmap_y_label,
                        title,
                        heatmap_details[h]["box_text_size"],
                    )
                    if not is_dashboard:
                        fig.tight_layout()
                        export_figure(
                            plt,
                            plotOutDir,
                            f"new_gene_mRNA_{ntp_id[:-3]}_fraction_heatmap"
                            + filename_addition
                            + plot_suffix,
                        )
                        plt.close(fig)
                    else:
                        col_ax += 1
                        if col_ax == dashboard_ncols:
                            col_ax = 0
                            row_ax += 1
        else:
            raise Exception(
                f"Heatmap {h} is neither a standard plot nor a"
                f" nonstandard plot that has specific instructions for"
                f" plotting."
            )
    if is_dashboard:
        fig.tight_layout()
        export_figure(
            plt, plotOutDir, f"00_new_gene_exp_trl_eff_dashboard{plot_suffix}"
        )  ## TODO: Revert back after running new plots on Sherlock sims
        plt.close("all")


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
    Create either a single multi-heatmap plot or 1+ separate heatmaps of data
    for a grid of new gene variant simulations with varying expression and
    translation efficiencies.

    Params (override corresponding hard-coded global variables):
        font_size, dashboard_flag, std_dev_flag, count_index, min_cell_index,
        max_cell_index
    """
    # Extract plot parameters
    min_cell_index = params.get("min_cell_index", MIN_CELL_INDEX)
    max_cell_index = params.get("max_cell_index", MAX_CELL_INDEX)
    dashboard_flag = params.get("dashboard_flag", DASHBOARD_FLAG)
    std_dev_flag = params.get("std_dev_flag", STD_DEV_FLAG)
    count_index = params.get("count_index", COUNT_INDEX)

    # Filter to specified generation range
    history_sql = (
        f"FROM ({history_sql}) WHERE generation >= {min_cell_index}"
        f" AND generation < {max_cell_index}"
    )
    config_sql = (
        f"FROM ({config_sql}) WHERE generation >= {min_cell_index}"
        f" AND generation < {max_cell_index}"
    )
    # Define baseline variant (ID = 0) as 0 new gene expr. and trans. eff.
    experiment_id = next(iter(variant_metadata.keys()))
    variant_metadata = variant_metadata[experiment_id]
    variant_metadata[0] = {"exp_trl_eff": {"exp": 0, "trl_eff": 0}}

    with open_arbitrary_sim_data(sim_data_dict) as f:
        sim_data = pickle.load(f)

    # Determine new gene cistron and monomer ids
    (new_gene_cistron_ids, _, new_gene_monomer_ids, _) = get_new_gene_ids_and_indices(
        sim_data
    )

    # Assuming we ran a workflow with `n` new gene expression factors
    # and `m` new gene translation efficiency values, create an `n * m`
    # grid sorted in ascending order along both axes and calculate
    # mapping from variant numbers to row and column indices in grid
    assert "exp_trl_eff" in next(iter(variant_metadata.values())), (
        "This plot is intended to be run on simulations where the"
        " new gene expression-translation efficiency variant was "
        "enabled, but no parameters for this variant were found."
    )
    new_gene_expression_factors = sorted(
        set(
            variant_params["exp_trl_eff"]["exp"]
            for variant_params in variant_metadata.values()
        )
    )
    new_gene_translation_efficiency_values = sorted(
        set(
            variant_params["exp_trl_eff"]["trl_eff"]
            for variant_params in variant_metadata.values()
        )
    )
    variant_matrix_shape = (
        len(new_gene_translation_efficiency_values),
        len(new_gene_expression_factors),
    )
    variant_to_row_col = {
        variant: (
            new_gene_translation_efficiency_values.index(
                variant_params["exp_trl_eff"]["trl_eff"]
            ),
            new_gene_expression_factors.index(variant_params["exp_trl_eff"]["exp"]),
        )
        for variant, variant_params in variant_metadata.items()
    }
    variant_mask = get_variant_mask(
        conn, config_sql, variant_to_row_col, variant_matrix_shape
    )

    bulk_ids = field_metadata(conn, config_sql, "bulk")
    ntp_ids = list(sim_data.ntp_code_to_id_ordered.values())
    # Get indices for data extraction
    rnap_subunit_mRNA_ids = get_mRNA_ids_from_monomer_ids(
        sim_data, sim_data.molecule_groups.RNAP_subunits
    )
    rnap_subunit_mRNA_indexes = list(
        set(
            itertools.chain.from_iterable(
                cast(
                    list[list[int]],
                    get_indexes(conn, config_sql, "mRNA", rnap_subunit_mRNA_ids),
                )
            )
        )
    )
    rnap_subunit_monomer_indexes = get_indexes(
        conn, config_sql, "monomer", sim_data.molecule_groups.RNAP_subunits
    )
    ribosomal_mRNA_ids = get_mRNA_ids_from_monomer_ids(
        sim_data, sim_data.molecule_groups.ribosomal_proteins
    )
    ribosomal_mRNA_indexes = list(
        set(
            itertools.chain.from_iterable(
                cast(
                    list[list[int]],
                    get_indexes(conn, config_sql, "mRNA", ribosomal_mRNA_ids),
                )
            )
        )
    )
    ribosomal_monomer_indexes = get_indexes(
        conn, config_sql, "monomer", sim_data.molecule_groups.ribosomal_proteins
    )
    cistron_ids = field_metadata(
        conn, config_sql, "listeners__rna_counts__full_mRNA_cistron_counts"
    )
    capacity_gene_mRNA_ids = get_mRNA_ids_from_monomer_ids(
        sim_data, capacity_gene_monomer_ids
    )
    capacity_gene_mRNA_indexes = cast(
        list[list[int]], get_indexes(conn, config_sql, "mRNA", capacity_gene_mRNA_ids)
    )
    capacity_gene_monomer_indexes = cast(
        list[int], get_indexes(conn, config_sql, "monomer", capacity_gene_monomer_ids)
    )
    capacity_gene_common_names = [
        sim_data.common_names.get_common_name(monomer_id[:-3])
        for monomer_id in capacity_gene_monomer_ids
    ]
    # Convert cistron IDs to RNA IDs
    RNA_ids = sim_data.process.transcription.rna_data["id"]
    new_gene_mRNA_ids = []
    for cistron_id in new_gene_cistron_ids:
        target_RNA_idx = sim_data.process.transcription.cistron_id_to_rna_indexes(
            cistron_id
        )
        new_gene_mRNA_ids.append(RNA_ids[target_RNA_idx])
    new_gene_mRNA_indexes = cast(
        list[list[int]], get_indexes(conn, config_sql, "mRNA", new_gene_mRNA_ids)
    )
    new_gene_monomer_indexes = cast(
        list[int], get_indexes(conn, config_sql, "monomer", new_gene_monomer_ids)
    )
    new_gene_RNA_indexes = cast(
        list[list[int]], get_indexes(conn, config_sql, "RNA", new_gene_mRNA_ids)
    )
    new_gene_cistron_indexes = cast(
        list[int], get_indexes(conn, config_sql, "cistron", new_gene_cistron_ids)
    )

    # Determine glucose index in exchange fluxes
    external_molecule_ids = np.array(
        field_metadata(
            conn, config_sql, "listeners__fba_results__external_exchange_fluxes"
        )
    )
    if "GLC[p]" not in external_molecule_ids:
        print("This plot only runs when glucose is the carbon source.")
        return
    glucose_idx = np.where(external_molecule_ids == "GLC[p]")[0][0] + 1
    flux_scaling_factor = (
        sim_data.getter.get_mass("GLC[p]")
        * (units.mmol / units.g / units.h)
        * units.fg  # Flux * dry mass units
    ).asNumber()

    # Get normalized translation efficiency for all mRNAs
    trl_effs = sim_data.process.translation.translation_efficiencies_by_monomer
    trl_eff_ids = sim_data.process.translation.monomer_data["cistron_id"]
    mRNA_cistron_idx_dict = {rna: i for i, rna in enumerate(cistron_ids)}
    trl_eff_id_mapping = np.array([mRNA_cistron_idx_dict[id] for id in trl_eff_ids])
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
    # Specify unique fields and non-default values here
    heatmap_details: dict[str, dict] = {
        "completed_gens_heatmap": {
            "columns": ["time"],
            "custom_sql": f"""
                WITH max_gen_per_seed AS (
                    SELECT max(generation) AS max_generation, experiment_id, variant
                    FROM ({{subquery}})
                    GROUP BY experiment_id, variant, lineage_seed
                )
                -- Boolean values must be explicitly cast to numeric for aggregation
                SELECT variant, avg((max_generation = {count_index})::BIGINT) AS mean,
                    stddev((max_generation = {count_index})::BIGINT) AS std
                FROM max_gen_per_seed
                GROUP BY experiment_id, variant
                """,
            "plot_title": f"Percentage of Sims That Reached Generation {count_index}",
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
                SELECT variant, avg(doubling_time) AS mean, stddev(doubling_time) AS std
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
            "plot_title": "RNA Polymerase (RNAP) Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [
                get_rnap_counts_projection(sim_data, bulk_ids),
                "listeners__unique_molecule_counts__active_RNAP",
            ],
            "custom_sql": """
                WITH total_counts AS (
                    SELECT avg(bulk +
                        listeners__unique_molecule_counts__active_RNAP) AS rnap_counts,
                        experiment_id, variant
                    FROM ({subquery})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(rnap_counts) AS mean, stddev(rnap_counts) AS std
                FROM total_counts
                GROUP BY experiment_id, variant
                """,
        },
        "ribosome_counts_heatmap": {
            "plot_title": "Active Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [
                get_ribosome_counts_projection(sim_data, bulk_ids),
                "listeners__unique_molecule_counts__active_ribosome",
            ],
            "custom_sql": """
                WITH total_counts AS (
                    SELECT avg(bulk +
                        listeners__unique_molecule_counts__active_ribosome)
                        AS ribosome_counts,
                        experiment_id, variant
                    FROM ({subquery})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(ribosome_counts) AS mean,
                    stddev(ribosome_counts) AS std
                FROM total_counts
                GROUP BY experiment_id, variant
                """,
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
            "plot_title": "Free Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [get_ribosome_counts_projection(sim_data, bulk_ids)],
            "custom_sql": """
                WITH avg_per_cell AS (
                    SELECT avg(bulk) AS avg_col, experiment_id, variant
                    FROM ({subquery})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(avg_col) AS mean, stddev(avg_col) AS std
                FROM avg_per_cell
                GROUP BY experiment_id, variant
                """,
        },
        "free_rnap_counts_heatmap": {
            "plot_title": "Free RNA Polymerase (RNAP) Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [get_rnap_counts_projection(sim_data, bulk_ids)],
            "custom_sql": """
                WITH avg_per_cell AS (
                    SELECT avg(bulk) AS avg_col, experiment_id, variant
                    FROM ({subquery})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(avg_col) AS mean, stddev(avg_col) AS std
                FROM avg_per_cell
                GROUP BY experiment_id, variant
                """,
        },
        "rnap_ribosome_counts_ratio_heatmap": {
            "plot_title": "RNAP Counts / Ribosome Counts",
            "num_digits_rounding": 4,
            "box_text_size": "x-small",
            "columns": [
                get_rnap_counts_projection(sim_data, bulk_ids)
                + "_rnap, "
                + get_ribosome_counts_projection(sim_data, bulk_ids)
                + "_ribosome",
                "listeners__unique_molecule_counts__active_RNAP",
                "listeners__unique_molecule_counts__active_ribosome",
            ],
            "custom_sql": """
                WITH total_counts AS (
                    SELECT avg(bulk_ribosome +
                        listeners__unique_molecule_counts__active_ribosome)
                        AS ribosome_counts,
                        avg(bulk_rnap +
                        listeners__unique_molecule_counts__active_RNAP) AS rnap_counts,
                        experiment_id, variant
                    FROM ({subquery})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(rnap_counts / ribosome_counts) AS mean,
                    stddev(rnap_counts / ribosome_counts) AS std,
                FROM total_counts
                GROUP BY experiment_id, variant
                """,
        },
        "rnap_crowding_heatmap": {
            "columns": [
                "listeners__rna_synth_prob__target_rna_synth_prob",
                "listeners__rna_synth_prob__actual_rna_synth_prob",
            ],
            "plot_title": "RNAP Crowding: # of TUs",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "custom_sql": get_overcrowding_sql(
                "listeners__rna_synth_prob__target_rna_synth_prob",
                "listeners__rna_synth_prob__actual_rna_synth_prob",
            ),
        },
        "ribosome_crowding_heatmap": {
            "columns": [
                "listeners__ribosome_data__target_prob_translation_per_transcript",
                "listeners__ribosome_data__actual_prob_translation_per_transcript",
            ],
            "plot_title": "Ribosome Crowding: # of Monomers",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "custom_sql": get_overcrowding_sql(
                "listeners__ribosome_data__target_prob_translation_per_transcript",
                "listeners__ribosome_data__actual_prob_translation_per_transcript",
            ),
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
                        unnest({ordered_trl_effs.tolist()}) AS trl_eff,
                        experiment_id, variant, lineage_seed, generation, agent_id
                    FROM ({{subquery}})
                ),
                -- Materialize in memory so we can left join per-gene averages with
                -- sum of all gene averages without re-reading data
                avg_per_cell AS MATERIALIZED (
                    SELECT avg(array_col) AS avg_array_col,
                        experiment_id, variant, lineage_seed,
                        generation, agent_id, array_idx, any_value(trl_eff) AS trl_eff
                    FROM unnested_counts
                    GROUP BY experiment_id, variant, lineage_seed,
                        generation, agent_id, array_idx
                ),
                total_avg_per_cell AS (
                    SELECT sum(avg_array_col) AS sum_avg_array_col,
                        experiment_id, variant, lineage_seed,
                        generation, agent_id
                    FROM avg_per_cell
                    GROUP BY experiment_id, variant, lineage_seed,
                        generation, agent_id
                ),
                weighted_avg_per_cell AS (
                    SELECT sum(avg_array_col / sum_avg_array_col * trl_eff) AS
                        weighted_avg_array_col, experiment_id, variant
                    FROM avg_per_cell
                    LEFT JOIN total_avg_per_cell USING (experiment_id,
                        variant, lineage_seed, generation, agent_id)
                    GROUP BY experiment_id, variant, lineage_seed,
                        generation, agent_id
                )
                SELECT variant, avg(weighted_avg_array_col) AS mean,
                    stddev(weighted_avg_array_col) AS std
                FROM weighted_avg_per_cell
                GROUP BY experiment_id, variant
                """,
        },
        "capacity_gene_mRNA_counts_heatmap": {
            "plot_title": "Log(Capacity Gene mRNA Counts+1): ",
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_counts__mRNA_counts",
                    capacity_gene_mRNA_indexes,
                    "gene_counts",
                )
            ],
            "custom_sql": GENE_COUNTS_SQL,
        },
        "capacity_gene_monomer_counts_heatmap": {
            "plot_title": "Log(Capacity Gene Protein Counts+1): ",
            "columns": [
                "list_select(listeners__monomer_counts, "
                f"{capacity_gene_monomer_indexes}) AS gene_counts"
            ],
            "custom_sql": GENE_COUNTS_SQL,
        },
        "capacity_gene_mRNA_mass_fraction_heatmap": {
            "plot_title": "Capacity Gene mRNA Mass Fraction: ",
            "num_digits_rounding": 3,
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_counts__mRNA_counts",
                    capacity_gene_mRNA_indexes,
                    "gene_counts",
                ),
                "listeners__mass__mRna_mass AS mass",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql("gene_counts", "mass"),
            "post_func": get_gene_mass_prod_func(
                sim_data, "mRNA", capacity_gene_mRNA_ids
            ),
        },
        "capacity_gene_monomer_mass_fraction_heatmap": {
            "plot_title": "Capacity Gene Protein Mass Fraction: ",
            "num_digits_rounding": 3,
            "columns": [
                "list_select(listeners__monomer_counts, "
                f"{capacity_gene_monomer_indexes}) AS gene_counts",
                "listeners__mass__protein_mass AS mass",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql("gene_counts", "mass"),
            "post_func": get_gene_mass_prod_func(
                sim_data, "monomer", capacity_gene_monomer_ids
            ),
        },
        "capacity_gene_mRNA_counts_fraction_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts"],
            "plot_title": "Capacity Gene mRNA Counts Fraction: ",
            "num_digits_rounding": 3,
            "custom_sql": get_gene_count_fraction_sql(
                capacity_gene_mRNA_indexes,
                "listeners__rna_counts__mRNA_counts",
                "mRNA",
            ),
        },
        "capacity_gene_monomer_counts_fraction_heatmap": {
            "columns": ["listeners__monomer_counts"],
            "plot_title": "Capacity Gene Protein Counts Fraction: ",
            "num_digits_rounding": 3,
            "custom_sql": get_gene_count_fraction_sql(
                capacity_gene_monomer_indexes, "listeners__monomer_counts", "monomer"
            ),
        },
        "new_gene_copy_number_heatmap": {
            "plot_title": "New Gene Copy Number",
            "columns": [
                "list_select(listeners__rna_synth_prob__gene_copy_number, "
                f"{new_gene_cistron_indexes}) AS gene_counts"
            ],
            "num_digits_rounding": 3,
            "custom_sql": avg_1d_array_sql("gene_counts"),
        },
        "new_gene_mRNA_counts_heatmap": {
            "plot_title": "Log(New Gene mRNA Counts+1)",
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_counts__mRNA_counts",
                    new_gene_mRNA_indexes,
                    "gene_counts",
                )
            ],
            "custom_sql": GENE_COUNTS_SQL,
        },
        "new_gene_monomer_counts_heatmap": {
            "plot_title": "Log(New Gene Protein Counts+1)",
            "columns": [
                "list_select(listeners__monomer_counts, "
                f"{new_gene_monomer_indexes}) AS gene_counts"
            ],
            "custom_sql": GENE_COUNTS_SQL,
        },
        "new_gene_mRNA_mass_fraction_heatmap": {
            "plot_title": "New Gene mRNA Mass Fraction",
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_counts__mRNA_counts",
                    new_gene_mRNA_indexes,
                    "gene_counts",
                ),
                "listeners__mass__mRna_mass AS mass",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql("gene_counts", "mass"),
            "post_func": get_gene_mass_prod_func(sim_data, "mRNA", new_gene_mRNA_ids),
        },
        "new_gene_mRNA_counts_fraction_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts"],
            "plot_title": "New Gene mRNA Counts Fraction",
            "custom_sql": get_gene_count_fraction_sql(
                new_gene_mRNA_indexes, "listeners__rna_counts__mRNA_counts", "mRNA"
            ),
        },
        "new_gene_mRNA_NTP_fraction_heatmap": {
            "columns": ["listeners__rna_counts__mRNA_counts"],
            "plot_title": "New Gene",
            "num_digits_rounding": 4,
            "box_text_size": "x-small",
            "custom_sql": get_new_gene_mRNA_NTP_fraction_sql(
                sim_data, new_gene_mRNA_indexes, ntp_ids
            ),
            "is_nonstandard_plot": True,
        },
        "new_gene_monomer_mass_fraction_heatmap": {
            "plot_title": "New Gene Protein Mass Fraction",
            "columns": [
                "list_select(listeners__monomer_counts, "
                f"{new_gene_monomer_indexes}) AS gene_counts",
                "listeners__mass__protein_mass AS mass",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql("gene_counts", "mass"),
            "post_func": get_gene_mass_prod_func(
                sim_data, "monomer", new_gene_monomer_ids
            ),
        },
        "new_gene_monomer_counts_fraction_heatmap": {
            "columns": ["listeners__monomer_counts"],
            "plot_title": "New Gene Protein Counts Fraction",
            "custom_sql": get_gene_count_fraction_sql(
                new_gene_monomer_indexes, "listeners__monomer_counts", "monomer"
            ),
        },
        "new_gene_rnap_init_rate_heatmap": {
            "plot_title": "New Gene RNAP Initialization Rate",
            "columns": [
                "list_select(listeners__rna_synth_prob__gene_copy_number, "
                f"{new_gene_cistron_indexes}) AS gene_copy_number",
                "list_select(listeners__rnap_data__rna_init_event_per_cistron, "
                f"{new_gene_cistron_indexes}) AS rna_init_event_per_cistron",
            ],
            "custom_sql": avg_ratio_of_1d_arrays_sql(
                "rna_init_event_per_cistron", "gene_copy_number"
            ),
        },
        "new_gene_ribosome_init_rate_heatmap": {
            "plot_title": "New Gene Ribosome Initalization Rate",
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_counts__mRNA_counts",
                    new_gene_mRNA_indexes,
                    "mRNA_counts",
                ),
                "list_select(listeners__ribosome_data__ribosome_init_event_per_monomer, "
                f"{new_gene_monomer_indexes}) AS ribosome_init_event_per_monomer",
            ],
            "custom_sql": avg_ratio_of_1d_arrays_sql(
                "ribosome_init_event_per_monomer", "mRNA_counts"
            ),
        },
        "new_gene_rnap_time_overcrowded_heatmap": {
            "plot_title": "Fraction of Time RNAP Overcrowded New Gene",
            # Need to explicitly cast boolean list to numeric list for aggregation
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_synth_prob__tu_is_overcrowded",
                    new_gene_RNA_indexes,
                    "overcrowded",
                    "UTINYINT[]",
                )
            ],
            "custom_sql": avg_1d_array_sql("overcrowded"),
        },
        "new_gene_ribosome_time_overcrowded_heatmap": {
            "plot_title": "Fraction of Time Ribosome Overcrowded New Gene",
            # Need to explicitly cast boolean list to numeric list for aggregation
            "columns": [
                "list_select(listeners__ribosome_data__mRNA_is_overcrowded, "
                f"{new_gene_monomer_indexes})::UTINYINT[] AS overcrowded"
            ],
            "custom_sql": avg_1d_array_sql("overcrowded"),
        },
        "new_gene_actual_protein_init_prob_heatmap": {
            "plot_title": "New Gene Actual Protein Init Prob",
            "num_digits_rounding": 4,
            "columns": [
                "list_select(listeners__ribosome_data__actual_prob_translation_per_transcript, "
                f"{new_gene_monomer_indexes}) AS init_prob"
            ],
            "custom_sql": avg_1d_array_sql("init_prob"),
        },
        "new_gene_target_protein_init_prob_heatmap": {
            "plot_title": "New Gene Target Protein Init Prob",
            "num_digits_rounding": 4,
            "columns": [
                "list_select(listeners__ribosome_data__target_prob_translation_per_transcript, "
                f"{new_gene_monomer_indexes}) AS init_prob"
            ],
            "custom_sql": avg_1d_array_sql("init_prob"),
        },
        "new_gene_protein_init_prob_max_p_target_ratio_heatmap": {
            "plot_title": "New Gene Protein Max Prob / Target Prob Ratio",
            "num_digits_rounding": 4,
            "columns": [
                "list_select(listeners__ribosome_data__target_prob_translation_per_transcript, "
                f"{new_gene_monomer_indexes}) AS target_prob",
                "list_select(listeners__ribosome_data__max_p_per_protein, "
                f"{new_gene_monomer_indexes}) AS max_p",
            ],
            "custom_sql": avg_ratio_of_1d_arrays_sql("target_prob", "max_p"),
        },
        "new_gene_rna_synth_prob_max_p_target_ratio_heatmap": {
            "plot_title": "New Gene Protein Max Prob / Target Prob Ratio",
            "num_digits_rounding": 4,
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_synth_prob__target_rna_synth_prob",
                    new_gene_RNA_indexes,
                    "target_prob",
                ),
                "listeners__rna_synth_prob__max_p AS max_p",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql("target_prob", "max_p"),
        },
        "new_gene_ribosome_init_events_heatmap": {
            "plot_title": "New Gene Ribosome Init Events Per Time Step",
            "num_digits_rounding": 0,
            "box_test_size": "x-small",
            "columns": [
                "list_select(listeners__ribosome_data__ribosome_init_event_per_monomer, "
                f"{new_gene_monomer_indexes}) AS init_events"
            ],
            "custom_sql": avg_1d_array_sql("init_events"),
        },
        "new_gene_ribosome_init_events_portion_heatmap": {
            "plot_title": "New Gene Portion of Initiated Ribosomes",
            "num_digits_rounding": 4,
            "columns": [
                "list_select(listeners__ribosome_data__ribosome_init_event_per_monomer, "
                f"{new_gene_monomer_indexes}) AS init_events",
                "listeners__ribosome_data__did_initialize AS did_initialize",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql("init_events", "did_initialize"),
        },
        "new_gene_actual_rna_synth_prob_heatmap": {
            "plot_title": "New Gene Actual RNA Synth Prob",
            "num_digits_rounding": 4,
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_synth_prob__actual_rna_synth_prob",
                    new_gene_RNA_indexes,
                    "synth_probs",
                )
            ],
            "custom_sql": avg_1d_array_sql("synth_probs"),
        },
        "new_gene_target_rna_synth_prob_heatmap": {
            "plot_title": "New Gene Target RNA Synth Prob",
            "num_digits_rounding": 4,
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_synth_prob__target_rna_synth_prob",
                    new_gene_RNA_indexes,
                    "synth_probs",
                )
            ],
            "custom_sql": avg_1d_array_sql("synth_probs"),
        },
        "new_gene_rnap_counts_heatmap": {
            "plot_title": "New Gene RNAP Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_counts__partial_mRNA_counts",
                    new_gene_mRNA_indexes,
                    "rnap_counts",
                )
            ],
            "custom_sql": avg_1d_array_sql("rnap_counts"),
        },
        "new_gene_rnap_portion_heatmap": {
            "plot_title": "New Gene RNAP Portion",
            "num_digits_rounding": 3,
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_counts__partial_mRNA_counts",
                    new_gene_mRNA_indexes,
                    "rnap_counts",
                ),
                "listeners__unique_molecule_counts__active_RNAP AS active_counts",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql("rnap_counts", "active_counts"),
        },
        "rrna_rnap_counts_heatmap": {
            "plot_title": "rRNA RNAP Counts",
            "num_digits_rounding": 0,
            "columns": ["listeners__rna_counts__partial_rRNA_counts AS rnap_counts"],
            "custom_sql": avg_sum_1d_array_sql("rnap_counts"),
        },
        "rrna_rnap_portion_heatmap": {
            "plot_title": "rRNA RNAP Portion",
            "num_digits_rounding": 3,
            "columns": [
                "listeners__rna_counts__partial_rRNA_counts AS rnap_counts",
                "listeners__unique_molecule_counts__active_RNAP AS active_counts",
            ],
            "custom_sql": avg_sum_1d_array_over_scalar_sql(
                "rnap_counts", "active_counts"
            ),
        },
        "rnap_subunit_rnap_counts_heatmap": {
            "plot_title": "RNAP Subunit RNAP Counts",
            "num_digits_rounding": 0,
            "columns": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                f"{rnap_subunit_mRNA_indexes}) AS rnap_counts"
            ],
            "custom_sql": avg_sum_1d_array_sql("rnap_counts"),
        },
        "rnap_subunit_rnap_portion_heatmap": {
            "plot_title": "RNAP Subunit RNAP Portion",
            "num_digits_rounding": 3,
            "columns": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                f"{rnap_subunit_mRNA_indexes}) AS rnap_counts",
                "listeners__unique_molecule_counts__active_RNAP AS active_counts",
            ],
            "custom_sql": avg_sum_1d_array_over_scalar_sql(
                "rnap_counts", "active_counts"
            ),
        },
        "rnap_subunit_ribosome_counts_heatmap": {
            "plot_title": "RNAP Subunit Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                f"{rnap_subunit_monomer_indexes}) AS ribosome_counts"
            ],
            "custom_sql": avg_sum_1d_array_sql("ribosome_counts"),
        },
        "rnap_subunit_ribosome_portion_heatmap": {
            "plot_title": "RNAP Subunit Ribosome Portion",
            "num_digits_rounding": 3,
            "columns": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                f"{rnap_subunit_monomer_indexes}) AS ribosome_counts",
                "listeners__unique_molecule_counts__active_ribosome AS active_counts",
            ],
            "custom_sql": avg_sum_1d_array_over_scalar_sql(
                "ribosome_counts", "active_counts"
            ),
        },
        "ribosomal_protein_rnap_counts_heatmap": {
            "plot_title": "Ribosomal Protein RNAP Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                f"{ribosomal_mRNA_indexes}) AS rnap_counts"
            ],
            "custom_sql": avg_sum_1d_array_sql("rnap_counts"),
        },
        "ribosomal_protein_rnap_portion_heatmap": {
            "plot_title": "Ribosomal Protein RNAP Portion",
            "num_digits_rounding": 3,
            "columns": [
                "list_select(listeners__rna_counts__partial_mRNA_counts, "
                f"{ribosomal_mRNA_indexes}) AS rnap_counts",
                "listeners__unique_molecule_counts__active_RNAP AS active_counts",
            ],
            "custom_sql": avg_sum_1d_array_over_scalar_sql(
                "rnap_counts", "active_counts"
            ),
        },
        "ribosomal_protein_ribosome_counts_heatmap": {
            "plot_title": "Ribosomal Protein Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                f"{ribosomal_monomer_indexes}) AS ribosome_counts"
            ],
            "custom_sql": avg_sum_1d_array_sql("ribosome_counts"),
        },
        "ribosomal_protein_ribosome_portion_heatmap": {
            "plot_title": "Ribosomal Protein Ribosome Portion",
            "num_digits_rounding": 3,
            "columns": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                f"{ribosomal_monomer_indexes}) AS ribosome_counts",
                "listeners__unique_molecule_counts__active_ribosome AS active_counts",
            ],
            "custom_sql": avg_sum_1d_array_over_scalar_sql(
                "ribosome_counts", "active_counts"
            ),
        },
        "new_gene_ribosome_counts_heatmap": {
            "plot_title": "New Gene Ribosome Counts",
            "num_digits_rounding": 0,
            "box_text_size": "x-small",
            "columns": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                f"{new_gene_monomer_indexes}) AS ribosome_counts"
            ],
            "custom_sql": avg_1d_array_sql("ribosome_counts"),
        },
        "new_gene_ribosome_portion_heatmap": {
            "plot_title": "New Gene Ribosome Portion",
            "num_digits_rounding": 3,
            "columns": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                f"{new_gene_monomer_indexes}) AS ribosome_counts",
                "listeners__unique_molecule_counts__active_ribosome AS active_counts",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql(
                "ribosome_counts", "active_counts"
            ),
        },
        "capacity_gene_rnap_portion_heatmap": {
            "plot_title": "Capacity Gene RNAP Portion: ",
            "num_digits_rounding": 4,
            "columns": [
                get_rnas_combined_as_genes_projection(
                    "listeners__rna_counts__partial_mRNA_counts",
                    capacity_gene_mRNA_indexes,
                    "rnap_counts",
                ),
                "listeners__unique_molecule_counts__active_RNAP AS active_counts",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql("rnap_counts", "active_counts"),
        },
        "capacity_gene_ribosome_portion_heatmap": {
            "plot_title": "Capacity Gene Ribosome Portion: ",
            "num_digits_rounding": 4,
            "columns": [
                "list_select(listeners__ribosome_data__n_ribosomes_per_transcript, "
                f"{capacity_gene_monomer_indexes}) AS ribosome_counts",
                "listeners__unique_molecule_counts__active_ribosome AS active_counts",
            ],
            "custom_sql": avg_1d_array_over_scalar_sql(
                "ribosome_counts", "active_counts"
            ),
        },
        "glucose_consumption_rate": {
            "plot_title": "Average Glucose Consumption Rate (fg/hr)",
            "num_digits_rounding": 1,
            "columns": [
                "-listeners__fba_results__external_exchange_fluxes["
                f"{glucose_idx}] AS glucose_flux",
                "listeners__mass__dry_mass AS dry_mass",
            ],
            "remove_first": True,
            "custom_sql": f"""
                WITH avg_per_cell AS (
                    SELECT avg(glucose_flux * dry_mass) * {flux_scaling_factor}
                        AS avg_fg_glc_per_hr, experiment_id, variant
                    FROM ({{subquery}})
                    GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
                )
                SELECT variant, avg(avg_fg_glc_per_hr) AS mean,
                    stddev(avg_fg_glc_per_hr) AS std
                FROM avg_per_cell
                GROUP BY experiment_id, variant
                """,
        },
        "new_gene_yield_per_glucose": {
            "plot_title": "New Gene fg Protein Yield per fg Glucose",
            "num_digits_rounding": 3,
            "columns": [
                "-listeners__fba_results__external_exchange_fluxes["
                f"{glucose_idx}] AS glucose_flux",
                "listeners__mass__dry_mass AS dry_mass",
                "time",
                "list_select(listeners__monomer_counts, "
                f"{new_gene_monomer_indexes}) AS monomer_counts",
            ],
            "remove_first": True,
            "custom_sql": f"""
                WITH unnested_counts AS (
                    SELECT unnest(monomer_counts) AS monomer_counts,
                        generate_subscripts(monomer_counts, 1) AS monomer_idx,
                        glucose_flux, dry_mass, time, experiment_id, variant,
                        lineage_seed, generation, agent_id
                    FROM ({{subquery}})
                ),
                avg_per_cell AS (
                    SELECT avg(glucose_flux * dry_mass) * {flux_scaling_factor}
                        AS avg_fg_glc_per_hr, (max(time) - min(time)) / 60
                        AS doubling_time, last(monomer_counts::BIGINT ORDER BY time) -
                        first(monomer_counts::BIGINT ORDER BY time) AS monomer_delta,
                        experiment_id, variant, monomer_idx
                    FROM unnested_counts
                    GROUP BY experiment_id, variant, lineage_seed, generation,
                        agent_id, monomer_idx
                ),
                avg_per_variant AS (
                    SELECT experiment_id, variant, avg(monomer_delta / (
                        avg_fg_glc_per_hr * doubling_time))
                        AS avg_monomer_per_fg_glc, monomer_idx,
                        stddev(monomer_delta / (avg_fg_glc_per_hr *
                        doubling_time)) AS std_monomer_per_fg_glc
                    FROM avg_per_cell
                    GROUP BY experiment_id, variant, monomer_idx
                )
                SELECT variant, list(avg_monomer_per_fg_glc ORDER BY monomer_idx)
                    AS mean, list(std_monomer_per_fg_glc ORDER BY monomer_idx) AS std
                FROM avg_per_variant
                GROUP BY experiment_id, variant
                """,
            "post_func": get_gene_mass_prod_func(
                sim_data, "monomer", new_gene_monomer_ids
            ),
        },
        "new_gene_yield_per_hour": {
            "plot_title": "New Gene fg Protein Yield per Hour",
            "num_digits_rounding": 2,
            "columns": [
                "time",
                "list_select(listeners__monomer_counts, "
                f"{new_gene_monomer_indexes}) AS monomer_counts",
            ],
            "remove_first": True,
            "custom_sql": """
                WITH unnested_counts AS (
                    SELECT unnest(monomer_counts) AS monomer_counts, time,
                        generate_subscripts(monomer_counts, 1) AS monomer_idx,
                        experiment_id, variant, lineage_seed, generation, agent_id
                    FROM ({subquery})
                ),
                avg_per_cell AS (
                    SELECT (max(time) - min(time)) / 60 AS doubling_time,
                        last(monomer_counts::BIGINT ORDER BY time) - first(
                            monomer_counts::BIGINT ORDER BY time) AS monomer_delta,
                        experiment_id, variant, monomer_idx
                    FROM unnested_counts
                    GROUP BY experiment_id, variant, lineage_seed, generation,
                        agent_id, monomer_idx
                ),
                avg_per_variant AS (
                    SELECT experiment_id, variant, avg(monomer_delta /
                        doubling_time) AS avg_monomer_per_hr, monomer_idx,
                        stddev(monomer_delta / doubling_time) AS std_monomer_per_hr
                    FROM avg_per_cell
                    GROUP BY experiment_id, variant, monomer_idx
                )
                SELECT variant, list(avg_monomer_per_hr ORDER BY monomer_idx)
                    AS mean, list(std_monomer_per_hr ORDER BY monomer_idx) AS std
                FROM avg_per_variant
                GROUP BY experiment_id, variant
                """,
            "post_func": get_gene_mass_prod_func(
                sim_data, "monomer", new_gene_monomer_ids
            ),
        },
    }

    # Check validity of requested heatmaps and fill in default values where needed
    heatmaps_to_make = HEATMAPS_TO_MAKE_LIST
    total_heatmaps_to_make = 0
    for h in heatmaps_to_make:
        assert h in heatmap_details, "Heatmap " + h + " is not an option"
        heatmap_details[h].setdefault("is_nonstandard_plot", False)
        heatmap_details[h].setdefault("remove_first", False)
        heatmap_details[h].setdefault("num_digits_rounding", 2)
        heatmap_details[h].setdefault("box_text_size", "medium")
        heatmap_details[h].setdefault("is_new_gene_heatmap", False)
        heatmap_details[h].setdefault("is_capacity_gene_heatmap", False)
        heatmap_details[h].setdefault("default_value", -1)
        heatmap_details[h].setdefault("order_results", False)
        heatmap_details[h].setdefault("sucess_sql", success_sql)
        heatmap_details[h].setdefault("custom_sql", None)
        heatmap_details[h].setdefault("post_func", None)
        if h == "new_gene_mRNA_NTP_fraction_heatmap":
            total_heatmaps_to_make += len(ntp_ids) * len(new_gene_cistron_ids)
            heatmap_details[h]["default_value"] = np.full(
                (len(ntp_ids), len(new_gene_cistron_ids)), -1
            )
        elif h.startswith("new_gene_"):
            total_heatmaps_to_make += len(new_gene_cistron_ids)
            heatmap_details[h]["is_new_gene_heatmap"] = True
            heatmap_details[h]["default_value"] = np.full(len(new_gene_mRNA_ids), -1)
        elif h.startswith("capacity_gene_"):
            total_heatmaps_to_make += len(capacity_gene_monomer_ids)
            heatmap_details[h]["is_capacity_gene_heatmap"] = True
            heatmap_details[h]["default_value"] = np.full(
                len(capacity_gene_monomer_ids), -1
            )
        else:
            total_heatmaps_to_make += 1

    # Data extraction
    print("---Data Extraction---")
    heatmap_data = {}
    for h in tqdm(heatmaps_to_make):
        h_details = heatmap_details[h]
        print(h)
        mean_matrix, std_matrix = get_mean_and_std_matrices(
            conn,
            variant_to_row_col,
            variant_matrix_shape,
            history_sql,
            h_details["columns"],
            h_details["remove_first"],
            None,
            h_details["order_results"],
            h_details["sucess_sql"],
            h_details["custom_sql"],
            h_details["post_func"],
            h_details["num_digits_rounding"],
            h_details["default_value"],
        )
        heatmap_data[h] = {"mean": mean_matrix, "std_dev": std_matrix}

    # Plotting
    print("---Plotting---")
    plot_suffix = "_gens_" + str(MIN_CELL_INDEX) + "_through_" + str(MAX_CELL_INDEX)
    heatmap_x_label = "Expression Variant"
    heatmap_y_label = "Translation Efficiency Value"
    figsize_x = 2 + 2 * len(new_gene_expression_factors) / 3
    figsize_y = 2 * len(new_gene_translation_efficiency_values) / 2

    # Figure out whether to create dashboard / individual plots and
    # whether to make std. dev. plots in addition to mean plots
    summary_statistics = ["mean"]
    if dashboard_flag == 0:
        is_dashboards = [False]
    elif dashboard_flag == 1:
        is_dashboards = [True]
    elif dashboard_flag == 2:
        is_dashboards = [True, False]
    if std_dev_flag:
        summary_statistics.append("std_dev")
    for is_dashboard in is_dashboards:
        for summary_statistic in summary_statistics:
            plot_heatmaps(
                heatmap_data,
                heatmap_details,
                new_gene_cistron_ids,
                ntp_ids,
                capacity_gene_common_names,
                total_heatmaps_to_make,
                is_dashboard,
                variant_mask,
                heatmap_x_label,
                heatmap_y_label,
                new_gene_expression_factors,
                new_gene_translation_efficiency_values,
                summary_statistic,
                figsize_x,
                figsize_y,
                outdir,
                plot_suffix,
            )
