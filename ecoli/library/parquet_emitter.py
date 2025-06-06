import atexit
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, cast, Mapping, Optional
from urllib import parse

import duckdb
import numpy as np
import polars as pl
from fsspec.core import url_to_fs, OpenFile
from fsspec.spec import AbstractFileSystem
from tqdm import tqdm
from vivarium.core.emitter import Emitter

METADATA_PREFIX = "output_metadata__"
"""
In the config dataset, user-defined metadata for each store
(see :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.get_output_metadata`)
will be contained in columns with this prefix.
"""

USE_UINT16 = {
    "listeners__rna_synth_prob__n_bound_TF_per_TU",
    "listeners__rna_synth_prob__n_bound_TF_per_cistron",
    "listeners__rnap_data__rna_init_event_per_cistron",
    "listeners__rna_synth_prob__gene_copy_number",
    "listeners__rna_synth_prob__expected_rna_init_per_cistron",
    "listeners__rna_degradation_listener__count_RNA_degraded_per_cistron",
    "listeners__rna_degradation_listener__count_rna_degraded",
    "listeners__transcript_elongation_listener__count_rna_synthesized",
    "listeners__rnap_data__rna_init_event",
    "listeners__rna_synth_prob__promoter_copy_number",
    "listeners__ribosome_data__n_ribosomes_on_each_mRNA",
    "listeners__ribosome_data__mRNA_TU_index",
    "listeners__complexation_listener__complexation_events",
    "listeners__rnap_data__active_rnap_n_bound_ribosomes",
    "listeners__rnap_data__active_rnap_domain_indexes",
    "listeners__rna_synth_prob__bound_TF_indexes",
    "listeners__rna_synth_prob__bound_TF_domains",
}
"""uint16 is 4x smaller than int64 for values between 0 - 65,535."""

USE_UINT32 = {
    "listeners__ribosome_data__ribosome_init_event_per_monomer",
    "listeners__ribosome_data__n_ribosomes_per_transcript",
    "listeners__rna_counts__partial_mRNA_cistron_counts",
    "listeners__rna_counts__mRNA_cistron_counts",
    "listeners__rna_counts__full_mRNA_cistron_counts",
    "listeners__ribosome_data__n_ribosomes_on_partial_mRNA_per_transcript",
    "listeners__monomer_counts",
    "listeners__rna_counts__partial_mRNA_counts",
    "listeners__rna_counts__mRNA_counts",
    "listeners__rna_counts__full_mRNA_counts",
    "listeners__fba_results__catalyst_counts",
}
"""uint32 is 2x smaller than int64 for values between 0 - 4,294,967,295."""


def json_to_parquet(
    emit_dict: dict[str, np.ndarray],
    outfile: str,
):
    """Convert dictionary to Parquet with lower memory footprint."""
    tbl = pl.DataFrame(emit_dict)
    del emit_dict
    tbl.write_parquet(outfile, statistics=False)


def union_by_name(query_sql: str) -> str:
    """
    Modifies SQL query string from :py:func:`~.get_dataset_sql` to
    include ``union_by_name = true`` in the DuckDB ``read_parquet``
    function. This allows data to be read from simulations that have
    different columns by filling in nulls and casting as necessary.
    This comes with a performance penalty and should be avoided if possible.

    Args:
        query_sql: SQL query string from :py:func:`~.get_dataset_sql`
    """
    return query_sql.replace(
        "hive_partitioning = true,", "hive_partitioning = true, union_by_name = true,"
    )


def get_dataset_sql(out_dir: str, experiment_ids: list[str]) -> tuple[str, str, str]:
    """
    Creates DuckDB SQL strings for sim outputs, configs, and metadata on which
    sims were successful.

    Args:
        out_dir: Path to output directory for workflows to retrieve data
            for (relative or absolute local path OR URI beginning with
            ``gcs://`` or ``gs://`` for Google Cloud Storage bucket)
        experiment_ids: List of experiment IDs to include in query. To read data
            from more than one experiment ID, the listeners in the output of the
            first experiment ID in the list must be a strict subset of the listeners
            in the output of the subsequent experiment ID(s).

    Returns:
        3-element tuple containing

        - **history_sql**: SQL query for sim output (see :py:func:`~.read_stacked_columns`),
        - **config_sql**: SQL query for sim configs (see :py:func:`~.get_field_metadata`
          and :py:func:`~.get_config_value`)
        - **success_sql**: SQL query for metadata marking successful sims
          (see :py:func:`~.read_stacked_columns`)

    """
    sql_queries = []
    for query_type in ("history", "configuration", "success"):
        query_files = []
        for experiment_id in experiment_ids:
            query_files.append(
                f"'{os.path.join(out_dir, experiment_id)}/{query_type}/*/*/*/*/*/*.pq'"
            )
        query_files = ", ".join(query_files)
        sql_queries.append(
            f"""
            FROM read_parquet(
                [{query_files}],
                hive_partitioning = true,
                hive_types = {{
                    'experiment_id': VARCHAR,
                    'variant': BIGINT,
                    'lineage_seed': BIGINT,
                    'generation': BIGINT,
                    'agent_id': VARCHAR,
                }}
            )
            """
        )
    return sql_queries[0], sql_queries[1], sql_queries[2]


def num_cells(conn: duckdb.DuckDBPyConnection, subquery: str) -> int:
    """
    Return cell count in DuckDB subquery containing ``experiment_id``,
    ``variant``, ``lineage_seed``, ``generation``, and ``agent_id`` columns.
    """
    return cast(
        tuple,
        conn.sql(f"""SELECT count(
        DISTINCT (experiment_id, variant, lineage_seed, generation, agent_id)
        ) FROM ({subquery})""").fetchone(),
    )[0]


def skip_n_gens(subquery: str, n: int) -> str:
    """
    Modifies a DuckDB SQL query to skip the first ``n`` generations of data.
    """
    return f"SELECT * FROM ({subquery}) WHERE generation >= {n}"


def ndlist_to_ndarray(s) -> np.ndarray:
    """
    Convert a PyArrow series of nested lists with fixed dimensions into
    a Numpy ndarray. This should really only be necessary if you are trying
    to perform linear algebra (e.g. matrix multiplication, dot products) inside
    a user-defined function (see DuckDB documentation on Python Function API and
    ``func`` kwarg for :py:func:`~read_stacked_columns`).

    For elementwise arithmetic of two nested list columns, this can be used
    to define a custom DuckDB function as follows::

        import duckdb
        import polars as pl
        from ecoli.library.parquet_emitter import ndlist_to_ndarray
        def sum_arrays(col_0, col_1):
            return pl.Series(
                ndlist_to_ndarray(col_0) +
                ndlist_to_ndarray(col_1)
            ).to_arrow()
        conn = duckdb.connect()
        conn.create_function(
            "sum_2d_int_arrays", # Function name for use in SQL (must be unique)
            sum_arrays, # Python function that takes and returns PyArrow arrays
            [list[list[int]], list[list[int]]], # Input types (2D lists here)
            list[list[int]], # Return type (2D list here)
            type = "arrow" # Tell DuckDB function operates on Arrow arrays
        )
        conn.sql("SELECT sum_2d_int_arrays(int_col_0, int_col_1) from input_table")
        # Note that function must be registered under different name for each
        # set of unique input/output types
        conn.create_function(
            "sum_2d_int_and_float",
            sum_arrays,
            [list[list[int]], list[list[float]]], # Second input is 2D float array
            list[list[float]], # Adding int to float array gives float in Numpy
            type = "arrow"
        )
        conn.sql("SELECT sum_2d_int_and_float(int_col_0, float_col_0) from input_table")

    """
    inner_s = pl.Series(s)
    dimensions = []
    while inner_s.dtype.is_nested() and len(inner_s) > 0:
        inner_s = inner_s[0]
        dimensions.append(len(inner_s))
    inner_s = inner_s.dtype
    while inner_s.is_nested():
        inner_s = inner_s.inner
        dimensions.append(0)
    return pl.Series(s, dtype=pl.Array(inner_s, tuple(dimensions))).to_numpy()


def ndidx_to_duckdb_expr(
    name: str, idx: list[int | list[int] | list[bool] | str]
) -> str:
    """
    Returns a DuckDB expression for a column equivalent to converting each row
    of ``name`` into an ndarray ``name_arr`` (:py:func:`~.ndlist_to_ndarray`)
    and getting ``name_arr[idx]``. ``idx`` can contain 1D lists of integers,
    boolean masks, or ``":"`` (no 2D+ indices like ``x[[[1,2]]]``). See also
    :py:func:`~named_idx` if pulling out a relatively small set of indices.

    .. WARNING:: DuckDB arrays are 1-indexed so this function adds 1 to every
        supplied integer index!

    Args:
        name: Name of column to recursively index
        idx: To get all elements for a dimension, supply the string ``":"``.
            Otherwise, only single integers or 1D integer lists of indices are
            allowed for each dimension. Some examples::

                [0, 1] # First row, second column
                [[0, 1], 1] # First and second row, second column
                [0, 1, ":"] # First element of axis 1, second of 2, all of 3
                # Final example differs between this function and Numpy
                # This func: 1st and 2nd of axis 1, all of 2, 1st and 2nd of 3
                # Numpy: Complicated, see Numpy docs on advanced indexing
                [[0, 1], ":", [0, 1]]

    """
    idx = idx.copy()
    idx.reverse()
    # Construct expression from inside out (deepest to shallowest axis)
    first_idx = idx.pop(0)
    if isinstance(first_idx, list):
        if isinstance(first_idx[0], int):
            one_indexed_idx = ", ".join(str(i + 1) for i in first_idx)
            select_expr = f"list_select(x_0, [{one_indexed_idx}])"
        elif isinstance(first_idx[0], bool):
            select_expr = f"list_where(x_0, {first_idx})"
        else:
            raise TypeError("Indices must be integers or boolean masks.")
    elif first_idx == ":":
        select_expr = "x_0"
    elif isinstance(first_idx, int):
        select_expr = f"x_0[{int(first_idx) + 1}]"
    else:
        raise TypeError("All indices must be lists, ints, or ':'.")
    i = -1
    for i, indices in enumerate(idx):
        if isinstance(indices, list):
            if isinstance(indices[0], int):
                one_indexed_idx = ", ".join(str(i + 1) for i in indices)
                select_expr = f"list_transform(list_select(x_{i + 1}, [{one_indexed_idx}]), x_{i} -> {select_expr})"
            elif isinstance(indices[0], bool):
                select_expr = f"list_transform(list_where(x_{i + 1}, {indices}), x_{i} -> {select_expr})"
            else:
                raise TypeError("Indices must be integers or boolean masks.")
        elif indices == ":":
            select_expr = f"list_transform(x_{i + 1}, x_{i} -> {select_expr})"
        elif isinstance(indices, int):
            select_expr = (
                f"list_transform(x_{i + 1}[{int(indices) + 1}], x_{i} -> {select_expr})"
            )
        else:
            raise TypeError("All indices must be lists, ints, or ':'.")
    select_expr = select_expr.replace(f"x_{i + 1}", name)
    return select_expr + f" AS {name}"


def named_idx(col: str, names: list[str], idx: list[int]) -> str:
    """
    Create DuckDB expressions for given indices from a list column. Can be
    used in ``projection`` kwarg of :py:func:`~.read_stacked_columns`. Since
    each index gets pulled out into its own column, this greatly simplifies
    aggregations like averages, etc. Only use this if the number of indices
    is relatively small (<100) and the list column is 1-dimensional. For 2+
    dimensions or >100 indices, see :py:func:`~.ndidx_to_duckdb_expr`.

    .. WARNING:: DuckDB arrays are 1-indexed so this function adds 1 to every
        supplied index!

    Args:
        col: Name of list column.
        names: New column names, one for each index.
        idx: Indices to retrieve from ``col``

    Returns:
        DuckDB SQL expression for a set of named columns corresponding to
        the values at given indices of a list column
    """
    return ", ".join(f'{col}[{i + 1}] AS "{n}"' for n, i in zip(names, idx))


def get_field_metadata(
    conn: duckdb.DuckDBPyConnection, config_subquery: str, field: str
) -> list:
    """
    Gets the saved metadata (see
    :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.get_output_metadata`)
    for a given field as a list.

    Args:
        conn: DuckDB connection
        config_subquery: DuckDB query containing sim config data
        field: Name of field to get metadata for
    """
    metadata = cast(
        tuple,
        conn.sql(
            f'SELECT first("{METADATA_PREFIX + field}") FROM ({config_subquery})'
        ).fetchone(),
    )[0]
    if isinstance(metadata, list):
        return metadata
    return list(metadata)


def get_config_value(
    conn: duckdb.DuckDBPyConnection, config_subquery: str, field: str
) -> Any:
    """
    Gets the saved configuration option (anything in config JSON, with
    double underscore concatenation for nested fields due to
    :py:func:`~.flatten_dict`).

    Args:
        conn: DuckDB connection
        config_subquery: DuckDB query containing sim config data
        field: Name of configuration option to get value of
    """
    return cast(
        tuple,
        conn.sql(f'SELECT first("{field}") FROM ({config_subquery})').fetchone(),
    )[0]


def get_plot_metadata(
    conn: duckdb.DuckDBPyConnection, config_subquery: str, variant_name: str
) -> dict[str, Any]:
    """
    Gets dictionary that can be used as ``metadata`` kwarg to
    :py:func:`wholecell.utils.plotting_tools.export_figure`.

    Args:
        conn: DuckDB connection
        config_subquery: DuckDB query containing sim config data
        variant_name: Name of variant
    """
    return {
        "git_hash": get_config_value(conn, config_subquery, "git_hash"),
        "time": get_config_value(conn, config_subquery, "time"),
        "description": get_config_value(conn, config_subquery, "description"),
        "variant_function": variant_name,
        "variant_index": conn.sql(f"SELECT DISTINCT variant FROM ({config_subquery})")
        .arrow()
        .to_pydict()["variant"],
        "seed": conn.sql(f"SELECT DISTINCT lineage_seed FROM ({config_subquery})")
        .arrow()
        .to_pydict()["lineage_seed"],
        "total_gens": cast(
            tuple,
            conn.sql(
                f"SELECT count(DISTINCT generation) FROM ({config_subquery})"
            ).fetchone(),
        )[0],
        "total_variants": cast(
            tuple,
            conn.sql(
                f"SELECT count(DISTINCT variant) FROM ({config_subquery})"
            ).fetchone(),
        )[0],
    }


def open_output_file(outfile: str) -> OpenFile:
    """
    Open a file by its path, whether that be a path on local storage or
    Google Cloud Storage.

    Args:
        outfile: Path to file. Must have ``gs://`` or ``gcs://`` prefix if
            on Google Cloud Storage. Can be relative or absolute path if
            on local storage.

    Returns:
        File object that supports reading, seeking, etc. in bytes
    """
    return url_to_fs(outfile).open(outfile)


def open_arbitrary_sim_data(sim_data_dict: dict[str, dict[int, Any]]) -> OpenFile:
    """
    Given a mapping from experiment ID(s) to mappings from variant ID(s)
    to sim_data path(s), pick an arbitrary sim_data to read.

    Args:
        sim_data_dict: Generated by :py:mod:`runscripts.analysis` and passed to
            each analysis script as an argument.

    Returns:
        File object for arbitrarily chosen sim_data to be loaded
        with ``pickle.load``
    """
    sim_data_path = next(iter(next(iter(sim_data_dict.values())).values()))
    return open_output_file(sim_data_path)


def read_stacked_columns(
    history_sql: str,
    columns: list[str],
    remove_first: bool = False,
    func: Optional[Callable[[pl.DataFrame], pl.DataFrame]] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    order_results: bool = True,
    success_sql: Optional[str] = None,
) -> pl.DataFrame | str:
    """
    Loads columns for many cells. If you would like to perform more advanced
    computatations (aggregations, window functions, etc.) using the optimized
    DuckDB API, you can omit ``conn``, in which case this function will return
    an SQL string that can be used as a subquery. For computations that cannot
    be easily performed using the DuckDB API, you can define a custom function
    ``func`` that will be called on the data for each cell. By default, the
    return value (whether it be the actual data or an SQL subquery) will
    also include the ``experiment_id``, ``variant``, ``lineage_seed``,
    ``generation``, ``agent_id``, and ``time`` columns.

    For example, to get the average total concentration of three bulk molecules
    with indices 100, 1000, and 10000 per cell::

        import duckdb
        from ecoli.library.parquet_emitter import (
            get_dataset_sql, read_stacked_columns)
        history_sql, config_sql, _ = get_dataset_sql('out/', 'exp_id')
        subquery = read_stacked_columns(
            history_sql,
            # Note DuckDB arrays are 1-indexed
            ["bulk[100 + 1] + bulk[1000 + 1] + bulk[10000 + 1] AS bulk_sum",
            "listeners__enzyme_kinetics__counts_to_molar AS counts_to_molar"],
            order_results=False,
        )
        query = '''
            SELECT avg(bulk_sum * counts_to_molar) AS avg_total_conc
            FROM ({subquery})
            GROUP BY experiment_id, variant, lineage_seed, generation, agent_id
            '''
        conn = duckdb.connect()
        data = conn.sql(query).pl()

    Here is a more complicated example that defines a custom function to get
    the per-cell average RNA synthesis probability per cistron::

        import duckdb
        import pickle
        from ecoli.library.parquet_emitter import (
            get_dataset_sql, ndlist_to_ndarray, read_stacked_columns)
        history_sql, config_sql, _ = get_dataset_sql('out/', 'exp_id')
        # Load sim data
        with open("reconstruction/sim_data/kb/simData.cPickle", "rb") as f:
            sim_data = pickle.load(f)
        # Get mapping from RNAs (TUs) to cistrons
        cistron_tu_mat = sim_data.process.transcription.cistron_tu_mapping_matrix
        # Custom aggregation function with Numpy dot product and mean
        def avg_rna_synth_prob_per_cistron(rna_synth_prob):
            # Convert rna_synth_prob into 2-D Numpy array (time x TU)
            rna_synth_prob = ndlist_to_ndarray(rna_synth_prob[
                "listeners__rna_synth_prob__actual_rna_synth_prob"])
            rna_synth_prob_per_cistron = cistron_tu_mat.dot(rna_synth_prob.T).T
            # Return value must be a PyArrow table
            return pl.DataFrame({'avg_rna_synth_prob_per_cistron': [
                rna_synth_prob_per_cistron.mean(axis=0)]}).to_arrow()
        conn = duckdb.connect()
        result = read_stacked_columns(
            history_sql,
            ["listeners__rna_synth_prob__actual_rna_synth_prob"],
            func=avg_rna_synth_prob_per_cistron,
            conn=conn,
        )

    Args:
        history_sql: DuckDB SQL string from :py:func:`~.get_dataset_sql`,
            potentially with filters appended in ``WHERE`` clause
        columns: Names of columns to read data for. Alternatively, DuckDB
            expressions of columns (e.g. ``avg(listeners__mass__cell_mass) AS avg_mass``
            or the output of :py:func:`~.named_idx` or :py:func:`~.ndidx_to_duckdb_expr`).
        remove_first: Remove data for first timestep of each cell
        func: Function to call on data for each cell, should take and
            return a Polars DataFrame with columns equal to ``columns``
        conn: DuckDB connection instance with which to run query. Typically
            provided by :py:func:`runscripts.analysis.main` to the ``plot``
            method of analysis scripts (tweaked some DuckDB settings). Can
            be omitted to return SQL query string to be used as subquery
            instead of running query immediately and returning result.
        order_results: Whether to sort returned table by ``experiment_id``,
            ``variant``, ``lineage_seed``, ``generation``, ``agent_id``, and
            ``time``. If no ``conn`` is provided, this can usually be disabled
            and any sorting can be deferred until the last step in the query with
            a manual ``ORDER BY``. Doing this can greatly reduce RAM usage.
        success_sql: Final DuckDB SQL string from :py:func:`~.get_dataset_sql`.
            If provided, will be used to filter out unsuccessful sims.
    """
    id_cols = "experiment_id, variant, lineage_seed, generation, agent_id, time"
    columns = ", ".join(columns)
    sql_query = f"SELECT {columns}, {id_cols} FROM ({history_sql})"
    # Use a semi join to filter out unsuccessful sims
    if success_sql is not None:
        sql_query = f"""
            SELECT * FROM ({sql_query})
            SEMI JOIN ({success_sql})
            USING (experiment_id, variant, lineage_seed, generation, agent_id)
            """
    # Use an anti join to remove rows for first timestep of each sim
    if remove_first:
        sql_query = f"""
            SELECT * FROM ({sql_query})
            ANTI JOIN (
                SELECT experiment_id, variant, lineage_seed, generation,
                    agent_id, MIN(time) AS time
                FROM ({history_sql.replace("COLNAMEHERE", "time")})
                GROUP BY experiment_id, variant, lineage_seed, generation,
                    agent_id
            ) USING (experiment_id, variant, lineage_seed, generation,
                agent_id, time)
            """
    if func is not None:
        if conn is None:
            raise RuntimeError("`conn` must be provided with `func`.")
        # Get all cell identifiers
        cell_ids = conn.sql(f"""SELECT DISTINCT ON(experiment_id, variant,
            lineage_seed, generation, agent_id) experiment_id, variant,
            lineage_seed, generation, agent_id FROM ({history_sql}) ORDER BY {id_cols}
        """).fetchall()
        all_cell_tbls = []
        for experiment_id, variant, lineage_seed, generation, agent_id in tqdm(
            cell_ids
        ):
            cell_joined = f"""SELECT * FROM ({sql_query})
                WHERE experiment_id = '{experiment_id}' AND
                    variant = {variant} AND
                    lineage_seed = {lineage_seed} AND
                    generation = {generation} AND
                    agent_id = '{agent_id}'
                ORDER BY time
                """
            # Apply func to data for each cell
            all_cell_tbls.append(func(conn.sql(cell_joined).pl()))
        return pl.concat(all_cell_tbls)
    if order_results:
        query = f"SELECT * FROM ({sql_query}) ORDER BY {id_cols}"
    else:
        query = sql_query
    if conn is None:
        return query
    return conn.sql(query).pl()


def get_encoding(
    val: Any, field_name: str, use_uint16: bool = False, use_uint32: bool = False
) -> tuple[Any, str]:
    """
    Get optimal Numpy type for input value.
    """
    if isinstance(val, (float, np.floating)):
        return np.float64, ()
    elif isinstance(val, bool):
        return np.bool_, ()
    elif isinstance(val, (int, np.integer)):
        # Optimize memory usage for select integer fields
        if use_uint16:
            np_type = np.uint16
        elif use_uint32:
            np_type = np.uint32
        else:
            np_type = np.int64
        return np_type, ()
    elif isinstance(val, (str, np.str_)):
        return np.dtypes.StringDType, ()
    elif isinstance(val, np.ndarray):
        if len(val) > 0:
            np_type, _ = get_encoding(
                val.flat[0].item(), field_name, use_uint16, use_uint32
            )
            return np_type, val.shape
        return None, val.shape
    elif isinstance(val, (list, tuple)):
        if len(val) > 0:
            for inner_val in val:
                if inner_val is not None:
                    np_type, inner_dims = get_encoding(val[0], use_uint16, use_uint32)
                    if np_type is None:
                        continue
                    return (np_type, (len(val), *inner_dims))
        return None, (0,)
    elif val is None:
        return None, ()
    raise TypeError(f"{field_name} has unsupported type {type(val)}.")


_FLAG_FIRST = object()


def flatten_dict(d: dict):
    """
    Flatten nested dictionary down to key-value pairs where each key
    concatenates all the keys needed to reach the corresponding value
    in the input. Allows each leaf field in a nested emit to be turned
    into a column in a Parquet file for efficient storage and retrieval.
    """
    results: list[tuple[str, Any]] = []

    def visit_key(subdict, results, partialKey):
        for k, v in subdict.items():
            newKey = k if partialKey == _FLAG_FIRST else f"{partialKey}__{k}"
            if isinstance(v, Mapping):
                visit_key(v, results, newKey)
            else:
                results.append((newKey, v))

    visit_key(d, results, _FLAG_FIRST)
    return dict(results)


class ParquetEmitter(Emitter):
    """
    Emit data to a Parquet dataset.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Configure emitter.

        Args:
            config: Should be a dictionary as follows::

                {
                    'type': 'parquet',
                    'emits_to_batch': Number of emits per Parquet row
                        group (optional, default: 400),
                    # One of the following is REQUIRED
                    'out_dir': local output directory (absolute/relative),
                    'out_uri': Google Cloud storage bucket URI
                }

        """
        if "out_uri" not in config:
            self.out_uri = os.path.abspath(config["out_dir"])
        else:
            self.out_uri = config["out_uri"]
        self.filesystem: AbstractFileSystem
        self.filesystem, _ = url_to_fs(self.out_uri)
        self.batch_size = config.get("batch_size", 400)
        self.executor = ThreadPoolExecutor(2)
        # Keep a cache of field encodings and fields encountered
        self.buffered_emits = {}
        self.non_null_keys: set[str] = set()
        self.num_emits = 0
        # Wait until next batch of emits to check whether last batch
        # was successfully written to Parquet in order to avoid blocking
        self.last_batch_future: Future = Future()
        # Set either by EcoliSim or by EngineProcess if sim reaches division
        self.success = False
        atexit.register(self._finalize)

    def _finalize(self):
        """Convert remaining batched emits to Parquet at sim shutdown
        and mark sim as successful if ``success`` flag was set. In vEcoli,
        this is done by :py:class:`~ecoli.experiments.ecoli_master_sim.EcoliSim`
        upon reaching division.
        """
        outfile = os.path.join(
            self.out_uri,
            self.experiment_id,
            "history",
            self.partitioning_path,
            f"{self.num_emits}.pq",
        )
        for k, v in self.buffered_emits.items():
            self.buffered_emits[k] = v[: self.num_emits % self.batch_size]
        if not self.filesystem.exists(outfile):
            json_to_parquet(self.buffered_emits, outfile)
        # Hive-partitioned directory that only contains successful sims
        if self.success:
            success_file = os.path.join(
                self.out_uri,
                self.experiment_id,
                "success",
                self.partitioning_path,
                "s.pq",
            )
            try:
                self.filesystem.delete(os.path.dirname(success_file), recursive=True)
            except (FileNotFoundError, OSError):
                pass
            self.filesystem.makedirs(os.path.dirname(success_file))
            pl.DataFrame({"success": [True]}).write_parquet(
                success_file,
                statistics=False,
            )

    def emit(self, data: dict[str, Any]):
        """
        Flattens emit dictionary by concatenating nested key names with double
        underscores (:py:func:`~.flatten_dict`), serializes flattened emit with
        ``orjson``, and writes newline-delimited JSONs in a temporary file to be
        batched for some number of timesteps before conversion to Parquet by
        :py:func:`~.json_to_parquet`.

        The output directory (``config["out_dir"]`` or ``config["out_uri"]``) will
        have the following structure::

            {experiment_id}
            |-- history
            |   |-- experiment_id={experiment_id}
            |   |   |-- variant={variant}
            |   |   |   |-- lineage_seed={seed}
            |   |   |   |   |-- generation={generation}
            |   |   |   |   |   |-- agent_id={agent_id}
            |   |   |   |   |   |   |-- 400.pq (batched emits)
            |   |   |   |   |   |   |-- 800.pq
            |   |   |   |   |   |   |-- ..
            |-- configuration
            |   |-- experiment_id={experiment_id}
            |   |   |-- variant={variant}
            |   |   |   |-- lineage_seed={seed}
            |   |   |   |   |-- generation={generation}
            |   |   |   |   |   |-- agent_id={agent_id}
            |   |   |   |   |   |   |-- config.pq (sim config data)

        This Hive-partioned directory structure can be efficiently filtered
        and queried using DuckDB (see :py:func:`~.get_dataset_sql`).
        """
        # Config will always be first emit
        if data["table"] == "configuration":
            data = {**data["data"].pop("metadata"), **data["data"]}
            data["time"] = data.get("initial_global_time", 0.0)
            # Manually create filepaths with hive partitioning
            # Start agent ID with 1 to avoid leading zeros
            agent_id = data.get("agent_id", "1")
            quoted_experiment_id = parse.quote_plus(
                data.get("experiment_id", "default")
            )
            partitioning_keys = {
                "experiment_id": quoted_experiment_id,
                "variant": data.get("variant", 0),
                "lineage_seed": data.get("lineage_seed", 0),
                "generation": len(agent_id),
                "agent_id": agent_id,
            }
            self.experiment_id = quoted_experiment_id
            self.partitioning_path = os.path.join(
                *(f"{k}={v}" for k, v in partitioning_keys.items())
            )
            data = flatten_dict(data)
            config_emit = {}
            for k, v in data.items():
                np_type, dims = get_encoding(v, k)
                if np_type is None:
                    config_emit[k] = [None]
                    continue
                config_emit[k] = np.zeros((1,) + dims, dtype=np_type)
                config_emit[k][0] = v
            outfile = os.path.join(
                self.out_uri,
                self.experiment_id,
                "configuration",
                self.partitioning_path,
                "config.pq",
            )
            # Cleanup any existing output files from previous runs then
            # create new folder for config / simulation output
            try:
                self.filesystem.delete(os.path.dirname(outfile), recursive=True)
            except (FileNotFoundError, OSError):
                pass
            self.filesystem.makedirs(os.path.dirname(outfile))
            self.last_batch_future = self.executor.submit(
                json_to_parquet,
                config_emit,
                outfile,
            )
            # Delete any sim output files in final filesystem
            history_outdir = os.path.join(
                self.out_uri, self.experiment_id, "history", self.partitioning_path
            )
            try:
                self.filesystem.delete(history_outdir, recursive=True)
            except (FileNotFoundError, OSError):
                pass
            self.filesystem.makedirs(history_outdir)
            return
        # Each Engine that uses this emitter should only simulate a single cell
        # In lineage simulations, StopAfterDivision Step will terminate
        # Engine in timestep immediately after division (first with 2 cells)
        # In colony simulations, EngineProcess will terminate simulation
        # immediately upon division (following branch is never invoked)
        if len(data["data"]["agents"]) > 1:
            return
        for agent_data in data["data"]["agents"].values():
            agent_data["time"] = float(data["data"]["time"])
            agent_data = flatten_dict(agent_data)
            for k, v in agent_data.items():
                if k not in self.buffered_emits:
                    np_type, dims = get_encoding(v, k, k in USE_UINT16, k in USE_UINT32)
                    if np_type is None:
                        continue
                    self.buffered_emits[k] = np.zeros(
                        (self.batch_size,) + dims, dtype=np_type
                    )
                emit_idx = self.num_emits % self.batch_size
                if isinstance(self.buffered_emits[k], np.ndarray):
                    v_arr = np.asarray(v, dtype=self.buffered_emits[k].dtype)
                    if v_arr.shape != self.buffered_emits[k].shape[1:]:
                        self.buffered_emits[k] = self.buffered_emits[k][
                            :emit_idx
                        ].tolist() + [0] * (self.batch_size - emit_idx)
                        v = v_arr.tolist()
                    else:
                        v = v_arr
                elif isinstance(v, np.ndarray):
                    v = v.tolist()
                self.buffered_emits[k][emit_idx] = v
        self.num_emits += 1
        if self.num_emits % self.batch_size == 0:
            # If last batch of emits failed, exception should be raised here
            self.last_batch_future.result()
            outfile = os.path.join(
                self.out_uri,
                self.experiment_id,
                "history",
                self.partitioning_path,
                f"{self.num_emits}.pq",
            )
            self.last_batch_future = self.executor.submit(
                json_to_parquet,
                self.buffered_emits,
                outfile,
            )
            self.buffered_emits = {}
