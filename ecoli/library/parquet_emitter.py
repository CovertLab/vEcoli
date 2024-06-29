import atexit
import os
import pathlib
from itertools import pairwise
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, cast, Mapping, Optional
from urllib import parse

import duckdb
import numpy as np
import orjson
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import fs
from pyarrow import json as pj
from pyarrow import parquet as pq
from tqdm import tqdm
from vivarium.core.emitter import Emitter
from vivarium.core.serialize import make_fallback_serializer_function

METADATA_PREFIX = "data__output_metadata__"
"""
In the config dataset, user-defined metadata for each store
(see :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.get_output_metadata`)
will be contained in columns with this prefix.
"""

EXPERIMENT_SCHEMA_SUFFIX = "variant=0/lineage_seed=0/generation=1/agent_id=1/_metadata"
"""
Hive partitioning suffix following experiment ID partition for Parquet file
containing schema unified over all cells in this simulation.
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
    ndjson: str,
    encodings: dict[str, str],
    schema: pa.Schema,
    outfile: str,
    filesystem: Optional[fs.FileSystem] = None,
):
    """
    Reads newline-delimited JSON file and converts to Parquet file.

    Args:
        ndjson: Path to newline-delimited JSON file.
        encodings: Mapping of column names to Parquet encodings
        schema: PyArrow schema of Parquet file to write
        outfile: Filepath of output Parquet file
        filesystem: PyArrow filesystem for Parquet output (local if None)
    """
    parse_options = pj.ParseOptions(explicit_schema=schema)
    read_options = pj.ReadOptions(use_threads=False, block_size=int(1e7))
    t = pj.read_json(ndjson, read_options=read_options, parse_options=parse_options)
    pq.write_table(
        t,
        outfile,
        use_dictionary=False,
        compression="zstd",
        column_encoding=encodings,
        filesystem=filesystem,
        # Writing statistics with giant nested columns bloats metadata
        # and dramatically slows down reading while increasing RAM usage
        write_statistics=False,
    )
    pathlib.Path(ndjson).unlink()


def get_dataset_sql(out_dir: str) -> tuple[str, str]:
    """
    Creates DuckDB SQL strings for sim configs and outputs.

    Args:
        out_dir: Path to directory containing ``history`` and
            ``configuration`` subdirectories (URI beginning with ``gcs://``
            or ``gs://`` if sim output in Google Cloud Storage)

    Returns:
        Tuple of DuckDB SQL strings: first can be passed to
            :py:func:`~.read_stacked_columns` to read one or more columns
            of sim output by name, second can be passed to
            :py:func:`~.get_field_metadata` or :py:func:`~.get_config_value`
            to retrieve metadata / config options for sims
    """
    return (
        f"""
        FROM read_parquet(
            ['{os.path.join(out_dir, '*')}/history/*/{EXPERIMENT_SCHEMA_SUFFIX}',
            '{os.path.join(out_dir, '*')}/history/*/*/*/*/*/*.pq'],
            hive_partitioning = true,
            hive_types = {{
                'experiment_id': VARCHAR,
                'variant': BIGINT,
                'lineage_seed': BIGINT,
                'generation': BIGINT,
                'agent_id': VARCHAR,
            }}
        )
        """,
        f"""
        FROM read_parquet(
            '{os.path.join(out_dir, '*')}/configuration/*/*/*/*/*/*.pq',
            hive_partitioning = true,
            hive_types = {{
                'experiment_id': VARCHAR,
                'variant': BIGINT,
                'lineage_seed': BIGINT,
                'generation': BIGINT,
                'agent_id': VARCHAR,
            }}
        )
        """,
    )


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


def ndlist_to_ndarray(s: pa.Array) -> np.ndarray:
    """
    Convert a PyArrow series of nested lists with fixed dimensions into
    a Numpy ndarray. This should really only be necessary if you are trying
    to perform linear algebra (e.g. matrix multiplication, dot products) inside
    a user-defined function (see DuckDB documentation on Python Function API and
    ``func`` kwarg for :py:func:`~read_stacked_columns`).

    For elementwise arithmetic with two nested list columns, this can be used
    in combination with :py:func:`~.ndarray_to_ndlist` to define a custom DuckDB
    function as follows::

        import duckdb
        from ecoli.library.parquet_emitter import (
            ndarray_to_ndlist, ndlist_to_ndarray)
        def sum_arrays(col_0, col_1):
            return ndarray_to_ndlist(
                ndlist_to_ndarray(col_0) +
                ndlist_to_ndarray(col_1)
            )
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
        conn.sql("SELECT sum_2d_int_arrays(int_col_0, float_col_0) from input_table")

    """
    dimensions = [1, len(s)]
    while isinstance(s.type, pa.ListType) or isinstance(s.type, pa.FixedSizeListType):
        s = pc.list_flatten(s)
        dimensions.append(len(s))
    dimensions = [p[1] // p[0] for p in pairwise(dimensions)]
    return s.to_numpy().reshape(dimensions)


def ndarray_to_ndlist(arr: np.ndarray) -> pa.FixedSizeListArray:
    """
    Convert a Numpy ndarray into a PyArrow FixedSizeListArray. This is useful
    for writing user-defined functions (see DuckDB documentation on Python
    Function API and ``func`` kwarg for :py:func:`~read_stacked_columns`)
    that expect a PyArrow return type.

    Note that the number of rows in the returned PyArrow array is equal to the
    size of the first dimension of the input array. This means for a 3 x 4 x 5
    Numpy array, the return PyArrow array will have 3 rows where each row is
    a nested list with 4 lists of length 5.
    """
    arrow_flat_array = pa.array(arr.flatten())
    nested_array = arrow_flat_array
    for dim_size in reversed(arr.shape[1:]):
        nested_array = pa.FixedSizeListArray.from_arrays(nested_array, dim_size)
    return nested_array


def ndidx_to_duckdb_expr(name: str, idx: list[int | list[int | bool] | str]) -> str:
    """
    Returns a DuckDB expression for a column equivalent to converting each row
    of ``name`` into an ndarray ``name_arr`` (:py:func:`~.ndlist_to_ndarray`)
    and getting ``name_arr[idx]``. ``idx`` can contain 1D lists of integers,
    boolean masks, or ``":"`` (no 2D+ indices like x[[[1,2]]]). See also
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
                select_expr = f"list_transform(list_select(x_{i+1}, [{one_indexed_idx}]), x_{i} -> {select_expr})"
            elif isinstance(indices[0], bool):
                select_expr = f"list_transform(list_where(x_{i+1}, {indices}), x_{i} -> {select_expr})"
            else:
                raise TypeError("Indices must be integers or boolean masks.")
        elif indices == ":":
            select_expr = f"list_transform(x_{i+1}, x_{i} -> {select_expr})"
        elif isinstance(first_idx, int):
            select_expr = f"list_transform(x_{i+1}[{cast(int, indices) + 1}], x_{i} -> {select_expr})"
        else:
            raise TypeError("All indices must be lists, ints, or ':'.")
    select_expr = select_expr.replace(f"x_{i+1}", name)
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
    Gets the saved metadata for a given field as a list.

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
    Gets the saved configuration option.

    Args:
        conn: DuckDB connection
        config_subquery: DuckDB query containing sim config data
        field: Name of configuration option to get value of
    """
    return cast(
        tuple,
        conn.sql(f'SELECT first("data__{field}") FROM ({config_subquery})').fetchone(),
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


def open_output_file(outfile: str) -> pa.NativeFile:
    """
    Open a file by its path, whether that be a path on local storage or
    Google Cloud Storage.

    Args:
        outfile: Path to file. Must have ``gs://`` or ``gcs://`` prefix if
            on Google Cloud Storage. Can be relative or absolute path if
            on local storage.

    Returns:
        PyArrow file object that supports reading, seeking, etc. in bytes
    """
    if not (outfile.startswith("gs://") or outfile.startswith("gcs://")):
        outfile = os.path.abspath(outfile)
    filesystem, outfile = fs.FileSystem.from_uri(outfile)
    return filesystem.open_input_file(outfile)


def open_arbitrary_sim_data(sim_data_dict: dict[str, dict[int, Any]]) -> pa.NativeFile:
    """
    Given a mapping from experiment ID(s) to mappings from variant ID(s)
    to sim_data path(s), pick an arbitrary sim_data to read.

    Args:
        sim_data_dict: Generated by ``scripts/analysis.py`` and passed to
            each analysis script as an argument.

    Returns:
        PyArrow file object for arbitrarily chosen sim_data to be loaded
        with ``pickle.load``
    """
    sim_data_path = next(iter(next(iter(sim_data_dict.values())).values()))
    return open_output_file(sim_data_path)


def read_stacked_columns(
    history_sql: str,
    columns: list[str],
    projections: Optional[list[str]] = None,
    remove_first: bool = False,
    func: Optional[Callable[[pa.Table], pa.Table]] = None,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    order_results: bool = True,
) -> pa.Table | str:
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
        history_sql, config_sql = get_dataset_sql('out/')
        subquery = read_stacked_columns(
            history_sql,
            ["bulk", "listeners__enzyme_kinetics__counts_to_molar"],
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
        data = conn.sql(query).arrow()

    Here is a more complicated example that defines a custom function to get
    the per-cell average RNA synthesis probability per cistron::

        import duckdb
        import pickle
        import pyarrow as pa
        from ecoli.library.parquet_emitter import (
            get_dataset_sql, ndlist_to_ndarray, read_stacked_columns)
        history_sql, config_sql = get_dataset_sql('out/')
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
            return pa.table({'avg_rna_synth_prob_per_cistron': [
                rna_synth_prob_per_cistron.mean(axis=0)]})
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
        columns: Names of columns to read data for. Unless you need to perform
            a computation involving multiple columns, calling this function
            many times with one column each time will use less RAM.
        projections: Expressions to project from ``columns`` that are read.
            If not given, all ``columns`` are projected as is.
        remove_first: Remove data for first timestep of each cell
        func: Function to call on data for each cell, should take and
            return a PyArrow table with columns equal to ``columns``
        conn: DuckDB connection instance with which to run query. Typically
            provided by :py:func:`scripts.analysis.main` to the ``plot``
            method of analysis scripts (tweaked some DuckDB settings). Can
            be omitted to return SQL query string to be used as subquery
            instead of running query immediately and returning result.
        order_results: Whether to sort returned table by ``experiment_id``,
            ``variant``, ``lineage_seed``, ``generation``, ``agent_id``, and
            ``time``. If no ``conn`` is provided, this can usually be disabled
            and any sorting can be deferred until the last step in the query with
            a manual ``ORDER BY``. Doing this can greatly reduce RAM usage.
    """
    id_cols = "experiment_id, variant, lineage_seed, generation, agent_id, time"
    if projections is None:
        projections = columns
    projections.append(id_cols)
    projections = ", ".join(projections)
    sql_query = f"SELECT {projections} FROM ({history_sql})"
    # Use an antijoin to remove rows for first timestep of each sim
    if remove_first:
        sql_query = f"""
            SELECT * FROM ({sql_query})
            WHERE (experiment_id, variant, lineage_seed, generation,
                agent_id, time)
            NOT IN (
                SELECT (experiment_id, variant, lineage_seed, generation,
                    agent_id, MIN(time))
                FROM ({history_sql.replace("COLNAMEHERE", "time")})
                GROUP BY experiment_id, variant, lineage_seed, generation,
                    agent_id
            )"""
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
            all_cell_tbls.append(func(conn.sql(cell_joined).arrow()))
        return pa.concat_tables(all_cell_tbls)
    if order_results:
        query = f"SELECT * FROM ({sql_query}) ORDER BY {id_cols}"
    else:
        query = sql_query
    if conn is None:
        return query
    return conn.sql(query).arrow()


def get_encoding(
    val: Any, field_name: str, use_uint16: bool = False, use_uint32: bool = False
) -> tuple[Any, str, str, bool]:
    """
    Get optimal PyArrow type and Parquet encoding for input value.
    """
    if isinstance(val, float):
        # Polars does not support BYTE_STREAM_SPLIT yet
        return pa.float64(), "PLAIN", field_name, False
    elif isinstance(val, bool):
        return pa.bool_(), "RLE", field_name, False
    elif isinstance(val, int):
        # Optimize memory usage for select integer fields
        if use_uint16:
            pa_type = pa.uint16()
        elif use_uint32:
            pa_type = pa.uint32()
        else:
            pa_type = pa.int64()
        return pa_type, "DELTA_BINARY_PACKED", field_name, False
    elif isinstance(val, str):
        return pa.string(), "DELTA_BYTE_ARRAY", field_name, False
    elif isinstance(val, list):
        if len(val) > 0:
            inner_type, _, field_name, is_null = get_encoding(
                val[0], field_name, use_uint16, use_uint32
            )
            # PLAIN encoding yields overall better compressed size for lists
            return pa.list_(inner_type), "PLAIN", field_name + ".list.element", is_null
        return pa.list_(pa.null()), "PLAIN", field_name + ".list.element", True
    elif val is None:
        return pa.null(), "PLAIN", field_name, True
    raise TypeError(f"{field_name} has unsupported type {type(val)}.")


_FLAG_FIRST = object()


def flatten_dict(d: dict):
    """
    Flatten nested dictionary down to key-value pairs where each key
    concatenates all the keys needed to reach the corresponding value
    in the input. Prunes empty dicts and lists. Allows each field in
    emits to be written to, compressed, and encoded as its own column
    in a Parquet file for efficient storage and retrieval.
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
                    'config': {
                        'emits_to_batch': Number of emits per Parquet row
                            group (default: 400),
                        # Only pick ONE of the following
                        'out_dir': local output directory (absolute/relative),
                        'out_uri': Google Cloud storage bucket URI
                    }
                }

        """
        if "out_uri" not in config["config"]:
            out_uri = os.path.abspath(config["config"]["out_dir"])
        else:
            out_uri = config["config"]["out_uri"]
        self.filesystem, self.outdir = fs.FileSystem.from_uri(out_uri)
        self.batch_size = config["config"].get("batch_size", 400)
        self.fallback_serializer = make_fallback_serializer_function()
        # Batch emits as newline-delimited JSONs in temporary file
        self.temp_data = tempfile.NamedTemporaryFile(delete=False)
        self.executor = ThreadPoolExecutor(2)
        # Keep a cache of field encodings and fields encountered
        self.encodings: dict[str, str] = {}
        self.schema = pa.schema([])
        self.non_null_keys: set[str] = set()
        self.num_emits = 0
        atexit.register(self._finalize)

    def _finalize(self):
        """Convert remaining batched emits to Parquet at sim shutdown.
        Read 10 most recent Parquet schema from ``_metadata`` file written
        by other simulations with the same experiment ID, then unify
        with the current simulation Parquet schema and write a ``_metadata``
        file for the current simulation. Write same ``_metadata`` to Hive
        partition of current experiment ID, variant 0, lineage_seed 0,
        generation 1, and agent_id 1. This is the first file read by DuckDB
        in the subquery generated by :py:func:`~.get_dataset_sql` and ensures
        that even columns which are all NULL for some simulations are read as
        the correct non-NULL type from other simulations."""
        outfile = os.path.join(
            self.outdir,
            self.experiment_id,
            "history",
            self.partitioning_path,
            f"{self.num_emits}.pq",
        )
        if self.filesystem.get_file_info(outfile).type == 0:
            json_to_parquet(
                self.temp_data.name,
                self.encodings,
                self.schema,
                outfile,
                self.filesystem,
            )
        experiment_dir = fs.FileSelector(
            os.path.join(
                self.outdir,
                self.experiment_id,
                "history",
                f"experiment_id={self.experiment_id}",
            ),
            recursive=True,
        )
        latest_schemas = sorted(
            (
                f
                for f in self.filesystem.get_file_info(experiment_dir)
                if os.path.basename(f.path) == "_metadata"
            ),
            key=lambda x: x.mtime_ns,
        )[-10:]
        schemas_to_unify = [self.schema]
        for schema in latest_schemas:
            schemas_to_unify.append(
                pq.read_schema(schema.path, filesystem=self.filesystem)
            )
        unified_schema = pa.unify_schemas(schemas_to_unify)
        unified_schema_path = os.path.join(
            self.outdir,
            self.experiment_id,
            "history",
            self.partitioning_path,
            "_metadata",
        )
        pq.write_metadata(
            unified_schema, unified_schema_path, filesystem=self.filesystem
        )
        experiment_schema_path = os.path.join(
            self.outdir, "history", self.experiment_id, EXPERIMENT_SCHEMA_SUFFIX
        )
        pq.write_metadata(
            unified_schema, experiment_schema_path, filesystem=self.filesystem
        )

    def emit(self, data: dict[str, Any]):
        """
        Serializes emit data with ``orjson`` and writes newline-delimited
        JSONs in a temporary file to be batched before conversion to Parquet.

        The output directory consists of two hive-partitioned datasets: one for
        sim metadata called ``configuration`` and another for sim output called
        ``history``. The partitioning keys are, in order, experiment_id (str),
        variant (int), lineage seed (int), generation (int), and agent_id (str).

        By using a single output directory for many runs of a model, advanced
        filtering and computation can be performed on data from all those
        runs using PyArrow datasets (see :py:func:`~.get_datasets`).
        """
        # Config will always be first emit
        if data["table"] == "configuration":
            metadata = data["data"].pop("metadata")
            data["data"] = {**metadata, **data["data"]}
            data["time"] = data["data"].get("initial_global_time", 0.0)
            # Manually create filepaths with hive partitioning
            # Start agent ID with 1 to avoid leading zeros
            agent_id = data["data"].get("agent_id", "1")
            quoted_experiment_id = parse.quote_plus(
                data["data"].get("experiment_id", "default")
            )
            partitioning_keys = {
                "experiment_id": quoted_experiment_id,
                "variant": data["data"].get("variant", 0),
                "lineage_seed": data["data"].get("lineage_seed", 0),
                "generation": len(agent_id),
                "agent_id": agent_id,
            }
            self.experiment_id = quoted_experiment_id
            self.partitioning_path = os.path.join(
                *(f"{k}={v}" for k, v in partitioning_keys.items())
            )
            data = flatten_dict(data)
            data_str = orjson.dumps(
                data,
                option=orjson.OPT_SERIALIZE_NUMPY,
                default=self.fallback_serializer,
            )
            self.temp_data.write(data_str)
            data = orjson.loads(data_str)
            encodings = {}
            schema = []
            for k, v in data.items():
                pa_type, encoding, field_name, _ = get_encoding(v, k)
                if encoding is not None:
                    encodings[field_name] = encoding
                schema.append((k, pa_type))
            outfile = os.path.join(
                self.outdir,
                self.experiment_id,
                data["table"],
                self.partitioning_path,
                "config.pq",
            )
            # Cleanup any existing output files from previous runs then
            # create new folder for config / simulation output
            try:
                self.filesystem.delete_dir(os.path.dirname(outfile))
            except FileNotFoundError:
                pass
            self.filesystem.create_dir(os.path.dirname(outfile))
            self.executor.submit(
                json_to_parquet,
                self.temp_data.name,
                self.encodings,
                pa.schema(schema),
                outfile,
                self.filesystem,
            )
            self.temp_data = tempfile.NamedTemporaryFile(delete=False)
            # Delete any sim output files in final filesystem
            history_outdir = os.path.join(
                self.outdir, self.experiment_id, "history", self.partitioning_path
            )
            try:
                self.filesystem.delete_dir(history_outdir)
            except FileNotFoundError:
                pass
            self.filesystem.create_dir(history_outdir)
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
            # If we encounter columns that have, up until this point,
            # been NULL, serialize/deserialize them and update their
            # type in our cached Parquet schema
            new_keys = set(agent_data) - set(self.non_null_keys)
            if len(new_keys) > 0:
                new_key_data = orjson.loads(
                    orjson.dumps(
                        {k: agent_data[k] for k in new_keys},
                        option=orjson.OPT_SERIALIZE_NUMPY,
                        default=self.fallback_serializer,
                    )
                )
                for k, v in new_key_data.items():
                    pa_type, encoding, field_name, is_null = get_encoding(
                        v, k, k in USE_UINT16, k in USE_UINT32
                    )
                    if encoding is not None:
                        self.encodings[field_name] = encoding
                    if not is_null:
                        self.non_null_keys.add(k)
                    new_field = pa.field(k, pa_type)
                    field_index = self.schema.get_field_index(k)
                    if field_index >= 0:
                        self.schema = self.schema.set(field_index, new_field)
                    else:
                        self.schema = self.schema.append(new_field)
            self.temp_data.write(
                orjson.dumps(
                    agent_data,
                    option=orjson.OPT_SERIALIZE_NUMPY,
                    default=self.fallback_serializer,
                )
            )
            self.temp_data.write("\n".encode("utf-8"))
        self.num_emits += 1
        if self.num_emits % self.batch_size == 0:
            self.temp_data.close()
            outfile = os.path.join(
                self.outdir,
                self.experiment_id,
                "history",
                self.partitioning_path,
                f"{self.num_emits}.pq",
            )
            self.executor.submit(
                json_to_parquet,
                self.temp_data.name,
                self.encodings,
                self.schema,
                outfile,
                self.filesystem,
            )
            self.temp_data = tempfile.NamedTemporaryFile(delete=False)
