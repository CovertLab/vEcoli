"""
Cookbook of common DuckDB manipulations for analysis scripts.
"""

import os
import pickle
from itertools import pairwise
from typing import Any, TYPE_CHECKING

import duckdb
import pyarrow as pa
from pyarrow import compute as pc
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from reconstruction.ecoli.simulation_data import SimulationDataEcoli


METADATA_PREFIX = "data__output_metadata__"
"""
In the config dataset, user-defined metadata for each store
(see :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.get_output_metadata`)
will be contained in columns with this prefix.
"""


def num_cells(config_num_cells: duckdb.DuckDBPyRelation) -> int:
    """
    Return cell count in filtered DuckDB relation (requires ``experiment_id``,
    ``variant``, ``lineage_seed``, ``generation``, and ``agent_id`` columns).
    """
    return duckdb.sql("""SELECT count(
        DISTINCT (experiment_id, variant, lineage_seed, generation, agent_id)
        ) AS m FROM config_num_cells""").arrow()["m"][0].as_py()


def ndlist_to_ndarray(s: pa.Array) -> np.ndarray:
    """
    Convert a series consisting of nested lists with fixed dimensions into
    a Numpy ndarray. This should really only be necessary if you are trying
    to perform linear algebra (e.g. matrix multiplication, dot products) inside
    a user-defined function (see DuckDB documentation on Python Function API).
    """
    dimensions = [1, len(s)]
    while isinstance(s.type, pa.ListType):
        s = pc.list_flatten(s)
        dimensions.append(len(s))
    dimensions = [p[1] // p[0] for p in pairwise(dimensions)]
    return s.to_numpy().reshape(dimensions)


def ndidx_to_pl_expr(name: str, idx: list[int | list[int | bool] | str]) -> str:
    """
    Returns a DuckDB expression for a column equivalent to converting each row
    of ``name`` into an ndarray ``name_arr`` (:py:func:`~.ndlist_to_ndarray`)
    and getting ``name_arr[idx]``. ``idx`` can contain 1D lists of integers,
    boolean masks, or ``":"`` (no 2D+ indices like x[[[1,2]]]).

    This function is useful for reducing the amount of data loaded for columns
    with lists of 2 or more dimensions. For list columns with one dimension,
    use :py:func:`~.named_idx` to create expressions for many indices at once.

    .. WARNING:: DuckDB arrays are 1-indexed!

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
    else:
        select_expr = f"x_0[{first_idx + 1}]"
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
        else:
            select_expr = f"list_transform(x_{i+1}[{indices + 1}], x_{i} -> {select_expr})"
    select_expr = select_expr.replace(f"x_{i+1}", name)
    return select_expr + f" AS {name}"


def named_idx(col: str, names: list[str], idx: list[int]) -> list[str]:
    """
    Create SQL SELECT expressions for given indices from a list column.

    .. WARNING:: DuckDB arrays are 1-indexed!

    Args:
        col: Name of list column.
        names: New column names, one for each index.
        idx: Indices to retrieve from ``col``

    Returns:
        List of strings that can be put after SELECT in DuckDB query
    """
    return [f"{col}[{i + 1}] AS \"{n}\"" for n, i in zip(names, idx)]


def get_field_metadata(
    config_field_metadata: duckdb.DuckDBPyRelation,
    field: str
) -> list:
    """
    Gets the saved metadata for a given field as a list.

    Args:
        config_lf: DuckDB relation of configuration data from
            :py:func:`~ecoli.library.parquet_emitter.get_duckdb_relation`
        field: Name of field to get metadata for
    """
    metadata = duckdb.sql(
        f"SELECT first({METADATA_PREFIX + field}) AS m FROM config_field_metadata"
    ).arrow()["m"][0]
    metadata_val = metadata.as_py()
    if isinstance(metadata_val, list):
        return metadata_val
    return list(metadata_val)


def get_config_value(
    config_get_value: duckdb.DuckDBPyRelation,
    config_opt: str
) -> Any:
    """
    Gets the saved configuration option.

    Args:
        config_lf: DuckDB relation of configuration data from
            :py:func:`~ecoli.library.parquet_emitter.get_duckdb_relation`
        field: Name of configuration option to get value pf
    """
    metadata = duckdb.sql(
        f"SELECT first(data__{config_opt}) AS m FROM config_get_value"
    ).arrow()["m"][0]
    return metadata.as_py()


def plot(
    params: dict[str, Any],
    configuration: duckdb.DuckDBPyRelation,
    history: duckdb.DuckDBPyRelation,
    sim_data_path: list[str],
    validation_data_path: list[str],
    outdir: str,
):
    """
    Template for analysis function with sample code for common operations.
    All analysis files should have a function called plot with the same args.

    Args:
        params: Parameters for analysis from config JSON
        configuration: DuckDB relation containing configuration data
        history: DuckDB relation containing simulation output
        sim_data_path: Path to sim_data pickle
        validation_data_path: Path to validation_data pickle
        outdir: Output directory
    """
    # Load sim data, validation data, neither, or both
    with open(sim_data_path, "rb") as f:
        sim_data: "SimulationDataEcoli" = pickle.load(f)

    # Create filters for the data you want to read
    data_filters = [
        "experiment_id = 'some_experiment_id'",
        f"generation IN {tuple(range(4, 8))}",
        "listeners__mass__dry_mass < 300"
    ]
    # Combine filters with AND, OR, NOT
    data_filters = " AND ".join(data_filters)

    # Read values of interest from configs, such as field-specific metadata
    # Say we wanted to get the indices of certain bulk molecules in the
    # bulk counts array by name for agent_id ``01``
    molecules_of_interest = ["GUANOSINE-5DP-3DP[c]", "WATER[c]", "PROTON[c]"]
    bulk_names = get_field_metadata(configuration, "bulk")
    bulk_idx = {}
    for mol in molecules_of_interest:
        bulk_idx[mol] = bulk_names.index(mol)

    # Use SELECT statement to select and perform calculations on columns
    selected_history = duckdb.sql(f"""
        SELECT
            -- Simple selection (comments in SQL are marked with double dash)
            time, experiment_id,
            -- Can perform calculations and give columns aliases with AS
            time / 60 AS minutes_since_division,
            listeners__mass__cell_mass / 1000 as cell_mass_picogram,
            listeners__rna_synth_prob__actual_rna_synth_prob as rna_synth_prob,
            {ndidx_to_pl_expr("listeners__rna_synth_prob__n_bound_TF_per_TU", [[1,2], [3,4]])},
            {", ".join(named_idx("bulk", bulk_idx.keys(), bulk_idx.values()))}
        FROM history
        -- Apply filter using WHERE clause
        WHERE {data_filters}
        -- MUST explicitly order by time
        ORDER BY time ASC
    """).arrow()

    # DuckDB offers many tools to filter, join, and transform data
    # Check their documentation to see if they have a function to
    # do what you want before converting to Numpy, Pandas, Polars, etc.
    rna_synth_prob = ndlist_to_ndarray(selected_history["rna_synth_prob"])
    # Leverage the 2-D array to perform vectorized operations
    cistron_tu_mat = sim_data.process.transcription.cistron_tu_mapping_matrix
    rna_synth_prob_per_cistron = cistron_tu_mat.dot(rna_synth_prob.T).T

    plt.plot(selected_history["minutes_since_division"], rna_synth_prob_per_cistron[0])

    # Anything that you want to save should be saved using relative file
    # paths. The absolute output directory is given as a CLI or JSON
    # config option to scripts/run_analysis.py
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)
    plt.savefig(os.path.join(outdir, "plots/test.svg"))
