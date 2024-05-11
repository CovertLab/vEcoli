"""
Cookbook of common Polars manipulations for analysis scripts.
"""

import pickle
from itertools import pairwise
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


METADATA_PREFIX = 'data__output_metadata__'
"""
In the config dataset, user-defined metadata for each store
(see :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.get_output_metadata`)
will be contained in columns with this prefix.
"""


def ndlist_to_ndarray(s: pl.Series) -> np.ndarray:
    """
    Convert a series consisting of nested lists with fixed dimensions into
    a Numpy ndarray. This should really only be necessary if you are trying
    to perform linear algebra (e.g. matrix multiplication, dot products).
    You can accomplish most other manipulations with Polars alone.
    """
    dimensions = [1, len(s)]
    while s.dtype == pl.List:
        s = s.explode()
        dimensions.append(len(s))
    dimensions = [p[1] // p[0] for p in pairwise(dimensions)]
    return s.to_numpy().reshape(dimensions)


def ndidx_to_pl_expr(name: str, idx: list[Any]) -> pl.Expr:
    """
    Returns a Polars expression that evaluates to the equivalent of converting
    the ``name`` column into an ndarray (see :py:func:`~.ndlist_to_ndarray`)
    and calling ``name_arr[idx]``. Note that, unlike in Numpy, ``idx`` can only
    contain 1D lists of integer indices (no nesting or bool masks) or ``":"``.

    This function is useful for reducing the amount of data loaded for columns
    with lists of 2 or more dimensions. For 1D list columns, the ability to
    individually name indices with :py:func:`~.named_idx` is probably better.

    If more advanced Numpy indexing strategies are required, you can use this
    function to filter as much as possible, :py:func:`~.ndlist_to_ndarray` to
    convert to a Numpy ndarray, and proceed with native Numpy indexing. 

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
    idx = idx.copy().reverse()
    # Construct expression from inside out (deepest to shallowest axis)
    pl_expr = pl.element().list.gather(idx.pop(0))
    for indices in idx[:-1]:
        # Skip gathering indices for dimensions where all are selected
        if indices == ':':
            pl_expr = pl.element().list.eval(pl_expr)
        else:
            pl_expr = pl.element().list.gather(indices).list.eval(pl_expr)
    if idx[-1] == ':':
        return pl.col(name).list.eval(pl_expr)
    return pl.col(name).list.gather(idx[-1]).list.eval(pl_expr)


def named_idx(col: str, names: list[str], idx: list[int]) -> dict[str, pl.Expr]:
    """
    Returns a mapping of column names to Polars expressions for
    given indices from each row of a list column.

    Args:
        col: Name of list column.
        names: Suffixes to append to ``col`` for final column names
        idx: For each suffix in ``names``, the index to get from each
            row of the ``col``
    
    Returns:
        Dictionary that maps ``{col}__{names[i]}`` to Polars expression for
        ``idx[i]`` element of each row in ``col``
    """
    return {
        f'{col}__{name}': pl.col('col').list.get(index)
        for name, index in zip(names, idx)
    }


def get_field_metadata(config_lf: pl.LazyFrame, field: str) -> pl.Series:
    """
    Gets the saved metadata for a given field as a Polars Series.

    Args:
        config_lf: LazyFrame of configuration data from
            :py:func:`~ecoli.library.parquet_emitter.get_lazyframes`
        field: Name of field to get metadata for
    """
    return config_lf.select(METADATA_PREFIX + field
        ).collect()[METADATA_PREFIX + field][0]


def plot(
    params: dict[str, Any],
    config_lf: pl.LazyFrame,
    history_lf: pl.LazyFrame,
    sim_data_path: str,
    validation_data_path: str
):
    """
    Template for analysis function with sample code for common operations.
    All analysis files should have a function called plot with the same args.

    Args:
        params: Parameters for analysis from config JSON
        config_lf: Polars LazyFrame containing configuration data
        history_lf: Polars LazyFrame containing simulation output
        sim_data_path: Path to sim_data pickle
        validation_data_path: Path to validation_data pickle
    """
    # Load sim data, validation data, neither, or both
    with open(sim_data_path, 'rb') as f:
        sim_data = pickle.load(f)
    with open(validation_data_path, 'rb') as f:
        validation_data = pickle.load(f)

    # Create filters for the data you want to read
    exp_id_filter = pl.col('experiment_id') == 'some_experiment_id'
    generation_range_filter = pl.col('generation').is_in(list(range(4,8)))
    mass_filter = pl.col('listeners__mass__dry_mass') < 300
    # Filters support logical &, ~, and |
    config_lf = config_lf.filter(exp_id_filter & generation_range_filter)
    history_lf = history_lf.filter(exp_id_filter
        & generation_range_filter & mass_filter)

    # Read values of interest from configs, such as field-specific metadata
    # Say we wanted to get the indices of certain bulk molecules in the
    # bulk counts array by name for agent_id ``01``
    molecules_of_interest = ['GUANOSINE-5DP-3DP[c]', 'WATER[c]', 'PROTON[c]']
    bulk_names = get_field_metadata(config_lf, 'bulk')
    bulk_idx = {}
    for mol in molecules_of_interest:
        bulk_idx[mol] = bulk_names.index(mol)


    # Select specific columns to read with a list of column names
    # Column names are just their tuple paths in the simulation Store
    # concatenated with double underscores
    simple_col_select = ['time', 'experiment_id']
    # Can also use kewword args to rename columns, perform scalar operations,
    # slice/index lists, and more using expressions
    advanced_col_projection = {
        'time_since_division': pl.col('time'),
        'cell_mass_picogram': pl.col('listeners__mass__cell_mass') / 1000,
        'rna_synth_prob': pl.col(
            'listeners__rna_synth_prob__actual_rna_synth_prob'),
        **named_idx('bulk', bulk_idx.keys(), bulk_idx.values())
    }
    history_lf = history_lf.select(simple_col_select, **advanced_col_projection)

    # NOTE: Must explicitly sort by ``time`` column or rows may be out of order
    history_lf = history_lf.sort('time')

    # When satisfied with your query, call ``collect`` on your LazyFrame
    history_df = history_lf.collect()
    
    # Polars offers many tools to filter, join, and transform data
    # Check their documentation to see if they have a function to
    # do what you want before converting to a Numpy array / Pandas DF
    rna_synth_prob = ndlist_to_ndarray(history_df['rna_synth_prob'])
    # Leverage the 2-D array to perform vectorized operations
    cistron_tu_mat = sim_data.process.transcription.cistron_tu_mapping_matrix
    rna_synth_prob_per_cistron = cistron_tu_mat.dot(rna_synth_prob.T).T

    # Now you can use matplotlib as usual or, if you kept your data
    # in a Polars dataframe, use their plotting capabilities
    plt.plot(history_df['time'], rna_synth_prob_per_cistron[0])

    # Anything that you want to save should be saved using relative file
    # paths. The absolute output directory is given as a CLI or JSON
    # config option to scripts/run_analysis.py
    plt.savefig('plots/test.svg')
    history_df.write_parquet('data/test_data.pq')
