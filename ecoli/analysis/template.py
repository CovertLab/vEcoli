"""
Cookbook of common Polars manipulations for analysis scripts.
"""

import pickle
from itertools import pairwise
from typing import Any

import numpy as np
import polars as pl

from ecoli.library.parquet_emitter import get_lazyframes

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
    You can likely accomplish most other manipulations with Polars alone.
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
    and calling ``name_arr[idx]`` (only integer indices or ``":"``). Should
    really only be used for columns with lists of 2+ dimensions. To select
    elements from an unnested (1D) list column, prefer :py:func:`~.named_idx`. 

    Args:
        name: Name of column to recursively index
        idx: Sequence of indices in order, one for each dimension to index. To
            get all elements for a dimension, supply the string ``":"``.
            Here are some examples::

                [0, 1] # First row, second column
                [[0, 1], 1] # First and second row, second column
                [0, 1, ":"] # First element of axis 1, second of 2, all of 3

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
    Returns a mapping of column names to Polars expressions that get
    given indices from each row of a list column.

    Args:
        col: Name of list column.
        names: Suffixes to append to ``col`` for final column names
        idx: For each suffix in ``names``, the index to get from each
            row of the ``col``
    
    Returns:
        Dictionary that maps ``{col}__{names[i]}`` to Polars expression for
        ``idx[i]``th element of each row in ``col``
    """
    return {
        f'{col}__{name}': pl.col('col').list.get(index)
        for name, index in zip(names, idx)
    }


def analysis_template(config, sim_data_path, validation_data_path):
    """
    Template for analysis function with sample code for common operations.
    """
    # Load sim data, validation data, neither, or both
    with open(sim_data_path, 'rb') as f:
        sim_data = pickle.load(f)
    with open(validation_data_path, 'rb') as f:
        validation_data = pickle.load(f)

    # Load Parquet files from output directory / URI specified in config
    emitter_config = config['emitter']['config']
    config_lf, history_lf = get_lazyframes(
        emitter_config.get('out_dir', None),
        emitter_config.get('out_uri', None))

    # Create filters for the data you want to read
    exp_id_filter = pl.col('experiment_id') == 'some_experiment_id'
    generation_range_filter = pl.col('generation').is_in(list(range(4,8)))
    mass_filter = pl.col('listeners__mass__dry_mass') < 300
    # Filters support logical &, ~, and |
    # Construct your query step-by-step and call ``collect`` when ready.
    config_lf = config_lf.filter(exp_id_filter & generation_range_filter)
    history_lf = history_lf.filter(exp_id_filter | mass_filter)

    # Read values of interest from configs, such as field-specific metadata
    # Say we wanted to get the indices of certain bulk molecules in the
    # bulk counts array by name for agent_id 01
    molecules_of_interest = ['GUANOSINE-5DP-3DP[c]', 'WATER[c]', 'PROTON[c]']
    bulk_names = config_lf.select(METADATA_PREFIX + 'bulk'
        ).collect()[METADATA_PREFIX + 'bulk'][0]
    indices_of_interest = {}
    for mol in molecules_of_interest:
        indices_of_interest[mol] = bulk_names.index(mol)


    # Select specific columns to read with a list of column names
    # Column names are just their tuple paths in the simulation Store
    # concatenated with double underscores
    simple_col_select = ['time', 'experiment_id', 'listeners__mass__cell_mass']
    # Can also use projection to rename columns, perform scalar operations,
    # slice/index lists, etc using expressions composed of fields
    advanced_col_projection = {
        'time': ds.field('time'),
        'experiment_id': ds.field('experiment_id'),
        'variant': ds.field('variant'),
        'seed': ds.field('seed'),
        'generation': ds.field('generation'),
        'agent_id': ds.field('agent_id'),
        'cell_mass_picogram': ds.field('listeners__mass__cell_mass') / 1000,
        'rna_synth_prob': ds.field(
            'listeners__rna_synth_prob__actual_rna_synth_prob'),
        'n_bound_TF_per_TU': ds.field(
            'listeners__rna_synth_prob__n_bound_TF_per_TU'),
        **{
            mol: pc.list_element(ds.field('bulk'), idx)
            for mol, idx in indices_of_interest.items()
        }
    }

    # Retrieve data with projections/column selections/filters
    # NOTE: Use the same variable name where possible so Python can free
    # memory automatically. For example, converting an Arrow table to a Pandas
    # DataFrame involves a copy, after which the Arrow table will be
    # automatically freed as long as there are no variables referencing it.
    # NOTE: We have to sort the table to guarantee the order
    data = history_ds.to_table(columns=advanced_col_projection,
        ).sort_by([('experiment_id', 'ascending'), ('variant', 'ascending'),
                   ('seed', 'ascending'), ('generation', 'ascending'), 
                   ('agent_id', 'ascending'), ('time', 'ascending')])
    
    # PyArrow offers many functions for manipulating tables and columns.
    # Refer to the documentation for pyarrow.compute and pyarrow.Table.
    # PyArrow tables can also be converted into Pandas DataFrames and columns
    # can be converted into Numpy arrays.
    # NOTE: PyArrow columns of lists (e.g. RNA synthesis probabilities, etc)
    # are converted into Pandas object columns or Numpy object arrays with each
    # element being a Numpy array. For lists that always have a fixed size,
    # we usually would prefer if these columns were converted into ndarrays.
    # This can be done as shown below.
    rna_synth_prob = data['rna_synth_prob'].chunks[0].flatten().to_numpy(
        ).reshape(-1, len(data['rna_synth_prob'][0]))
    # For lists with 2-D arrays for each element, call flatten twice and add
    # the extra dimension when reshaping
    n_bound_TF_per_TU = data['n_bound_TF_per_TU'].chunks[0].flatten().flatten(
        ).to_numpy().reshape(
            -1,
            len(data['n_bound_TF_per_TU'][0]),
            len(data['n_bound_TF_per_TU'][0][0]))
    # Leverage the 2-D array to perform vectorized operations
    cistron_tu_mat = sim_data.process.transcription.cistron_tu_mapping_matrix
    rna_synth_prob_per_cistron = cistron_tu_mat.dot(rna_synth_prob.T).T

    # Now you can use matplotlib as usual
