import pickle

from pyarrow import compute as pc
from pyarrow import dataset as ds

from ecoli.library.parquet_emitter import get_datasets

METADATA_PREFIX = 'data__output_metadata__agents__'
"""
In the config dataset, user-defined metadata for each store
(see :py:meth:`~ecoli.experiments.ecoli_master_sim.EcoliSim.get_output_metadata`)
will be contained in columns with this prefix.
"""

def analysis_template(config, sim_data_path, validation_data_path):
    """
    Template for analysis function with sample code for common operations.
    """
    # Load sim data, validation data, neither, or both
    with open(sim_data_path, 'rb') as f:
        sim_data = pickle.load(f)
    with open(validation_data_path, 'rb') as f:
        validation_data = pickle.load(f)

    # Load Parquet datasets from output directory / URI specified in config
    emitter_config = config['emitter']['config']
    history_ds, config_ds = get_datasets(emitter_config.get('out_dir', None),
                                         emitter_config.get('out_uri', None))

    # Create filters for the data you want to read
    exp_id_filter = ds.field('experiment_id') == 'some_experiment_id'
    variant_filter = ds.field('variant') == 'some_variant_name'
    seed_filter = ds.field('seed') == 0
    generation_filer = ds.field('generation') == 2
    agent_filter = ds.field('agent_id') == '01'
    many_seed_filter = ds.field('seed').isin([1,2,3])
    seed_range_filter = ds.field('seed') <= 100
    # Filters support logical &, ~, and |
    layered_filter = exp_id_filter & variant_filter & seed_filter
    exclude_seeds = ~many_seed_filter
    combined_seeds = seed_filter | many_seed_filter

    # Make sure your preliminary filter condition above only includes the
    # experiment_id, variant, seed, generation, or agent_id columns. This
    # allows us to retrieve the configuration data for the specified set
    # of simulations.
    configs = config_ds.to_table(filter=layered_filter)

    # Read values of interest from configs, such as field-specific metadata
    # Say we wanted to get the indices of certain bulk molecules in the
    # bulk counts array by name for agent_id 01
    molecules_of_interest = ['GUANOSINE-5DP-3DP[c]', 'WATER[c]', 'PROTON[c]']
    bulk_names = configs.filter(ds.field('agent_id') == '01').column(
        METADATA_PREFIX + 'bulk')[0].values
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
        'time_since_division': ds.field('time'),
        'cell_mass_picogram': ds.field('listeners__mass__cell_mass') / 1000,
        'rna_synth_prob': ds.field(
            'listeners__rna_synth_prob__actual_rna_synth_prob'),
        'n_bound_TF_per_TU': ds.field(
            'listeners__rna_synth_prob__n_bound_TF_per_TU')
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
    # NOTE: Because PyArrow reads Parquet files with multiple threads
    # by default, we have to sort the table to guarantee the order
    data = history_ds.to_table(columns=advanced_col_projection,
        filter=layered_filter).sort_by(['experiment_id', 'variant', 'seed',
                                        'generation', 'agent_id', 'time'])
    
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
