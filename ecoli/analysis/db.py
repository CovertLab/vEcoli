import copy
import itertools
from bson import MinKey, MaxKey
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from vivarium.core.emitter import (
    data_from_database,
    DatabaseEmitter,
    assemble_data,
    get_local_client,
    get_data_chunks
)
from vivarium.library.dict_utils import deep_merge

from ecoli.library.sim_data import LoadSimData


def access(
        experiment_id, query=None, host='localhost', port=27017,
        func_dict=None, f=None, sampling_rate=1):
    config = {
        'host': '{}:{}'.format(host, port),
        'database': 'simulations'}
    emitter = DatabaseEmitter(config)
    db = emitter.db

    filters={'data.time': {'$mod': [sampling_rate, 0]}}
    data, sim_config = data_from_database(
        experiment_id, db, query, func_dict, f, filters)

    return data, experiment_id, sim_config


def get_agent_ids(experiment_id, host='localhost', port=27017):
    config = {
        'host': '{}:{}'.format(host, port),
        'database': 'simulations'}
    emitter = DatabaseEmitter(config)
    db = emitter.db

    result = db.history.aggregate([
        {'$match': {'experiment_id': experiment_id}},
        {'$project': {'agents': {'$objectToArray': '$data.agents'}}},
        {'$project': {'agents.k': 1, '_id': 0}},
    ])
    agents = set()
    for document in result:
        assert list(document.keys()) == ['agents']
        for sub_document in document['agents']:
            assert list(sub_document.keys()) == ['k']
            agents.add(sub_document['k'])
    return agents


def get_aggregation(host, port, aggregation):
    """Helper function for parallel aggregations"""
    history_collection = get_local_client(host, port, 'simulations').history
    return list(history_collection.aggregate(
        aggregation, hint={'experiment_id':1, 'data.time':1, '_id':1}))


def access_counts(experiment_id, monomer_names, mrna_names, 
    host='localhost', port=27017, sampling_rate=1, start_time=MinKey(),
    end_time=MaxKey(), cpus=1
):
    config = {
        'host': f'{host}:{port}',
        'database': 'simulations'
    }
    emitter = DatabaseEmitter(config)
    db = emitter.db

    # Retrieve and re-assemble experiment config
    experiment_query = {'experiment_id': experiment_id}
    experiment_config = db.configuration.find(experiment_query)
    experiment_assembly = assemble_data(experiment_config)
    assert len(experiment_assembly) == 1
    assembly_id = list(experiment_assembly.keys())[0]
    experiment_config = experiment_assembly[assembly_id]['metadata']
    # Load sim_data using parameters from experiment_config
    rnai_data = experiment_config['process_configs'].get(
        'ecoli-rna-interference')
    sim_data = LoadSimData(
        sim_data_path=experiment_config['sim_data_path'],
        seed=experiment_config['seed'],
        mar_regulon=experiment_config['mar_regulon'],
        rnai_data=rnai_data)

    aggregation = [{'$match': {
        **experiment_query,
        'data.time': {'$gte': start_time, '$lte': end_time, 
                      '$mod': [sampling_rate, 0]}}}]
    aggregation.append({'$project': {
        'data.agents': {'$objectToArray': '$data.agents'},
        'data.time': 1,
        'data.fields': 1,
        'data.dimensions': 1,
        'assembly_id': 1,
        }})
    monomer_idx = sim_data.get_monomer_counts_indices(monomer_names)
    projection = {
        '$project': {
            f'data.agents.v.bulk.{monomer}': {
                # Flatten array of length 1 into single count
                '$reduce': {
                    # Get monomer count at specified index with $slice
                    'input': {'$slice': [
                        # $objectToArray makes all embedded document fields 
                        # into arrays so we flatten here before slicing
                        {'$reduce': {
                            'input': '$data.agents.v.listeners.monomer_counts',
                            'initialValue': None,
                            'in': '$$this'
                        }},
                        monomer_index,
                        1
                    ]},
                    'initialValue': None,
                    'in': '$$this'
                }
            }
            for monomer, monomer_index in zip(monomer_names, monomer_idx)
        }
    }
    mrna_idx = sim_data.get_mrna_counts_indices(mrna_names)
    projection['$project'].update({
        f'data.agents.v.unique.{mrna}': {
            # Flatten array of length 1 into single count
            '$reduce': {
                # Get monomer count at specified index with $slice
                'input': {'$slice': [
                    # $objectToArray makes all embedded document fields 
                    # into arrays so we flatten here before slicing
                    {'$reduce': {
                        'input': '$data.agents.v.listeners.mRNA_counts',
                        'initialValue': None,
                        'in': '$$this'
                    }},
                    mrna_index,
                    1
                ]},
                'initialValue': None,
                'in': '$$this'
            }
        }
        for mrna, mrna_index in zip(mrna_names, mrna_idx)
    })
    projection['$project']['data.agents.v.listeners.mass'] = 1
    projection['$project']['data.agents.v.boundary'] = 1
    projection['$project']['data.agents.k'] = 1
    projection['$project']['data.time'] = 1
    projection['$project']['data.fields'] = 1
    projection['$project']['data.dimensions'] = 1
    projection['$project']['assembly_id'] = 1
    aggregation.append(projection)

    aggregation.append({'$project': {
        'data.agents': {'$arrayToObject': '$data.agents'},
        'data.time': 1,
        'data.fields': 1,
        'data.dimensions': 1,
        'assembly_id': 1,
    }})
    
    if cpus > 1:
        chunks = get_data_chunks(
            db.history, experiment_id, start_time, end_time, cpus)
        aggregations = []
        for chunk in chunks:
            agg_chunk = copy.deepcopy(aggregation)
            agg_chunk[0]['$match'] = {
                **experiment_query,
                '_id': {'$gte': chunk[0], '$lt': chunk[1]},
                'data.time': {'$gte': start_time, '$lte': end_time, 
                              '$mod': [sampling_rate, 0]}
            }
            aggregations.append(agg_chunk)
        partial_get_agg = partial(get_aggregation, host, port)
        with ProcessPoolExecutor(cpus) as executor:
            queried_chunks = executor.map(partial_get_agg, aggregations)
        result = itertools.chain.from_iterable(queried_chunks)
    else:
        result = db.history.aggregate(
            aggregation, 
            hint={'experiment_id':1, 'data.time':1, '_id':1})

    # re-assemble data
    assembly = assemble_data(list(result))

    # restructure by time
    data = {}
    for datum in assembly.values():
        time = datum['time']
        datum = datum.copy()
        datum.pop('_id', None)
        datum.pop('time', None)
        deep_merge(
            data,
            {time: datum},
        )

    return data
