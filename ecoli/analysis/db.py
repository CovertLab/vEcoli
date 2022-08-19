from vivarium.core.emitter import (
    data_from_database,
    DatabaseEmitter,
)


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
