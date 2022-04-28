from vivarium.core.emitter import (
    data_from_database,
    DatabaseEmitter,
)


def access(experiment_id, query=None, host='localhost', port=27017, f=None):
    config = {
        'host': '{}:{}'.format(host, port),
        'database': 'simulations'}
    emitter = DatabaseEmitter(config)
    db = emitter.db

    data, sim_config = data_from_database(
        experiment_id, db, query, f)

    return data, experiment_id, sim_config
