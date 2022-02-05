from vivarium.core.emitter import (
    data_from_database,
    DatabaseEmitter,
)


def access(experiment_id, query=None):
    config = {
        'host': '{}:{}'.format('localhost', 27017),
        'database': 'simulations'}
    emitter = DatabaseEmitter(config)
    db = emitter.db

    data, sim_config = data_from_database(
        experiment_id, db, query)

    return data, experiment_id, sim_config
