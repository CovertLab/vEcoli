import os
import json

from vivarium.core.serialize import serialize_value


def make_logging_process(process_class):
    logging_process = type(f"Logging_{process_class.__name__}",
                           (process_class,),
                           {})
    __class__ = logging_process  # set __class__ manually so super() knows what to do

    def ports_schema(self):
        ports = super().ports_schema()  # get the original port structure
        ports['log_update'] = {'_default' : {}, '_updater': 'set', '_emit': True}  # add a new port
        return ports

    def next_update(self, timestep, states):
        update = super().next_update(timestep, states)  # get the original update
        log_update = {'log_update' : update}  # log the update
        return {**update, **log_update}

    logging_process.ports_schema = ports_schema
    logging_process.next_update = next_update

    return logging_process


def write_json(path, numpy_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as outfile:
        json.dump(serialize_value(numpy_dict), outfile)
