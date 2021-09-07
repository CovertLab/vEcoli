import os
import json
import unum
import numpy as np
from vivarium.core.serialize import *
from vivarium.core.serialize import _serialize_list, _serialize_dictionary


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
        log_update = {'log_update' : update} # log the update
        return {**update, **log_update}

    logging_process.ports_schema = ports_schema
    logging_process.next_update = next_update

    return logging_process


def write_json(path, numpy_dict):
    INFINITY = float('inf')

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, unum.Unum):
                return obj.asNumber()
            elif isinstance(obj, set):
                return list(obj)
            elif obj == INFINITY:
                return '__INFINITY__'
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return super(NpEncoder, self).default(obj)

    # class VivariumSerializer(json.JSONEncoder):
    #     def default(self, value):
    #         if isinstance(value, unum.Unum):
    #             return value.asNumber()
    #         serialize_value(value)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as outfile:
        json.dump(numpy_dict, outfile, cls=NpEncoder)
