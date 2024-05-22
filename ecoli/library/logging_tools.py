import os
import json

from vivarium.core.serialize import serialize_value


def make_logging_process(process_class):
    class LoggingProcess(process_class):
        def ports_schema(self):
            ports = super().ports_schema()  # original port structure
            ports["log_update"] = {
                "_default": {},
                "_updater": "set",
                "_emit": True,
            }  # add a new port
            return ports

        def next_update(self, timestep, states):
            update = super().next_update(timestep, states)  # original update
            log_update = {"log_update": update}  # log the update
            return {**update, **log_update}

    LoggingProcess.__name__ = f"Logging_{process_class.__name__}"
    return LoggingProcess


def write_json(path, numpy_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w") as outfile:
        json.dump(serialize_value(numpy_dict), outfile)
