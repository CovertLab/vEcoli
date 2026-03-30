import os
import json

from fsspec import open as fsspec_open
from vivarium.core.serialize import serialize_value

from wholecell.utils.filepath import is_cloud_uri


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
    """Write JSON state to local file or cloud URI (s3://, gs://)."""
    if not is_cloud_uri(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with fsspec_open(path, "w") as outfile:
        json.dump(serialize_value(numpy_dict), outfile)
