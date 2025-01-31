from vivarium.core.process import Process
from ecoli.library.schema import bulk_name_to_idx, numpy_schema, counts


class BulkTimelineProcess(Process):
    """Timeline process that works with bulk molecules in Numpy arrays."""

    name = "bulk-timeline"

    defaults = {"time_step": 1.0, "timeline": []}

    def __init__(self, parameters=None):
        super().__init__(parameters)
        # Sort the timeline, which should be in the format:
        # {time: {(path, to, store): new_value, ...}, ...}
        self.timeline = dict(sorted(self.parameters["timeline"].items()))
        # Get top-level store names from paths in timeline
        self.timeline_ports = [
            path[0] for events in self.timeline.values() for path in events.keys()
        ]

    def ports_schema(self):
        schema = {
            "bulk": numpy_schema("bulk"),
            "global": {"time": {"_default": 0, "_updater": "accumulate"}},
        }
        other_ports = {port: {} for port in self.timeline_ports}
        return {**schema, **other_ports}

    def next_update(self, timestep, states):
        time = states["global"]["time"]
        update = {"global": {"time": timestep}}
        new_timeline = self.timeline.copy()
        for t, change_dict in self.timeline.items():
            if time >= t:
                for path, change in change_dict.items():
                    if path[0] == "bulk":
                        idx = bulk_name_to_idx(path[-1], states["bulk"]["id"])
                        curr_count = counts(states["bulk"], idx)
                        update.setdefault("bulk", []).append((idx, change - curr_count))
                    else:
                        curr = update
                        for i, subpath in enumerate(path):
                            if subpath not in curr:
                                if i == len(path) - 1:
                                    curr[subpath] = {
                                        "_value": change,
                                        "_updater": "set",
                                    }
                                else:
                                    curr[subpath] = {}
                            else:
                                # Note: this does not catch the case where
                                # the timeline sets a branch before a leaf
                                # Example: {0: {(1,): {2: 3}, (1, 2): 4}}
                                # will result in the store at (1, 2) being
                                # set to 4 instead of 3 at t = 0
                                if i == len(path) - 1:
                                    raise Exception(
                                        "Timeline trying to set "
                                        f"value at branch {path[: i + 1]} and "
                                        f"its leaves at the same time"
                                    )
                            curr = curr[subpath]
                new_timeline.pop(t)
            # self.timeline is sorted so can break after first time < t
            break
        self.timeline = new_timeline
        return update
