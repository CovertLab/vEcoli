from vivarium.core.process import Step
from vivarium.library.topology import assoc_path, get_in


class Aggregator(Step):
    """
    Given a list of paths and a list of functions, this will apply the ith
    function to the ith path and write the results through `aggregated`.
    If a list of functions is not supplied, len is used for all paths.
    """

    name = "aggregator"
    defaults: dict[str, tuple] = {"paths": tuple(), "funcs": tuple()}

    def __init__(self, parameters):
        super().__init__(parameters)
        self.paths = self.parameters["paths"]
        if not self.parameters["funcs"]:
            self.funcs = (len,) * len(self.paths)
        else:
            self.funcs = self.parameters["funcs"]

    def ports_schema(self):
        schema = {}
        variables = []
        for path, func in zip(self.paths, self.funcs):
            assoc_path(schema, path, {"_default": {}})
            variable = f"{path[-1]}_{func.__name__}"
            assert variable not in variables
            variables.append(variable)
        assert "aggregated" not in schema
        schema["aggregated"] = {
            variable: {
                "_default": 0,
                "_divider": "zero",
                "_updater": "set",
                "_emit": True,
            }
            for variable in variables
        }
        return schema

    def next_update(self, timestep, states):
        counts = {}
        for path, func in zip(self.paths, self.funcs):
            variable = f"{path[-1]}_{func.__name__}"
            assert variable not in counts
            counts[variable] = func(get_in(states, path))
            assert counts[variable] is not None
        return {"aggregated": counts}


def len_squared(x):
    return len(x) ** 2


def len_plus_one(x):
    return len(x) + 1


def test_aggregator():
    state = {
        "a": {
            "b": {
                1: 0,
                2: 0,
                3: 0,
            },
            "c": {},
        },
    }
    proc = Aggregator(
        {
            "paths": (("a", "b"), ("a", "c"), ("a", "b")),
            "funcs": (len_squared, len_plus_one, len_plus_one),
        }
    )
    schema = proc.get_schema()
    expected_schema = {
        "a": {
            "b": {
                "_default": {},
            },
            "c": {
                "_default": {},
            },
        },
        "aggregated": {
            "b_len_squared": {
                "_default": 0,
                "_divider": "zero",
                "_updater": "set",
                "_emit": True,
            },
            "c_len_plus_one": {
                "_default": 0,
                "_divider": "zero",
                "_updater": "set",
                "_emit": True,
            },
            "b_len_plus_one": {
                "_default": 0,
                "_divider": "zero",
                "_updater": "set",
                "_emit": True,
            },
        },
    }
    assert schema == expected_schema
    update = proc.next_update(0, state)
    expected_update = {
        "aggregated": {
            "b_len_squared": 9,
            "c_len_plus_one": 1,
            "b_len_plus_one": 4,
        }
    }
    assert update == expected_update


if __name__ == "__main__":
    test_aggregator()
