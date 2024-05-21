import numpy as np

from vivarium.core.composition import simulate_process
from vivarium.processes.timeline import TimelineProcess


class FieldTimeline(TimelineProcess):
    name = "field_timeline"
    defaults = {
        "timeline": [],
        "bins": [1, 1],
    }

    def __init__(self, parameters):
        bins = parameters.pop("bins", self.defaults["bins"])
        timeline = [
            (
                time,
                {
                    ("fields", molecule): np.full(bins, concentration, dtype=np.float64)
                    for molecule, concentration in change.items()
                },
            )
            for time, change in parameters.pop("timeline", self.defaults["timeline"])
        ]
        super().__init__(
            {
                "timeline": timeline,
                **parameters,
            }
        )


def test_field_timeline():
    process = FieldTimeline(
        {
            "timeline": [
                (3, {"mol": 5}),
                (5, {"mol": 0}),
            ],
            "bins": [5, 5],
            "_schema": {
                "fields": {
                    "mol": {
                        "_default": np.full([5, 5], 0),
                        "_emit": True,
                    },
                },
            },
        }
    )
    data = simulate_process(process, {"total_time": 7})

    expected_fields = []
    expected_fields.append([[0] * 5] * 5)
    expected_fields.append([[0] * 5] * 5)
    expected_fields.append([[0] * 5] * 5)
    expected_fields.append([[0] * 5] * 5)
    expected_fields.append([[5] * 5] * 5)
    expected_fields.append([[5] * 5] * 5)
    expected_fields.append([[0] * 5] * 5)
    expected_fields.append([[0] * 5] * 5)

    assert data["fields"]["mol"] == expected_fields


if __name__ == "__main__":
    test_field_timeline()
