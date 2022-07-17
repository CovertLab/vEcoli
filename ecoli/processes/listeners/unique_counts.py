from vivarium.core.process import Step
from vivarium.library.topology import assoc_path, get_in


class UniqueCounts(Step):

    name = 'unique_counts'
    defaults = {
        'paths': tuple()
    }

    def ports_schema(self):
        schema = {}
        variables = []
        for path in self.parameters['paths']:
            assoc_path(schema, path, {'_default': {}})
            assert path[-1] not in variables
            variables.append(path[-1])
        assert 'unique_counts' not in schema
        schema['unique_counts'] = {
            variable: {
                '_default': 0,
                '_divider': 'zero',
                '_updater': 'set',
                '_emit': True,
            }
            for variable in variables
        }
        return schema

    def next_update(self, timestep, states):
        counts = {}
        for path in self.parameters['paths']:
            variable = path[-1]
            assert variable not in counts
            counts[variable] = len(get_in(states, path))
            assert counts[variable] is not None
        return {'unique_counts': counts}


def test_unique_counts():
    state = {
        'a': {
            'b': {
                1: 0,
                2: 0,
                3: 0,
            },
            'c': {},
        },
    }
    proc = UniqueCounts({'paths': (
        ('a', 'b'),
        ('a', 'c'),
    )})
    schema = proc.get_schema()
    expected_schema = {
        'a': {
            'b': {
                '_default': {},
            },
            'c': {
                '_default': {},
            },
        },
        'unique_counts': {
            'b': {
                '_default': 0,
                '_divider': 'zero',
                '_updater': 'set',
                '_emit': True,
            },
            'c': {
                '_default': 0,
                '_divider': 'zero',
                '_updater': 'set',
                '_emit': True,
            },
        }
    }
    assert schema == expected_schema
    update = proc.next_update(0, state)
    expected_update = {
        'unique_counts': {
            'b': 3,
            'c': 0,
        }
    }
    assert update == expected_update
