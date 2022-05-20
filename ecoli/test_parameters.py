from ecoli.parameters import Parameter, ParameterStore


def test_simple():
    param_dict = {
        'a': {
            'b': Parameter(
                1,
            ),
        },
    }
    param_store = ParameterStore(param_dict)
    assert param_store.get(('a', 'b')) == 1


def test_canonicalize():
    param_dict = {
        'a': {
            'b': Parameter(
                1,
                canonicalize=lambda x: x * 2
            ),
        },
    }
    param_store = ParameterStore(param_dict)
    assert param_store.get(('a', 'b')) == 2


def test_derivation():
    param_dict = {
        'a': {
            'b': Parameter(
                1,
            ),
        },
    }
    derivation_rules = {
        ('a', 'c'): lambda params: Parameter(
            params.get(('a', 'b')) * 2,
            canonicalize=lambda x: x * 3,
        )
    }
    param_store = ParameterStore(param_dict, derivation_rules)
    assert param_store.get(('a', 'c')) == 6
