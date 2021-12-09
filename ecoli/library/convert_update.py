'''
================================
Utilities for Converting Updates
================================
'''

import numpy as np


def convert_numpy_to_builtins(update):
    '''Convert an update to use Python builtins instead of Numpy types

    Convert the following:

    * np.int64 to int
    * np.str_ to str
    * np.float64 to float

    Args:
        update: The update to convert. Will not be modified.

    Returns:
        The converted update.
    '''
    if isinstance(update, np.int64):
        return int(update)
    if isinstance(update, np.str_):
        return str(update)
    if isinstance(update, np.float64):
        return float(update)
    if isinstance(update, dict):
        return {
            convert_numpy_to_builtins(key):
            convert_numpy_to_builtins(val)
            for key, val in update.items()
        }
    return update


def assert_types_equal(d1, d2):
    assert type(d1) == type(d2)
    if isinstance(d1, dict):
        for key1, key2 in zip(sorted(d1), sorted(d2)):
            assert type(key1) == type(key2)
            assert_types_equal(d1[key1], d2[key2])


def test_convert_numpy_to_builtins():
    update = {
        'a': {
            'b': 5,
            'c': np.float64(3.14),
            'd': np.int64(5),
        },
        np.str_('e'): {
            'f': True,
            'g': None,
            'h': np.str_('hi'),
        },
    }
    converted = convert_numpy_to_builtins(update)
    expected = {
        'a': {
            'b': 5,
            'c': 3.14,
            'd': 5,
        },
        'e': {
            'f': True,
            'g': None,
            'h': 'hi',
        },
    }
    assert converted == expected
    assert_types_equal(converted, expected)
