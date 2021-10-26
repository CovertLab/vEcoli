import numpy as np
from ecoli import UNIQUE_DIVIDERS

def array_from(d):
    return np.array(list(d.values()))

def array_to(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)}

def array_to_nonzero(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)
        if array[index] != 0}

def type_of(array):
    if len(array) == 0:
        return None

    head = array[0]
    if isinstance(head, (list, np.ndarray)):
        return type_of(head)
    else:
        return type(head)

def arrays_from(ds, keys):
    if not ds:
        return np.array([])

    arrays = {
        key: []
        for key in keys}

    for d in ds:
        for key, value in d.items():
            if key in arrays:
                arrays[key].append(value)

    return tuple([
        np.array(array, dtype=type_of(array))
        for array in arrays.values()])

def arrays_to(n, attrs):
    ds = []
    for index in np.arange(n):
        d = {}
        for attr in attrs.keys():
            d[attr] = attrs[attr][index]
        ds.append(d)

    return ds

def bulk_schema(
        elements,
        updater=None,
        partition=True,
):
    schema = {
        '_default': 0,
        '_divider': 'binomial_ecoli',
        '_emit': True}
    if partition:
        schema['_properties'] = {'bulk': True}
    if updater:
        schema['_updater'] = updater
    return {
        element: schema
        for element in elements}

def mw_schema(mass_dict):
    return {
        element: {
            '_properties': {
                'mw': mw}}
        for element, mw in mass_dict.items()}

def listener_schema(elements):
    return {
        element: {
            '_default': default,
            '_updater': 'set',
            '_emit': True}
        for element, default in elements.items()}

def add_elements(elements, id):
    return {
        '_add': [{
            'key': str(element[id]),
            'state': element}
            for element in elements]}
    
def submass_schema():
    return {
        '_default': np.zeros(9),
        '_emit': True}

def dict_value_schema(name):
    return {
        '_default': {},
        '_updater': f'{name}_updater',
        '_divider': UNIQUE_DIVIDERS[name],
        '_emit': True
    }
