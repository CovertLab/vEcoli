import numpy as np

def array_from(d):
    return np.array(list(d.values()))

def array_to(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)}

def arrays_from(ds, keys):
    arrays = {
        key: []
        for key in keys}

    for d in ds:
        for key, value in d.items():
            if key in arrays:
                arrays[key].append(value)

    return tuple([
        np.array(array)
        for array in arrays.values()])

def arrays_to(n, attrs):
    ds = []
    for index in np.arange(n):
        d = {}
        for attr in attrs.keys():
            d[attr] = attrs[attr][index]
        ds.append(d)

    return ds

def bulk_schema(elements):
    return {
        element: {
            '_default': 0,
            '_emit': True}
        for element in elements}

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
            'path': (element[id],),
            'state': element}
            for element in elements]}
