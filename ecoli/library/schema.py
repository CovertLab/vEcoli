import numpy as np

def array_from(d):
    return np.array(list(d.values()))

def arrays_from(ds, keys):
    arrays = {
        key: []
        for key in keys}

    for d in ds:
        for key, value in d.items():
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

