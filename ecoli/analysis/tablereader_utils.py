from warnings import warn

import numpy as np


def warn_incomplete(array):
    warn("Used incomplete TableReader mapping, expect missing data!")
    return array


def replace_scalars(array):
    """Ensures that all elements in an input list of 1D arrays have same length
    and can be converted into a 2D array. Does so by replacing [] and scalar
    values with 1D list of zeroes with same length as others."""
    for value in array:
        if value != [] and type(value) in {list, np.ndarray}:
            array_len = len(value)
            break

    for i in range(len(array)):
        if array[i] == [] or type(array[i]) not in {list, np.ndarray}:
            array[i] = np.zeros(array_len)

    array = np.array(array)
    return array


def replace_scalars_2d(array):
    """Ensures that all elements in an input list of 2D arrays have same dims
    and can be converted into a 3D array. Does so by replacing [] and scalar
    values with 2D list of zeroes with same length as others."""
    for value in array:
        if value != [] and type(value) in {list, np.ndarray}:
            rows = len(value)
            cols = len(value[0])
            break

    for i in range(len(array)):
        if array[i] == [] or type(array[i]) not in {list, np.ndarray}:
            array[i] = np.zeros((rows, cols))

    array = np.array(array)
    return array


def camel_case_to_underscored(string):
    # Make sure first character is lowercase
    string = string[0].lower() + string[1:]
    
    # Find where words start/end
    capital_indices = np.where(np.array([c != c.lower() for c in string]))[0]
    word_limits = [0, *capital_indices, len(string)]

    # Get all of the words, in lowercase
    words = [string[word_start: word_end].lower()
             for word_start, word_end
             in zip(word_limits[:-1], word_limits[1:])]
    
    # return words joined with underscore
    return "_".join(words)
