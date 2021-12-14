'''
==================================
Utilities for Lattice Environments
==================================
'''

import numpy as np
from scipy import constants

from vivarium.library.units import units, Quantity

AVOGADRO = constants.N_A / units.mol


def get_bin_site(location, n_bins, bounds):
    '''Get a bin's indices in the lattice

    Parameters:
        location (list): A list of 2 floats that specify the x and y
            coordinates of a point inside the desired bin.
        n_bins (list): A list of 2 ints that specify the number of bins
            along the x and y axes, respectively.
        bounds (list): A list of 2 floats that define the dimensions of
            the lattice environment along the x and y axes,
            respectively.

    Returns:
        tuple: A 2-tuple of the x and y indices of the bin in the
        lattice.
    '''
    bin_site_no_rounding = np.array([
        location[0] * n_bins[0] / bounds[0],
        location[1] * n_bins[1] / bounds[1]
    ])
    bin_site = tuple(
        np.floor(bin_site_no_rounding).astype(int) % n_bins)
    return bin_site


def get_bin_volume(n_bins, bounds, depth):
    '''Get a bin's volume

    Parameters:
        n_bins (list): A list of 2 ints that specify the number of bins
            along the x and y axes, respectively.
        bounds (list): A list of 2 floats that specify the lengths of
            the environment's sides in the x and y directions,
            respectively. In units of microns.
        depth (float): The depth of the environment, in microns.

    Returns:
        float: The volume of each bin in the lattice, in Liters.
    '''
    total_volume = (depth * bounds[0] * bounds[1]) * 1e-15  # (L)
    return total_volume / (n_bins[0] * n_bins[1])


def count_to_concentration(count, bin_volume):
    '''Convert a molecule count into a concentration.

    Parameters should all have units. Returned value will have units.

    Parameters:
        count (int): The number of molecules in the bin.
        bin_volume (float): The volume of the bin.

    Returns:
        float: The concentration of molecule in the bin.
    '''
    return count / (bin_volume * AVOGADRO)


def make_gradient(gradient, n_bins, size):
    '''Create a gradient from a configuration

    **Random**
    A random gradient fills the field randomly with each molecule,
    with values between 0 and the concentrations specified.

    Example configuration:

    .. code-block:: python

        'gradient': {
            'type': 'random',
            'molecules': {
                'mol_id1': 1.0,
                'mol_id2': 2.0
            }},

    **Uniform**

    A uniform gradient fills the field evenly with each molecule, at
    the concentrations specified.

    Example configuration:

    .. code-block:: python

        'gradient': {
            'type': 'uniform',
            'molecules': {
                'mol_id1': 1.0,
                'mol_id2': 2.0
            }},

    **Gaussian**

    A gaussian gradient multiplies the base concentration of the given
    molecule by a gaussian function of distance from center and
    deviation. Distance is scaled by 1/1000 from microns to millimeters.

    Example configuration:

    .. code-block:: python

        'gradient': {
            'type': 'gaussian',
            'molecules': {
                'mol_id1':{
                    'center': [0.25, 0.5],
                    'deviation': 30},
                'mol_id2': {
                    'center': [0.75, 0.5],
                    'deviation': 30}
            }},

    **Linear**

    A linear gradient sets a site's concentration (c) of the given
    molecule as a function of distance (d) from center and slope (b),
    and base concentration (a). Distance is scaled by 1/1000 from
    microns to millimeters.

    .. math::
        c = a + b * d

    Example configuration:

    .. code-block:: python

        'gradient': {
            'type': 'linear',
            'molecules': {
                'mol_id1':{
                    'center': [0.0, 0.0],
                    'base': 0.1,
                    'slope': -10},
                'mol_id2': {
                    'center': [1.0, 1.0],
                    'base': 0.1,
                    'slope': -5}
            }},

    **Exponential**

    An exponential gradient sets a site's concentration (c) of the given
    molecule as a function of distance (d) from center, with parameters
    base (b) and scale (a). Distance is scaled by 1/1000 from microns to
    millimeters. Note: base > 1 makes concentrations increase from the
    center.

    .. math::

        c=a*b^d.

    Example configuration:

    .. code-block:: python

        'gradient': {
            'type': 'exponential',
            'molecules': {
                'mol_id1':{
                    'center': [0.0, 0.0],
                    'base': 1+2e-4,
                    'scale': 1.0},
                'mol_id2': {
                    'center': [1.0, 1.0],
                    'base': 1+2e-4,
                    'scale' : 0.1}
            }},

    Parameters:
        gradient: Configuration dictionary that includes the ``type``
            key to specify the type of gradient to make.
        n_bins: A list of two elements that specify the number of bins
            to have along each axis.
        size: A list of two elements that specifies the size of the
            environment.
    '''
    bins_x = n_bins[0]
    bins_y = n_bins[1]
    length_x = size[0]
    length_y = size[1]
    fields = {}

    if gradient.get('type') == 'random':
        for molecule_id, fill_value in gradient['molecules'].items():
            field = fill_value * np.random.rand(bins_x, bins_y)
            fields[molecule_id] = field

    if gradient.get('type') == 'gaussian':
        for molecule_id, specs in gradient['molecules'].items():
            field = np.ones((bins_x, bins_y), dtype=np.float64)
            center = [specs['center'][0] * length_x,
                      specs['center'][1] * length_y]
            deviation = specs['deviation']

            for x_bin in range(bins_x):
                for y_bin in range(bins_y):
                    # distance from middle of bin to center coordinates
                    dx = (x_bin + 0.5) * length_x / bins_x - center[0]
                    dy = (y_bin + 0.5) * length_y / bins_y - center[1]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    scale = gaussian(deviation, (distance/1000))
                    # multiply gradient by scale
                    field[x_bin][y_bin] *= scale
            fields[molecule_id] = field

    elif gradient.get('type') == 'linear':
        for molecule_id, specs in gradient['molecules'].items():
            field = np.zeros((bins_x, bins_y), dtype=np.float64)
            center = [specs['center'][0] * length_x,
                      specs['center'][1] * length_y]
            base = specs.get('base', 0.0)
            slope = specs['slope']

            for x_bin in range(bins_x):
                for y_bin in range(bins_y):
                    dx = (x_bin + 0.5) * length_x / bins_x - center[0]
                    dy = (y_bin + 0.5) * length_y / bins_y - center[1]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    field[x_bin][y_bin] += base + slope * (distance/1000)
            fields[molecule_id] = field

    elif gradient.get('type') == 'exponential':
        for molecule_id, specs in gradient['molecules'].items():
            field = np.zeros((bins_x, bins_y), dtype=np.float64)
            center = [specs['center'][0] * length_x,
                      specs['center'][1] * length_y]
            base = specs['base']
            scale = specs.get('scale', 1)

            for x_bin in range(bins_x):
                for y_bin in range(bins_y):
                    dx = (x_bin + 0.5) * length_x / bins_x - center[0]
                    dy = (y_bin + 0.5) * length_y / bins_y - center[1]
                    distance = np.sqrt(dx ** 2 + dy ** 2)
                    field[x_bin][y_bin] = scale * base ** (distance/1000)
            fields[molecule_id] = field

    elif gradient.get('type') == 'uniform':
        for molecule_id, fill_value in gradient['molecules'].items():
            fields[molecule_id] = np.full((bins_x, bins_y), fill_value, dtype=np.float64)

    return fields


def gaussian(deviation, distance):
    return np.exp(-np.power(distance, 2.) / (2 * np.power(deviation, 2.)))