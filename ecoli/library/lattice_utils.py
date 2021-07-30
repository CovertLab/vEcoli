'''
==================================
Utilities for Lattice Environments
==================================
'''

from __future__ import absolute_import, division, print_function

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
