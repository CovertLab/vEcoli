import os
import json
import numpy as np
from scipy import constants
from copy import deepcopy

from vivarium.core.serialize import deserialize_value
from vivarium.core.emitter import timeseries_from_data
from vivarium.library.units import remove_units

from ecoli.analysis.analyze_db_experiment import access

# Parameters to calc
# Cell mass: ('listeners', 'mass', 'cell_mass',)
# Dry mass: ('listeners', 'mass', 'dry_mass',)
# Cell volume: ('listeners', 'mass', 'volume',)
# Surface area: ('boundary', 'surface_area',)
# AcrAB-TolC conc.: ('bulk', 'TRANS-CPLX-201[m]',)
# Beta-lactamase conc.: ('bulk', 'EG10040-MONOMER[p]',)
# OmpC: ('bulk', 'CPLX0-7533[o]',)
# OmpF: ('bulk', 'CPLX0-7534[o]',)

def retrieve_data(exp_id, paths):
    data, _, _ = access(exp_id, query=paths)
    data = deserialize_value(data)
    data = remove_units(data)
    data = timeseries_from_data(data)
    return data

def calculate_avg(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = calculate_avg(data[key])
    else:
        data = np.mean(data)
    return data

if __name__ == "__main__":
    paths = [
        ('agents', '0', 'listeners', 'mass', 'cell_mass',),
        ('agents', '0', 'listeners', 'mass', 'dry_mass',),
        ('agents', '0', 'listeners', 'mass', 'volume',),
        ('agents', '0', 'boundary', 'surface_area',),
        ('agents', '0', 'bulk', 'TRANS-CPLX-201[m]',),
        ('agents', '0', 'bulk', 'EG10040-MONOMER[p]',),
        ('agents', '0', 'bulk', 'CPLX0-7533[o]',),
        ('agents', '0', 'bulk', 'CPLX0-7534[o]',),
    ]
    data = retrieve_data("571cc5a8-fe31-11ec-a129-9cfce8b9977c", paths)
    data_copy = deepcopy(data)
    avgs = calculate_avg(data_copy)
    pump = np.array(
        data['agents']['0']['bulk']['TRANS-CPLX-201[m]']
        ) / constants.N_A * 1E3
    periplasm_vol = np.array(
        data['agents']['0']['listeners']['mass']['volume']) * 0.2 * 1E-15
    avgs['agents']['0']['bulk']['TRANS-CPLX-201[m]'] = np.mean(
        pump / periplasm_vol)
    b_lac = np.array(
        data['agents']['0']['bulk']['EG10040-MONOMER[p]']
        ) / constants.N_A * 1E3
    avgs['agents']['0']['bulk']['EG10040-MONOMER[p]'] = np.mean(
        b_lac / periplasm_vol)
    avgs.pop('time')
    if os.path.exists('out/'):
        with open('out/avg_params.json', 'w') as f:
            json.dump(avgs, f)
