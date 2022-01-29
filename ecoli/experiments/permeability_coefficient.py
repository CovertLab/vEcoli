"""
This file calculates the porin-concentration permeability coefficients for
ompC and ompF when diffusing cephaloridine and the average surface area of the cell.
These figures are used as parameters in ecoli/processes/antibiotics/porin_permeability.py.
"""
from vivarium.core.emitter import (
        data_from_database, get_local_client, timeseries_from_data)

EXPERIMENT_ID = '9cc838ec-7d6e-11ec-b2e8-1e00312eb299'  # shape.json config for 2687 seconds
OMPC_PERMEABILITY_COEFFICIENT = 4.5
OMPF_PERMEABILITY_COEFFICIENT = 52.6


def main():
    data, conf = data_from_database(EXPERIMENT_ID,
                                    get_local_client("localhost", "27017", "simulations"),
                                    query=[('bulk', 'CPLX0-7533[o]'),
                                           ('bulk', 'CPLX0-7534[o]'),
                                           ('boundary', 'surface_area')])
    data = timeseries_from_data(data)

    sa_sum = 0
    ompc_sum = 0
    ompf_sum = 0
    sa_len = len(data['boundary']['surface_area'])
    ompc_len = len(data['bulk']['CPLX0-7533[o]'])
    ompf_len = len(data['bulk']['CPLX0-7534[o]'])
    for i in range(sa_len):
        sa_sum += data['boundary']['surface_area'][i]
    for i in range(ompc_len):
        ompc_sum += data['bulk']['CPLX0-7533[o]'][i]
    for i in range(ompf_len):
        ompf_sum += data['bulk']['CPLX0-7534[o]'][i]
    sa_average = sa_sum / sa_len
    ompc_average = ompc_sum / ompc_len  # ompc porin count about 50,000 halfway through on ecocyc
    ompf_average = ompf_sum / ompf_len  # ompf porin count about 71,798 halfway through on ecocyc

    ompc_concentration = ompc_average / sa_average
    ompf_concentration = ompf_average / sa_average

    ompc_con_perm = OMPC_PERMEABILITY_COEFFICIENT / ompc_concentration  # porin concentration permeability coefficient
    ompf_con_perm = OMPF_PERMEABILITY_COEFFICIENT / ompf_concentration

    print("Average surface area: " + str(sa_average))
    print("ompC porin-concentration permeability coefficient: " + str(ompc_con_perm))
    print("ompF porin-concentration permeability coefficient: " + str(ompf_con_perm))


if __name__ == '__main__':
    main()
