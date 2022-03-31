'''Generate an SBML File for Bioscrape to Simulate

The SBML file defines a chemical reaction network (CRN) with following
reactions:

* Antibiotic influx via diffusion through porins.
* Antibiotic hydrolysis
* Antibiotic export by efflux pumps

The CRN can then be simulated by configuring
:py:class:`ecoli.processes.antibiotics.exchange_aware_bioscrape.ExchangeAwareBioscrape`
with the path to the generated SBML file.
'''

import os

from biocrnpyler import (
    ChemicalReactionNetwork,
    ParameterEntry,
    ProportionalHillPositive,
    GeneralPropensity,
    Reaction,
    Species,
)
from vivarium.library.units import units


DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data'))
# Calculated by dividing V_max reported in (Nagano & Nikaido, 2009) by the model's initial pump concentration of
# 20.179269875115253 counts / micron^2
CEPH_PUMP_KCAT = 0.0956090147363198  # / units.sec
# Reported in (Nagano & Nikaido, 2009)
CEPH_PUMP_KM = 288e-3  # * units.millimolar
# Reported in (Galleni et al., 1988)
CEPH_BETA_LACTAMASE_KCAT = 130  # / units.sec
# Reported in (Galleni et al., 1988)
CEPH_BETA_LACTAMASE_KM = 170  # * units.micromolar


# A ProportionalHillPositive propensity generates a propensity:
#
#   p = k*d*s^n/(s^n + K)
#
# Michaelis-Menten kinetics has a rate of the form:
#
#   r = kcat*[E]*[S] \ ([S] + Km)
#
# Therefore, we can model a Michaelis-Menten process using a
# ProportionalHillPositive propensity where:
#
# * k=kcat
# * d=[E]
# * s=[S]
# * n=1
# * K=Km


def main() -> None:
    # Define species.
    cephaloridine_in = Species('cephaloridine_periplasm')
    cephaloridine_out = Species('cephaloridine_environment')
    cephaloridine_hydrolyzed = Species('cephaloridine_hydrolyzed')

    pump = Species('AcrAB')
    beta_lactamase = Species('beta_lactamase')

    species = [
        cephaloridine_in,
        cephaloridine_out,
        cephaloridine_hydrolyzed,
        pump,
        beta_lactamase,
    ]

    # Define reactions.
    export_propensity = ProportionalHillPositive(
        k=ParameterEntry('export_kcat', CEPH_PUMP_KCAT),  # Hz
        s1=cephaloridine_in,
        K=ParameterEntry('export_km', CEPH_PUMP_KM),  # mM
        d=pump,
        n=1.75,
    )
    export = Reaction(
        inputs=[cephaloridine_in],
        outputs=[cephaloridine_out],
        propensity_type=export_propensity,
    )

    hydrolysis_propensity = ProportionalHillPositive(
        k=ParameterEntry('hydrolysis_kcat', CEPH_BETA_LACTAMASE_KCAT),  # Hz
        s1=cephaloridine_in,
        K=ParameterEntry('hydrolysis_km', CEPH_BETA_LACTAMASE_KM),  # mM
        d=beta_lactamase,
        n=1,
    )
    hydrolysis = Reaction(
        inputs=[cephaloridine_in],
        outputs=[cephaloridine_hydrolyzed],
        propensity_type=hydrolysis_propensity,
    )

    area_mass_ratio = ParameterEntry('x_am', 132)  # cm^2/mg
    permeability = ParameterEntry('perm', 52.6e-5 + 4.5e-5)  # cm/sec, ompF permeability + ompC permeability (Nikaido, 1983)
    mass = ParameterEntry('mass', 1170e-12)  # mg
    volume = ParameterEntry('volume', 0.32e-12)  # mL
    influx_propensity = GeneralPropensity(
        (
            f'x_am * perm * ({cephaloridine_out} - {cephaloridine_in}) '
            '* mass / (volume)'
        ),
        propensity_species=[cephaloridine_in, cephaloridine_out],
        propensity_parameters=[
            area_mass_ratio, permeability, mass, volume],
    )
    influx = Reaction(
        inputs=[cephaloridine_out],
        outputs=[cephaloridine_in],
        propensity_type=influx_propensity
    )

    initial_concentrations = {
        cephaloridine_in: 0,
        cephaloridine_out: 0.1239,
        cephaloridine_hydrolyzed: 0,
        pump: 0.0004525,
        beta_lactamase: 0.000525,
    }

    crn = ChemicalReactionNetwork(
        species=species,
        reactions=[export, hydrolysis, influx],
        initial_concentration_dict=initial_concentrations,
    )

    path = os.path.join(DATA_DIR, 'bioscrape_sbml.xml')
    print(f'Writing the following CRN to {path}:')
    print(crn.pretty_print(show_rates=True))
    crn.write_sbml_file(path)


if __name__ == '__main__':
    main()
