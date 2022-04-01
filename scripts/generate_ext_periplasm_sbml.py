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
# Cephaloridine default permeability = ompF permeability + ompC permeability (Nikaido, 1983)
DEFAULT_CEPH_OUTER_PERM = 52.6e-5 + 4.5e-5  # cm/sec

# Calculated by dividing V_max estimated in (Thanassi et al., 1995) by the model's initial pump concentration of
# 20.179269875115253 counts / micron^2
TET_PUMP_KCAT = 0.00015759727703788977  # / units.sec
# Estimated in (Thanassi et al., 1995)
TET_PUMP_KM = 200e-3  # * units.millimolar
# Estimated in (Thanassi et al., 1995)
DEFAULT_TET_OUTER_PERM = 1e-5  # cm/sec

PERIPLASM_FRACTION = 0.2
AREA_MASS_RATIO = 132  # cm^2/mg
MASS = 1170e-12  # mg
CELL_VOLUME = 1.2e-12  # mL
PERIPLASM_VOLUME = CELL_VOLUME * PERIPLASM_FRACTION  # mL
INITIAL_PERIPLASM_CEPH = 0
INITIAL_ENVIRONMENT_CEPH = 0.1239
INITIAL_HYDROLYZED_CEPH = 0
INITIAL_PERIPLASM_TET = 0
INITIAL_ENVIRONMENT_TET = 0.1239
INITIAL_PUMP = 0.0004525
INITIAL_BETA_LACTAMASE = 0.000525


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
    tetracycline_in = Species('tetracycline_periplasm')
    tetracycline_out = Species('tetracycline_environment')

    pump = Species('AcrAB')
    beta_lactamase = Species('beta_lactamase')

    species = [
        cephaloridine_in,
        cephaloridine_out,
        tetracycline_in,
        tetracycline_out,
        cephaloridine_hydrolyzed,
        pump,
        beta_lactamase,
    ]

    # Define reactions.
    ceph_export_propensity = ProportionalHillPositive(
        k=ParameterEntry('ceph_export_kcat', CEPH_PUMP_KCAT),  # Hz
        s1=cephaloridine_in,
        K=ParameterEntry('ceph_export_km', CEPH_PUMP_KM),  # mM
        d=pump,
        n=1.75,
    )
    ceph_export = Reaction(
        inputs=[cephaloridine_in],
        outputs=[cephaloridine_out],
        propensity_type=ceph_export_propensity,
    )

    tet_export_propensity = ProportionalHillPositive(
        k=ParameterEntry('tet_export_kcat', TET_PUMP_KCAT),  #Hz
        s1=tetracycline_in,
        K=ParameterEntry('tet_export_km', TET_PUMP_KM),  #mM
        d=pump,
        n=1
    )
    tet_export = Reaction(
        inputs=[tetracycline_in],
        outputs=[tetracycline_out],
        propensity_type=tet_export_propensity
    )

    hydrolysis_propensity = ProportionalHillPositive(
        k=ParameterEntry('tet_hydrolysis_kcat', CEPH_BETA_LACTAMASE_KCAT),  # Hz
        s1=cephaloridine_in,
        K=ParameterEntry('tet_hydrolysis_km', CEPH_BETA_LACTAMASE_KM),  # mM
        d=beta_lactamase,
        n=1,
    )
    hydrolysis = Reaction(
        inputs=[cephaloridine_in],
        outputs=[cephaloridine_hydrolyzed],
        propensity_type=hydrolysis_propensity,
    )

    area_mass_ratio = ParameterEntry('x_am', AREA_MASS_RATIO)  # cm^2/mg
    cephaloridine_permeability = ParameterEntry('cephaloridine_permeability', DEFAULT_CEPH_OUTER_PERM)  # cm/sec
    tetracycline_permeability = ParameterEntry('tetracycline_permeability', DEFAULT_TET_OUTER_PERM)  # cm/sec
    mass = ParameterEntry('mass', MASS)  # mg
    volume = ParameterEntry('volume', PERIPLASM_VOLUME)  # mL
    ceph_influx_propensity = GeneralPropensity(
        (
            f'x_am * cephaloridine_permeability * ({cephaloridine_out} - {cephaloridine_in}) '
            '* mass / (volume)'
        ),
        propensity_species=[cephaloridine_in, cephaloridine_out],
        propensity_parameters=[
            area_mass_ratio, cephaloridine_permeability, mass, volume],
    )
    ceph_influx = Reaction(
        inputs=[cephaloridine_out],
        outputs=[cephaloridine_in],
        propensity_type=ceph_influx_propensity
    )
    tet_influx_propensity = GeneralPropensity(
        (
            f'x_am * tetracycline_permeability * ({tetracycline_out} - {tetracycline_in}) '
            '* mass / (volume)'
        ),
        propensity_species=[tetracycline_in, tetracycline_out],
        propensity_parameters=[
            area_mass_ratio, tetracycline_permeability, mass, volume],
    )
    tet_influx = Reaction(
        inputs=[tetracycline_out],
        outputs=[tetracycline_in],
        propensity_type=tet_influx_propensity
    )

    initial_concentrations = {
        cephaloridine_in: INITIAL_PERIPLASM_CEPH,
        cephaloridine_out: INITIAL_ENVIRONMENT_CEPH,
        cephaloridine_hydrolyzed: INITIAL_HYDROLYZED_CEPH,
        pump: INITIAL_PUMP,
        tetracycline_in: INITIAL_PERIPLASM_TET,
        tetracycline_out: INITIAL_ENVIRONMENT_TET,
        beta_lactamase: INITIAL_BETA_LACTAMASE,
    }

    crn = ChemicalReactionNetwork(
        species=species,
        reactions=[ceph_export, tet_export, hydrolysis, ceph_influx, tet_influx],
        initial_concentration_dict=initial_concentrations,
    )

    path = os.path.join(DATA_DIR, 'ext_periplasm_sbml.xml')
    print(f'Writing the following CRN to {path}:')
    print(crn.pretty_print(show_rates=True))
    crn.write_sbml_file(path)


if __name__ == '__main__':
    main()
