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

# Estimated in (Thanassi et al., 1995)
DEFAULT_TET_INNER_PERM = 3 * 1e-6  # cm/sec

AREA_MASS_RATIO = 132  # cm^2/mg
MASS = 1170e-12  # mg
VOLUME = 0.32e-12  # mL  # TODO(Matt): Is this volume of the entire cell?
INITIAL_CYTOPLASM_TET = 0
INITIAL_PERIPLASM_TET = 0.1239


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
    tetracycline_in = Species('tetracycline_cytoplasm')
    tetracycline_out = Species('tetracycline_periplasm')

    species = [
        tetracycline_in,
        tetracycline_out,
    ]

    # Define reaction.
    area_mass_ratio = ParameterEntry('x_am', AREA_MASS_RATIO)  # cm^2/mg
    tet_permeability = ParameterEntry('perm', DEFAULT_TET_INNER_PERM)  # cm/sec
    mass = ParameterEntry('mass', MASS)  # mg
    volume = ParameterEntry('volume', VOLUME)  # mL
    tet_influx_propensity = GeneralPropensity(
        (
            f'x_am * perm * ({tetracycline_out} - {tetracycline_in}) '
            '* mass / (volume)'
        ),
        propensity_species=[tetracycline_in, tetracycline_out],
        propensity_parameters=[
            area_mass_ratio, tet_permeability, mass, volume],
    )
    tet_influx = Reaction(
        inputs=[tetracycline_out],
        outputs=[tetracycline_in],
        propensity_type=tet_influx_propensity
    )

    initial_concentrations = {
        tetracycline_in: INITIAL_CYTOPLASM_TET,
        tetracycline_out: INITIAL_PERIPLASM_TET,
    }

    crn = ChemicalReactionNetwork(
        species=species,
        reactions=[tet_influx],
        initial_concentration_dict=initial_concentrations,
    )

    path = os.path.join(DATA_DIR, 'periplasm_cytoplasm_sbml.xml')
    print(f'Writing the following CRN to {path}:')
    print(crn.pretty_print(show_rates=True))
    crn.write_sbml_file(path)


if __name__ == '__main__':
    main()
