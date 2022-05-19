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

from collections import namedtuple
import os

from biocrnpyler import (
    ChemicalReactionNetwork,
    ParameterEntry,
    GeneralPropensity,
    Reaction,
    Species,
)
from scipy import constants
from vivarium.library.units import units
from vivarium.library.topology import get_in

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data'))
FILENAME = 'amp_sbml.xml'
DIFFUSION_ONLY_FILENAME = 'amp_diffusion_only_sbml.xml'


MichaelisMentenKinetics = namedtuple(
    'MichaelisMentenKinetics', ('kcat', 'km', 'n'))


class Parameter:

    def __init__(self, value, source, canonicalize=None, note=''):
        self.value = value
        self.source = source
        self.canonicalize = canonicalize or (lambda x: x)
        self.note = note


class ParameterStore:

    def __init__(self, parameters):
        self.parameters = parameters

    def get(self, path):
        param_obj = get_in(self.parameters, path)
        if not param_obj:
            raise RuntimeError(
                f'No parameter found at path {path}')
        return param_obj.canonicalize(param_obj.value)


PARAMS = {
    'ampicillin': {
        'permeability': {
            'outer': Parameter(
                0.28e-5 * units.cm / units.sec,
                'Kojima and Nikaido (2013)',
                note='This is total, not per-porin, permeability.',
            ),
        },
        'mic': Parameter(
            2 * units.micrograms / units.mL,
            'Mazzariol, Cornaglia, and Nikaido (2000)',
            lambda x: (
                x / (349.4 * units.g / units.mol)
            ).to(units.mM),
        ),
        'efflux': {
            'kcat': Parameter(
                0.085 * units.nmol / units.mg / units.sec,  # Vmax
                'Kojima and Nikaido (2013)',
                lambda x: (
                    x / (6.7e-4 * units.mM)  # Pump concentration
                    * 1170 * units.fg  # Total cell mass
                    / (0.212799 * units.fL)  # Periplasm volume
                )
            ),
            'km': Parameter(
                2.16e-3 * units.mM,
                'Kojima and Nikaido (2013)',
            ),
            'n': Parameter(
                1.9,
                'Kojima and Nikaido (2013)',
            ),
        },
        'hydrolysis': {
            'kcat': Parameter(
                6.5e-3 / units.sec,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
            'km': Parameter(
                0.9e-3 * units.mM,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
            'n': Parameter(
                1,
                'Mazzariol, Cornaglia, and Nikaido (2000)',
            ),
        },
    },
    'shape': {
        'periplasm_fraction': Parameter(
            0.2,
            'Stock et al. (1977)'
        ),
        'initial_cell_mass': Parameter(
            1170 * units.fg,
            'Model',
        ),
        'initial_cell_volume': Parameter(
            1.2 * units.fL,
            'Model',
        ),
        'initial_area': Parameter(
            4.52 * units.um**2,
            'Model',
        ),
    },
    'concs': {
        'initial_pump': Parameter(
            6.7e-4 * units.mM,
            'Model',
        ),
        'initial_hydrolase': Parameter(
            7.1e-4 * units.mM,
            'Model',
        ),
    },
}


def make_beta_lactam_reactions(
        name, export_kinetics, hydrolysis_kinetics, initial_area,
        permeability, periplasm_volume, initial_environmental,
        initial_pump, initial_hydrolase):

    # DEFINE SPECIES.

    environmental = Species(f'{name}_environment')
    environmental_delta = Species(f'{name}_environment_delta')
    periplasmic = Species(f'{name}_periplasm')
    hydrolyzed = Species(f'{name}_hydrolyzed')

    pump = Species('acrab_tolc')
    hydrolase = Species('ampc')

    # DEFINE PARAMETERS.

    export_kcat = ParameterEntry(
        'export_kcat', export_kinetics.kcat.to(1 / units.sec).magnitude)
    export_km = ParameterEntry(
        'export_km', export_kinetics.km.to(units.mM).magnitude)
    export_n = ParameterEntry('export_n', export_kinetics.n)

    hydrolysis_kcat = ParameterEntry(
        'hydrolysis_kcat',
        hydrolysis_kinetics.kcat.to(1 / units.sec).magnitude)
    hydrolysis_km = ParameterEntry(
        'hydrolysis_km', hydrolysis_kinetics.km.to(units.mM).magnitude)
    hydrolysis_n = ParameterEntry('hydrolysis_n', hydrolysis_kinetics.n)

    area = ParameterEntry(
        'area', initial_area.to(units.dm**2).magnitude)
    permeability = ParameterEntry(
        'permeability', permeability.to(units.dm / units.sec).magnitude)

    volume_p = ParameterEntry(
        'volume_p', periplasm_volume.to(units.L).magnitude)

    # DEFINE REACTIONS.

    # Antibiotic being pumped out of the periplasm by AcrAB-TolC.
    export_propensity = GeneralPropensity(
        (
            f'export_kcat * {pump} '
            f'* ({periplasmic} / volume_p)^export_n '
            f'/ (({periplasmic} / volume_p)^export_n '
            f'+ export_km)'
        ),
        propensity_species=[pump, periplasmic],
        propensity_parameters=[
            export_kcat, export_km, export_n, volume_p],
    )
    export = Reaction(
        inputs=[periplasmic],
        outputs=[environmental_delta],
        propensity_type=export_propensity,
    )

    # Cephaloridine being hydrolyzed in the periplasm
    hydrolysis_propensity = GeneralPropensity(
        (
            f'hydrolysis_kcat * {hydrolase} '
            f'* ({periplasmic} / volume_p)^hydrolysis_n '
            f'/ (({periplasmic} / volume_p)^hydrolysis_n '
            f'+ hydrolysis_km)'
        ),
        propensity_species=[hydrolase, periplasmic],
        propensity_parameters=[
            hydrolysis_kcat, hydrolysis_km, hydrolysis_n,
            volume_p],
    )
    hydrolysis = Reaction(
        inputs=[periplasmic],
        outputs=[hydrolyzed],
        propensity_type=hydrolysis_propensity,
    )

    # Cephaloridine diffusion between environment and periplasm
    influx_propensity = GeneralPropensity(
        (
            f'area * permeability * {environmental}'
        ),
        propensity_species=[environmental],
        propensity_parameters=[area, permeability],
    )
    influx = Reaction(
        inputs=[environmental_delta],
        outputs=[periplasmic],
        propensity_type=influx_propensity
    )
    influx_rev_propensity = GeneralPropensity(
        (
            f'area * permeability * {periplasmic} / volume_p'
        ),
        propensity_species=[periplasmic],
        propensity_parameters=[area, permeability, volume_p],
    )
    influx_rev = Reaction(
        inputs=[periplasmic],
        outputs=[environmental_delta],
        propensity_type=influx_rev_propensity,
    )

    # DEFINE CRNs.

    crn = ChemicalReactionNetwork(
        species=[
            environmental,
            environmental_delta,
            periplasmic,
            hydrolyzed,
            pump,
            hydrolase,
        ],
        reactions=[
            export,
            hydrolysis,
            influx,
            influx_rev,
        ],
        initial_concentration_dict={
            environmental: initial_environmental.to(units.mM).magnitude,
            environmental_delta: 0,
            periplasmic: 0,
            hydrolyzed: 0,
            pump: initial_pump.to(units.mM).magnitude,
            hydrolase: initial_hydrolase.to(units.mM).magnitude,
        },
    )

    diffusion_only_crn = ChemicalReactionNetwork(
        species=[
            environmental,
            environmental_delta,
            periplasmic,
        ],
        reactions=[
            influx,
            influx_rev,
        ],
        initial_concentration_dict={
            environmental: initial_environmental.to(units.mM).magnitude,
            environmental_delta: 0,
            periplasmic: 0,
        },
    )

    return crn, diffusion_only_crn


def main() -> None:
    params = ParameterStore(PARAMS)

    crn, diffusion_only_crn = make_beta_lactam_reactions(
        'ampicillin',
        MichaelisMentenKinetics(
            params.get(('ampicillin', 'efflux', 'kcat')),
            params.get(('ampicillin', 'efflux', 'km')),
            params.get(('ampicillin', 'efflux', 'n')),
        ),
        MichaelisMentenKinetics(
            params.get(('ampicillin', 'hydrolysis', 'kcat')),
            params.get(('ampicillin', 'hydrolysis', 'km')),
            params.get(('ampicillin', 'hydrolysis', 'n')),
        ),
        params.get(('shape', 'initial_area')),
        params.get(('ampicillin', 'permeability', 'outer')),
        (
            params.get(('shape', 'initial_cell_volume'))
            * params.get(('shape', 'periplasm_fraction'))
        ),
        params.get(('ampicillin', 'mic')),
        params.get(('concs', 'initial_pump')),
        params.get(('concs', 'initial_hydrolase')),
    )

    # SAVE SBML FILES.

    path = os.path.join(DATA_DIR, FILENAME)
    print(f'Writing the following CRN to {path}:')
    print(crn.pretty_print(show_rates=True))
    crn.write_sbml_file(path)

    path = os.path.join(DATA_DIR, DIFFUSION_ONLY_FILENAME)
    print(f'Writing the following CRN to {path}:')
    print(diffusion_only_crn.pretty_print(show_rates=True))
    diffusion_only_crn.write_sbml_file(path)


if __name__ == '__main__':
    main()
