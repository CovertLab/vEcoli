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
from ecoli.parameters import param_store

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data'))
FILENAME = 'amp_sbml.xml'
DIFFUSION_ONLY_FILENAME = 'amp_diffusion_only_sbml.xml'


MichaelisMentenKinetics = namedtuple(
    'MichaelisMentenKinetics', ('kcat', 'km', 'n'))


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
    crn, diffusion_only_crn = make_beta_lactam_reactions(
        'ampicillin',
        MichaelisMentenKinetics(
            param_store.get(('ampicillin', 'efflux', 'kcat')),
            param_store.get(('ampicillin', 'efflux', 'km')),
            param_store.get(('ampicillin', 'efflux', 'n')),
        ),
        MichaelisMentenKinetics(
            param_store.get(('ampicillin', 'hydrolysis', 'kcat')),
            param_store.get(('ampicillin', 'hydrolysis', 'km')),
            param_store.get(('ampicillin', 'hydrolysis', 'n')),
        ),
        param_store.get(('shape', 'initial_area')),
        param_store.get(('ampicillin', 'permeability', 'outer')),
        (
            param_store.get(('shape', 'initial_cell_volume'))
            * param_store.get(('shape', 'periplasm_fraction'))
        ),
        param_store.get(('ampicillin', 'mic')),
        param_store.get(('concs', 'initial_pump')),
        param_store.get(('concs', 'initial_hydrolase')),
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
