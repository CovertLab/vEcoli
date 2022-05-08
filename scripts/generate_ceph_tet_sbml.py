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
    GeneralPropensity,
    Reaction,
    Species,
)
from scipy import constants

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data'))
FILENAME = 'ceph_tet_sbml.xml'
DIFFUSION_ONLY_FILENAME = 'ceph_tet_diffusion_only_sbml.xml'

# Calculated by dividing V_max reported in (Nagano & Nikaido, 2009) by
# the model's initial pump concentration of 20.179269875115253 counts /
# micron^2
CEPH_PUMP_KCAT = 0.0956090147363198  # / units.sec
# Reported in (Nagano & Nikaido, 2009)
CEPH_PUMP_KM = 288e-6  # M
# Reported in (Galleni et al., 1988)
CEPH_BETA_LACTAMASE_KCAT = 130  # / units.sec
# Reported in (Galleni et al., 1988)
CEPH_BETA_LACTAMASE_KM = 170e-3  # M
# Cephaloridine default permeability = ompF permeability + ompC
# permeability (Nikaido, 1983)
DEFAULT_CEPH_OUTER_PERM = 52.6e-5 + 4.5e-5  # cm/sec

# Calculated by dividing V_max estimated in (Thanassi et al., 1995) by
# the model's initial pump concentration of 20.179269875115253 counts /
# micron^2
TET_PUMP_KCAT = 0.00015759727703788977  # / units.sec
# Estimated in (Thanassi et al., 1995)
TET_PUMP_KM = 200e-6  # M
# Estimated in (Thanassi et al., 1995)
DEFAULT_TET_OUTER_PERM = 1e-5  # cm/sec
# Estimated in (Thanassi et al., 1995)
DEFAULT_TET_INNER_PERM = 3 * 1e-6  # cm/sec

# Estimated in (Stock et al., 1977)
PERIPLASM_FRACTION = 0.2
CYTOPLASM_FRACTION = 1 - PERIPLASM_FRACTION

AREA_MASS_RATIO = 132  # cm^2/mg
# cm^2/mg, Dividing by 0.8 as cytosol has 80% of mass
CYTO_AREA_MASS_RATIO = AREA_MASS_RATIO / CYTOPLASM_FRACTION
CELL_MASS = 1170e-12  # mg
CELL_VOLUME = 1.2e-15  # L
PERIPLASM_VOLUME = CELL_VOLUME * PERIPLASM_FRACTION  # L
CYTOPLASM_VOLUME = CELL_VOLUME * CYTOPLASM_FRACTION  # L

INITIAL_PERIPLASM_CEPH = 0  # mmol
INITIAL_ENVIRONMENT_CEPH = 0.1239  # mM
INITIAL_HYDROLYZED_CEPH = 0  # mmol
INITIAL_PERIPLASM_TET = 0  # mmol
INITIAL_ENVIRONMENT_TET = 0.1239  # mM
INITIAL_CYTOPLASM_TET = 0  # mmol
INITIAL_PUMP = 0.0004525  # mM
INITIAL_BETA_LACTAMASE = 0.000525  # mM

# Source: Berg, Howard C., E. coli in Motion. 1934. Page 105.
MEMBRANE_POTENTIAL = 0.12  # 120 mV with the interior negative.
# Source: Thanassi, Suh, and Nikaido 1995, page 999
TETRACYCLINE_CHARGE = 1  # Chelated to Mg when entering cell.
FARADAY_CONSTANT = constants.value('Faraday constant')  # C/mol
GAS_CONSTANT = constants.R  # J/mol/K
TEMPERATURE = 298  # K


def main() -> None:
    # DEFINE SPECIES.

    cephaloridine_e = Species('cephaloridine_environment')
    cephaloridine_p = Species('cephaloridine_periplasm')
    cephaloridine_e_delta = Species('cephaloridine_environment_delta')
    cephaloridine_hydrolyzed = Species('cephaloridine_hydrolyzed')

    tetracycline_p = Species('tetracycline_periplasm')
    tetracycline_e = Species('tetracycline_environment')
    tetracycline_e_delta = Species('tetracycline_environment_delta')
    tetracycline_c = Species('tetracycline_cytoplasm')

    pump = Species('AcrAB')
    beta_lactamase = Species('beta_lactamase')

    species = [
        cephaloridine_p,
        cephaloridine_e,
        cephaloridine_e_delta,
        tetracycline_p,
        tetracycline_e,
        tetracycline_e_delta,
        tetracycline_c,
        cephaloridine_hydrolyzed,
        pump,
        beta_lactamase,
    ]

    # DEFINE PARAMETERS.

    ceph_export_kcat = ParameterEntry(
        'ceph_export_kcat', CEPH_PUMP_KCAT)  # Hz
    ceph_export_km = ParameterEntry(
        'ceph_export_km', CEPH_PUMP_KM)  # mM
    ceph_export_n = ParameterEntry('ceph_export_n', 1.75)

    tet_export_kcat = ParameterEntry(
        'tet_export_kcat', TET_PUMP_KCAT)  #Hz
    tet_export_km = ParameterEntry('tet_export_km', TET_PUMP_KM)  #mM
    tet_export_n = ParameterEntry('tet_export_n', 1)

    ceph_hydrolysis_kcat = ParameterEntry(
        'ceph_hydrolysis_kcat', CEPH_BETA_LACTAMASE_KCAT)  # Hz
    ceph_hydrolysis_km = ParameterEntry(
        'ceph_hydrolysis_km', CEPH_BETA_LACTAMASE_KM)  # mM
    ceph_hydrolysis_n = ParameterEntry('ceph_hydrolysis_n', 1)

    periplasm_area_mass_ratio = ParameterEntry(
        'outer_x_am', AREA_MASS_RATIO)  # cm^2/mg
    cyto_area_mass_ratio = ParameterEntry(
        'inner_x_am', CYTO_AREA_MASS_RATIO)  # cm^2/mg
    mass = ParameterEntry('mass', CELL_MASS)  # mg

    cephaloridine_permeability = ParameterEntry(
        'outer_cephaloridine_permeability',
        DEFAULT_CEPH_OUTER_PERM)  # cm/sec
    tetracycline_permeability = ParameterEntry(
        'outer_tetracycline_permeability',
        DEFAULT_TET_OUTER_PERM)  # cm/sec
    inner_tet_perm = ParameterEntry(
        'inner_tetracycline_permeability',
        DEFAULT_TET_INNER_PERM)  # cm/sec

    membrane_potential = ParameterEntry(
        'E', MEMBRANE_POTENTIAL)
    faraday_constant = ParameterEntry('F', FARADAY_CONSTANT)
    tetracycline_charge = ParameterEntry('z', TETRACYCLINE_CHARGE)
    gas_constant = ParameterEntry('R', GAS_CONSTANT)
    temperature = ParameterEntry('T', TEMPERATURE)

    volume_p = ParameterEntry('volume_p', PERIPLASM_VOLUME)  # mL
    volume_c = ParameterEntry('volume_c', CYTOPLASM_VOLUME)

    # DEFINE REACTIONS.

    # Cephaloridine being pumped out of the periplasm by AcrAB-TolC
    ceph_export_propensity = GeneralPropensity(
        (
            f'ceph_export_kcat * {pump} '
            f'* ({cephaloridine_p} / volume_p)^ceph_export_n '
            f'/ (({cephaloridine_p} / volume_p)^ceph_export_n '
            f'+ ceph_export_km)'
        ),
        propensity_species=[pump, cephaloridine_p],
        propensity_parameters=[
            ceph_export_kcat, ceph_export_km, ceph_export_n, volume_p],
    )
    ceph_export = Reaction(
        inputs=[cephaloridine_p],
        outputs=[cephaloridine_e_delta],
        propensity_type=ceph_export_propensity,
    )

    # Tetracycline being pumped out of the periplasm by AcrAB-TolC
    tet_export_propensity = GeneralPropensity(
        (
            f'tet_export_kcat * {pump} '
            f'* ({tetracycline_p} / volume_p)^tet_export_n '
            f'/ (({tetracycline_p} / volume_p)^tet_export_n '
            f'+ tet_export_km)'
        ),
        propensity_species=[pump, tetracycline_p],
        propensity_parameters=[
            tet_export_kcat, tet_export_km, tet_export_n, volume_p],
    )
    tet_export = Reaction(
        inputs=[tetracycline_p],
        outputs=[tetracycline_e_delta],
        propensity_type=tet_export_propensity
    )

    # Cephaloridine being hydrolyzed in the periplasm
    hydrolysis_propensity = GeneralPropensity(
        (
            f'ceph_hydrolysis_kcat * {beta_lactamase} '
            f'* ({cephaloridine_p} / volume_p)^ceph_hydrolysis_n '
            f'/ (({cephaloridine_p} / volume_p)^ceph_hydrolysis_n '
            f'+ ceph_hydrolysis_km)'
        ),
        propensity_species=[beta_lactamase, cephaloridine_p],
        propensity_parameters=[
            ceph_hydrolysis_kcat, ceph_hydrolysis_km, ceph_hydrolysis_n,
            volume_p],
    )
    hydrolysis = Reaction(
        inputs=[cephaloridine_p],
        outputs=[cephaloridine_hydrolyzed],
        propensity_type=hydrolysis_propensity,
    )

    # Cephaloridine diffusion between environment and periplasm
    ceph_influx_propensity = GeneralPropensity(
        (
            # 1e-3 is to convert (mass * x_am * permeability), which is
            # in mL/sec, to L/sec to be compatible with the mmol/L
            # concentrations.
            'mass * outer_x_am * outer_cephaloridine_permeability * 1e-3 '
            f'* {cephaloridine_e}'
        ),
        propensity_species=[cephaloridine_e],
        propensity_parameters=[
            periplasm_area_mass_ratio, cephaloridine_permeability, mass],
    )
    ceph_influx = Reaction(
        inputs=[cephaloridine_e_delta],
        outputs=[cephaloridine_p],
        propensity_type=ceph_influx_propensity
    )
    ceph_influx_rev_propensity = GeneralPropensity(
        (
            # 1e-3 is to convert (mass * x_am * permeability), which is
            # in mL/sec, to L/sec to be compatible with the mmol/L
            # concentrations.
            'mass * outer_x_am * outer_cephaloridine_permeability * 1e-3 '
            f'* {cephaloridine_p} / volume_p'
        ),
        propensity_species=[cephaloridine_p],
        propensity_parameters=[
            periplasm_area_mass_ratio, cephaloridine_permeability, mass, volume_p],
    )
    ceph_influx_rev = Reaction(
        inputs=[cephaloridine_p],
        outputs=[cephaloridine_e_delta],
        propensity_type=ceph_influx_rev_propensity,
    )

    # Tetracycline diffusion between environment and periplasm

    # From the Nernst Equation, we know that E = RT/(zF) ln(i/o) for
    # membrane potential E, gas constant R, absolute temperature T,
    # tetracycline charge z, Faraday constant F, internal tetracycline
    # concentration i, and external tetracycline concentration o.
    # Rearranging, we get i = o * exp(zFE/(RT)). Since we can assume the
    # external concentrations are constant, the external concentraion in
    # Fick's law is just the equilibrium internal concentration, so we
    # can substitute in our expression for i like this:
    #     di/dt = D (o exp(zFE/(RT)) - i)
    # where D is the diffusivity, which we calculate as x_am * p * m.
    tet_e_p_influx_propensity = GeneralPropensity(
        (
            # 1e-3 is to convert (mass * x_am * permeability), which is
            # in mL/sec, to L/sec to be compatible with the mmol/L
            # concentrations.
            'mass * outer_x_am * outer_tetracycline_permeability * 1e-3 '
            f'* {tetracycline_e} * exp(z * F * E / R / T)'
        ),
        propensity_species=[tetracycline_e],
        propensity_parameters=[
            periplasm_area_mass_ratio, tetracycline_permeability, mass,
            membrane_potential, faraday_constant, tetracycline_charge,
            gas_constant, temperature],
    )
    tet_e_p_influx = Reaction(
        inputs=[tetracycline_e_delta],
        outputs=[tetracycline_p],
        propensity_type=tet_e_p_influx_propensity
    )
    tet_e_p_influx_rev_propensity = GeneralPropensity(
        (
            # 1e-3 is to convert (mass * x_am * permeability), which is
            # in mL/sec, to L/sec to be compatible with the mmol/L
            # concentrations.
            'mass * outer_x_am * outer_tetracycline_permeability * 1e-3 '
            f'* {tetracycline_p} / volume_p'
        ),
        propensity_species=[tetracycline_p],
        propensity_parameters=[
            periplasm_area_mass_ratio, tetracycline_permeability, mass, volume_p],
    )
    tet_e_p_influx_rev = Reaction(
        inputs=[tetracycline_p],
        outputs=[tetracycline_e_delta],
        propensity_type=tet_e_p_influx_rev_propensity,
    )

    # Tetracycline diffusion between periplasm and cytoplasm
    # dTp = D(Tp - Tc) / vol_p
    # dTc = D(Tp - Tc) / vol_c
    tet_p_c_influx_propensity = GeneralPropensity(
        (
            # 1e-3 is to convert (mass * x_am * permeability), which is
            # in mL/sec, to L/sec to be compatible with the mmol/L
            # concentrations.
            'mass * inner_x_am * inner_tetracycline_permeability * 1e-3 '
            f'* {tetracycline_p} / volume_p'
        ),
        propensity_species=[tetracycline_p],
        propensity_parameters=[
            cyto_area_mass_ratio, inner_tet_perm, mass, volume_p],
    )
    tet_p_c_influx = Reaction(
        inputs=[tetracycline_p],
        outputs=[tetracycline_c],
        propensity_type=tet_p_c_influx_propensity
    )
    tet_p_c_influx_rev_propensity = GeneralPropensity(
        (
            # 1e-3 is to convert (mass * x_am * permeability), which is
            # in mL/sec, to L/sec to be compatible with the mmol/L
            # concentrations.
            'mass * inner_x_am * inner_tetracycline_permeability * 1e-3 '
            f'* {tetracycline_c} / volume_c'
        ),
        propensity_species=[tetracycline_c],
        propensity_parameters=[
            cyto_area_mass_ratio, inner_tet_perm, mass, volume_c],
    )
    tet_p_c_influx_rev = Reaction(
        inputs=[tetracycline_c],
        outputs=[tetracycline_p],
        propensity_type=tet_p_c_influx_rev_propensity
    )

    # DEFINE CRNS.

    initial_concentrations = {
        cephaloridine_p: INITIAL_PERIPLASM_CEPH,
        cephaloridine_e: INITIAL_ENVIRONMENT_CEPH,
        cephaloridine_e_delta: 0,
        cephaloridine_hydrolyzed: INITIAL_HYDROLYZED_CEPH,
        pump: INITIAL_PUMP,
        tetracycline_p: INITIAL_PERIPLASM_TET,
        tetracycline_e: INITIAL_ENVIRONMENT_TET,
        tetracycline_e_delta: 0,
        tetracycline_c: INITIAL_CYTOPLASM_TET,
        beta_lactamase: INITIAL_BETA_LACTAMASE,
    }

    crn = ChemicalReactionNetwork(
        species=species,
        reactions=[
            ceph_export,
            tet_export,
            hydrolysis,
            ceph_influx,
            ceph_influx_rev,
            tet_e_p_influx,
            tet_e_p_influx_rev,
            tet_p_c_influx,
            tet_p_c_influx_rev,
        ],
        initial_concentration_dict=initial_concentrations,
    )

    diffusion_only_crn = ChemicalReactionNetwork(
        species=[
            cephaloridine_e,
            cephaloridine_e_delta,
            cephaloridine_p,
            tetracycline_e,
            tetracycline_e_delta,
            tetracycline_p,
            tetracycline_c
        ],
        reactions=[
            ceph_influx,
            ceph_influx_rev,
            tet_e_p_influx,
            tet_e_p_influx_rev,
            tet_p_c_influx,
            tet_p_c_influx_rev,
        ],
        initial_concentration_dict={
            cephaloridine_e: INITIAL_ENVIRONMENT_CEPH,
            cephaloridine_e_delta: 0,
            cephaloridine_p: INITIAL_PERIPLASM_CEPH,
            tetracycline_e: INITIAL_ENVIRONMENT_TET,
            tetracycline_e_delta: 0,
            tetracycline_p: INITIAL_PERIPLASM_TET,
            tetracycline_c: INITIAL_CYTOPLASM_TET,
        },
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
