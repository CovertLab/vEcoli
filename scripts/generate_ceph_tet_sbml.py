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

DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'data'))
FILENAME = 'ceph_tet_sbml.xml'
DIFFUSION_ONLY_FILENAME = 'ceph_tet_diffusion_only_sbml.xml'

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
# Estimated in (Thanassi et al., 1995)
DEFAULT_TET_INNER_PERM = 3 * 1e-6  # cm/sec

# Estimated in (Stock et al., 1977)
PERIPLASM_FRACTION = 0.2
CYTOPLASM_FRACTION = 1 - PERIPLASM_FRACTION

AREA_MASS_RATIO = 132  # cm^2/mg
CYTO_AREA_MASS_RATIO = AREA_MASS_RATIO / CYTOPLASM_FRACTION  # cm^2/mg, Dividing by 0.8 as cytosol has 80% of mass
CELL_MASS = 1170e-12  # mg
CELL_VOLUME = 1.2e-12  # mL
PERIPLASM_VOLUME = CELL_VOLUME * PERIPLASM_FRACTION  # mL
CYTOPLASM_VOLUME = CELL_VOLUME * CYTOPLASM_FRACTION  # mL
INITIAL_PERIPLASM_CEPH = 0
INITIAL_ENVIRONMENT_CEPH = 0.1239
INITIAL_HYDROLYZED_CEPH = 0
INITIAL_PERIPLASM_TET = 0
INITIAL_ENVIRONMENT_TET = 0.1239
INITIAL_CYTOPLASM_TET = 0
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
    cephaloridine_p = Species('cephaloridine_periplasm')
    cephaloridine_e = Species('cephaloridine_environment')
    cephaloridine_hydrolyzed = Species('cephaloridine_hydrolyzed')
    tetracycline_p = Species('tetracycline_periplasm')
    tetracycline_e = Species('tetracycline_environment')
    tetracycline_c = Species('tetracycline_cytoplasm')

    pump = Species('AcrAB')
    beta_lactamase = Species('beta_lactamase')

    species = [
        cephaloridine_p,
        cephaloridine_e,
        tetracycline_p,
        tetracycline_e,
        tetracycline_c,
        cephaloridine_hydrolyzed,
        pump,
        beta_lactamase,
    ]

    # Define reactions.

    # Cephaloridine being pumped out of the periplasm by AcrAB-TolC
    ceph_export_propensity = ProportionalHillPositive(
        k=ParameterEntry('ceph_export_kcat', CEPH_PUMP_KCAT),  # Hz
        s1=cephaloridine_p,
        K=ParameterEntry('ceph_export_km', CEPH_PUMP_KM),  # mM
        d=pump,
        n=1.75,
    )
    ceph_export = Reaction(
        inputs=[cephaloridine_p],
        outputs=[cephaloridine_e],
        propensity_type=ceph_export_propensity,
    )

    # Tetracycline being pumped out of the periplasm by AcrAB-TolC
    tet_export_propensity = ProportionalHillPositive(
        k=ParameterEntry('tet_export_kcat', TET_PUMP_KCAT),  #Hz
        s1=tetracycline_p,
        K=ParameterEntry('tet_export_km', TET_PUMP_KM),  #mM
        d=pump,
        n=1
    )
    tet_export = Reaction(
        inputs=[tetracycline_p],
        outputs=[tetracycline_e],
        propensity_type=tet_export_propensity
    )

    # Cephaloridine being hydrolyzed in the periplasm
    hydrolysis_propensity = ProportionalHillPositive(
        k=ParameterEntry('ceph_hydrolysis_kcat', CEPH_BETA_LACTAMASE_KCAT),  # Hz
        s1=cephaloridine_p,
        K=ParameterEntry('ceph_hydrolysis_km', CEPH_BETA_LACTAMASE_KM),  # mM
        d=beta_lactamase,
        n=1,
    )
    hydrolysis = Reaction(
        inputs=[cephaloridine_p],
        outputs=[cephaloridine_hydrolyzed],
        propensity_type=hydrolysis_propensity,
    )

    # Creating diffusion parameters
    periplasm_area_mass_ratio = ParameterEntry('outer_x_am', AREA_MASS_RATIO)  # cm^2/mg
    cephaloridine_permeability = ParameterEntry('outer_cephaloridine_permeability', DEFAULT_CEPH_OUTER_PERM)  # cm/sec
    tetracycline_permeability = ParameterEntry('outer_tetracycline_permeability', DEFAULT_TET_OUTER_PERM)  # cm/sec
    mass = ParameterEntry('mass', CELL_MASS)  # mg
    volume_p = ParameterEntry('volume_p', PERIPLASM_VOLUME)  # mL

    # Cephaloridine diffusion between environment and periplasm
    ceph_influx_propensity = GeneralPropensity(
        (
            'outer_x_am * outer_cephaloridine_permeability '
            f'* {cephaloridine_e} * mass / volume_p'
        ),
        propensity_species=[cephaloridine_e],
        propensity_parameters=[
            periplasm_area_mass_ratio, cephaloridine_permeability, mass, volume_p],
    )
    ceph_influx = Reaction(
        inputs=[cephaloridine_e],
        outputs=[cephaloridine_p],
        propensity_type=ceph_influx_propensity
    )
    ceph_influx_rev_propensity = GeneralPropensity(
        (
            'outer_x_am * outer_cephaloridine_permeability '
            f'* {cephaloridine_p} * mass / volume_p'
        ),
        propensity_species=[cephaloridine_p],
        propensity_parameters=[
            periplasm_area_mass_ratio, cephaloridine_permeability, mass, volume_p],
    )
    ceph_influx_rev = Reaction(
        inputs=[cephaloridine_p],
        outputs=[cephaloridine_e],
        propensity_type=ceph_influx_rev_propensity,
    )

    # Tetracycline diffusion between environment and periplasm
    tet_e_p_influx_propensity = GeneralPropensity(
        (
            'outer_x_am * outer_tetracycline_permeability '
            f'* {tetracycline_e} * mass / volume_p'
        ),
        propensity_species=[tetracycline_e],
        propensity_parameters=[
            periplasm_area_mass_ratio, tetracycline_permeability, mass, volume_p],
    )
    tet_e_p_influx = Reaction(
        inputs=[tetracycline_e],
        outputs=[tetracycline_p],
        propensity_type=tet_e_p_influx_propensity
    )
    tet_e_p_influx_rev_propensity = GeneralPropensity(
        (
            'outer_x_am * outer_tetracycline_permeability '
            f'* {tetracycline_p} * mass / volume_p'
        ),
        propensity_species=[tetracycline_p],
        propensity_parameters=[
            periplasm_area_mass_ratio, tetracycline_permeability, mass, volume_p],
    )
    tet_e_p_influx_rev = Reaction(
        inputs=[tetracycline_p],
        outputs=[tetracycline_e],
        propensity_type=tet_e_p_influx_rev_propensity,
    )

    # Tetracycline diffusion between periplasm and cytoplasm
    # dTp = D(Tp - Tc) / vol_p
    # dTc = D(Tp - Tc) / vol_c
    CYTO_AREA_MASS_RATIO = AREA_MASS_RATIO / CYTOPLASM_FRACTION  # cm^2/mg, Dividing by 0.8 as cytosol has 80% of mass
    cyto_area_mass_ratio = ParameterEntry('inner_x_am', CYTO_AREA_MASS_RATIO)  # cm^2/mg
    inner_tet_perm = ParameterEntry('inner_tetracycline_permeability', DEFAULT_TET_INNER_PERM)  # cm/sec
    volume_c = ParameterEntry('volume_c', CYTOPLASM_VOLUME)
    tet_p_c_influx_propensity = GeneralPropensity(
        (
            'inner_x_am * inner_tetracycline_permeability '
            f'* {tetracycline_p} * mass / volume_c'
        ),
        propensity_species=[tetracycline_p],
        propensity_parameters=[
            cyto_area_mass_ratio, inner_tet_perm, mass, volume_c],
    )
    tet_p_c_influx = Reaction(
        inputs=[tetracycline_p],
        outputs=[tetracycline_c],
        propensity_type=tet_p_c_influx_propensity
    )
    tet_p_c_influx_rev_propensity = GeneralPropensity(
        (
            'inner_x_am * inner_tetracycline_permeability '
            f'* {tetracycline_c} * mass / volume_c'
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

    initial_concentrations = {
        cephaloridine_p: INITIAL_PERIPLASM_CEPH,
        cephaloridine_e: INITIAL_ENVIRONMENT_CEPH,
        cephaloridine_hydrolyzed: INITIAL_HYDROLYZED_CEPH,
        pump: INITIAL_PUMP,
        tetracycline_p: INITIAL_PERIPLASM_TET,
        tetracycline_e: INITIAL_ENVIRONMENT_TET,
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


    path = os.path.join(DATA_DIR, FILENAME)
    print(f'Writing the following CRN to {path}:')
    print(crn.pretty_print(show_rates=True))
    crn.write_sbml_file(path)

    diffusion_only_crn = ChemicalReactionNetwork(
        species=[
            cephaloridine_e,
            cephaloridine_p,
            tetracycline_e,
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
            cephaloridine_p: INITIAL_PERIPLASM_CEPH,
            tetracycline_e: INITIAL_ENVIRONMENT_TET,
            tetracycline_p: INITIAL_PERIPLASM_TET,
            tetracycline_c: INITIAL_CYTOPLASM_TET,
        },
    )
    path = os.path.join(DATA_DIR, DIFFUSION_ONLY_FILENAME)
    print(f'Writing the following CRN to {path}:')
    print(diffusion_only_crn.pretty_print(show_rates=True))
    diffusion_only_crn.write_sbml_file(path)


if __name__ == '__main__':
    main()
