import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants

from iteround import saferound

from vivarium.core.process import Process
from vivarium.core.composition import (
    simulate_process_in_experiment,
    PROCESS_OUT_DIR,
)

NAME = 'diffusion_network'

class DiffusionNetwork(Process):
    ''' Models Brownian diffusion using a network of nodes and edges. Each
        node acts as a cellular compartment and each edge connects those
        compartments, indicating where diffusion can occur.

         This :term:`process class` models diffusion based off of Fick's law.
         The following equation is used:

         * Diffusion: :math:`\\frac{dc}{dt} = \\frac{DA}{V} * \\frac{dc}{dx}`

             * :math:`D`: Diffusion constant
             * :math:`A`: Cross-sectional area of edge
             * :math:`V`: Volume of node

         This diffusion equation is solved using implicit Euler to derive a
         matrix, M, that updates the concentration, c(t), on each timestep.

         * Concentration update: :math:`\\c^{t+1} = M^{-1} * c^{t}`

         This process takes in molecular weights in order to solve for
         molecule hydrodynamic radii, which is it turn used to solve for
         diffusion constants. These calculations assume that all molecules are
         spherical proteins. For more information on how this is done, see
         `calculate_rp_from_mw` and `compute_diffusion_constants_from_rp`.
         However, if a molecule radius is known, it can be passed in as
         `radii`. Similarly, if the diffusion constant is known, it can be
         passed in as a property of an edge as `diffusion_constants`.


         .. note::
             This model treats all molecule classes as concentrations and is
             a deterministic solution. This model should only be used when there
             is a sufficient number of molecules such that they can be treated
             deterministically.

         :term:`Ports`:

         * **nodes**: Expects a :term:`store` which is a dict of node names
         (the keys of the dict) to a dict, which has the key value pairs
         for `length`, `volume`, and `molecules`.

         Arguments:
             initial_parameters: A dictionary of configuration options.
                 The following configuration options may be provided:

                 * **nodes** (:py:class:`list`): A list of node names.
                 * **edges** (:py:class:`dict`): Maps edge
                   names (the keys of the dict) to a dict (the
                   values of the dict), which must include the key-value
                   pairs of `nodes` to a list of nodes each edge connects,
                   and `cross_sectional_area` to the area of that edge.
                   Additionally, known diffusion constants can be included as
                   `diffusion_constants` in units of um^2/s, and edge-specific
                   scaling of the diffusion constants can be included as,
                   `diffusion_scaling_constant`.
                 * **mw** (:py:class:`dict`): Maps from
                   names of molecules (the keys of the dict) to their
                   molecular weights in units of fg (the values of the dict).
                 * **mesh_size** (:py:class:`float`): Mesh size in units of nm.
                 * **time_step** (:py:class:`float`): The time step used in
                   units of s.
                 * **radii** (:py:class:`dict`): Maps from molecule names
                   of molecules (the keys of the dict) to their known
                   hydrodynamic radii in units of nm (the values of the dict).
                   This is an optional parameter.
                 * **temp** (:py:class:`float`): Temperature of experiment
                 in units of K. This is an optional parameter.

'''

    name = NAME

    defaults = {
        'nodes': [],
        'edges': {},
        'mw': {},   # in fg
        'mesh_size': 50,    # in nm
        'time_step': 0.1,     # in s
        'radii': {},        # nm
        'temp': 310.15,     # in K
    }

    def __init__(self, parameters=None):
        super(DiffusionNetwork, self).__init__(parameters)
        self.nodes = np.asarray(self.parameters['nodes'])
        self.edges = self.parameters['edges']
        self.mw = self.parameters['mw']
        self.molecule_ids = self.parameters['mw'].keys()
        self.mesh_size = self.parameters['mesh_size']
        self.radii = self.parameters['radii']
        self.temp = self.parameters['temp']
        self.remainder = np.zeros((len(self.nodes), len(self.molecule_ids)))

        # get molecule radii by molecular weights
        self.rp = calculate_rp_from_mw(self.molecule_ids, self.mw)
        for mol_id, r in self.radii.items():
            self.rp[np.where(np.asarray(list(self.molecule_ids)) == mol_id)[0][0]] = r

        # get diffusion constants per molecule
        self.diffusion_constants = compute_diffusion_constants_from_rp(
            self.molecule_ids, self.rp, self.mesh_size, self.edges, self.temp)

    def ports_schema(self):
        '''
        ports_schema returns a dictionary that declares how each state will behave.
        Each key can be assigned settings for the schema_keys declared in Store:

        * `_default`
        * `_updater`
        * `_divider`
        * `_value`
        * `_properties`
        * `_emit`
        * `_serializer`
        '''
        schema = {
            node_id: {
                'volume': {
                    '_default': 1.0,
                },
                'length': {
                    '_default': 1.0,
                },
                'molecules': {
                    '*': {
                        '_default': 0,
                        '_emit': True,
                        '_updater': 'accumulate',
                    }
                },
            } for node_id in self.parameters['nodes']
        }
        return schema

    def next_update(self, timestep, state):
        M = np.asarray([np.identity(len(self.nodes)) for mol in self.molecule_ids])

        # construct M matrix based off of graph, all edges assumed bidirectional
        for edge_id, edge in self.edges.items():
            node_index_1 = np.where(self.nodes == edge['nodes'][0])[0][0]
            node_index_2 = np.where(self.nodes == edge['nodes'][1])[0][0]
            cross_sectional_area = edge['cross_sectional_area']
            vol_1 = state[edge['nodes'][0]]['volume']
            vol_2 = state[edge['nodes'][1]]['volume']
            dx = state[edge['nodes'][0]]['length'] / 2 + state[edge['nodes'][1]]['length'] / 2
            diffusion_constants = array_from(self.diffusion_constants[edge_id])
            alpha = diffusion_constants * (cross_sectional_area / dx) * timestep
            M[:, node_index_1, node_index_1] += alpha / vol_1
            M[:, node_index_2, node_index_2] += alpha / vol_2
            M[:, node_index_1, node_index_2] -= alpha / vol_1
            M[:, node_index_2, node_index_1] -= alpha / vol_2

        # Calculates final concentration after one timestep
        c_initial = np.asarray(
            [np.multiply(array_from(state[node]['molecules']),
                         array_from(self.mw)) / state[node]['volume']
             for node in state])
        c_final = np.asarray([np.matmul(np.linalg.inv(a), c_initial[:, i])
                              for i, a in enumerate(M)]).T

        # Calculates final counts
        volumes = np.asarray(
            [state[node]['volume'] for node in state])
        count_initial = np.asarray([array_from(state[node]['molecules'])
                                    for node in state])
        count_final_unrounded = np.asarray(
            [np.divide(node * volumes[i],
                       array_from(self.mw))
             for i, node in enumerate(c_final)]) + self.remainder
        count_final = np.asarray([saferound(col, 0) for col in
                                  count_final_unrounded.T]).T

        # Keeps track of remainder after rounding counts to integers
        self.remainder = count_final_unrounded - count_final
        delta = np.subtract(count_final, count_initial)

        # Ensures conservation of molecules
        assert (np.array_equal(np.ndarray.sum(count_initial, axis=0),
                np.ndarray.sum(count_final, axis=0))), 'Molecule count is not conserved'

        update = {
            node_id: {
                'molecules': array_to(self.molecule_ids,
                                      delta[np.where(self.nodes ==
                                                     node_id)[0][0]].astype(int)),
            } for node_id in self.nodes
        }
        return update


# TODO: change this to multiple tests and add asserts
def test_diffusion_network_process(out_dir=None):
    # initialize the process by passing initial_parameters
    n = int(1E6)
    molecule_ids = [str(np.round(i, 1)) for i in np.arange(0.1, 19.6, 0.1)]
    initial_parameters = {
        'nodes': ['cytosol_front', 'nucleoid', 'cytosol_rear'],
        'edges': {
            '1': {
                'nodes': ['cytosol_front', 'nucleoid'],
                'cross_sectional_area': np.pi * 0.3 ** 2,
                'mesh': True,
            },
            '2': {
                'nodes': ['nucleoid', 'cytosol_rear'],
                'cross_sectional_area': np.pi * 0.3 ** 2,
                'mesh': True,
            },
            '3': {
                'nodes': ['cytosol_front', 'cytosol_rear'],
                'cross_sectional_area': np.pi * 0.3 ** 2,
            },
            },

        'mw': {str(np.round(i, 1)): np.round(i, 1) for i in np.arange(0.1, 19.6, 0.1)},
        'mesh_size': 50,
        'radii': {str(np.round(i, 1)): np.round(i, 1) for i in np.arange(0.1, 19.6, 0.1)},
    }

    diffusion_network_process = DiffusionNetwork(initial_parameters)

    # run the simulation
    sim_settings = {
        'total_time': 10,
        'initial_state': {
            'cytosol_front': {
                'length': 0.5,
                'volume': 0.25,
                'molecules': {
                    mol_id: n
                    for mol_id in molecule_ids}
            },
            'nucleoid': {
                'length': 1.0,
                'volume': 0.5,
                'molecules': {
                    mol_id: 0
                    for mol_id in molecule_ids}
            },
            'cytosol_rear': {
                'length': 0.5,
                'volume': 0.25,
                'molecules': {
                    mol_id: 0
                    for mol_id in molecule_ids}
            },
        },
    }

    output = simulate_process_in_experiment(diffusion_network_process, sim_settings)
    rp = diffusion_network_process.rp
    diffusion_constants = diffusion_network_process.diffusion_constants

    if out_dir:
        # plot the simulation output
        plot_output(output, sim_settings['initial_state'], out_dir)
        # plot_diff_range(diffusion_constants, rp, out_dir)


# Plots the diffusion constants by molecule sizes for edges with and without mesh
def plot_diff_range(diffusion_constants, rp, out_dir='out'):
    plt.figure()
    plt.plot(np.multiply(rp, 2), array_from(diffusion_constants['1']), color='#d8b365')
    plt.plot(np.multiply(rp, 2), array_from(diffusion_constants['3']), color='#5ab4ac')
    plt.yscale('log')
    plt.xlabel(r'Molecule size ($nm$)')
    plt.ylabel(r'Diffusion constant ($\mu m^2/s$)')
    plt.title('Diffusion constants of molecules')
    plt.legend(['with 50 nm mesh', 'without mesh'])
    out_file = out_dir + '/diffusion_constants.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


# Plots the normalized concentrations for the largest and smallest molecules
def plot_output(output, nodes, out_dir='out'):
    plt.figure()
    colors = ['#d8b365', '#5ab4ac', '#018571']
    large_total = array_from(
        output['cytosol_front']['molecules'])[-1][0] + array_from(
        output['nucleoid']['molecules'])[-1][0] + array_from(
        output['cytosol_rear']['molecules'])[-1][0]
    small_total = array_from(
        output['cytosol_front']['molecules'])[0][0] + array_from(
        output['nucleoid']['molecules'])[0][0] + array_from(
        output['cytosol_rear']['molecules'])[0][0]
    plt.plot(output['time'], np.divide(np.divide(
        array_from(output['cytosol_front']['molecules'])[-1],
        nodes['cytosol_front']['volume']), large_total), color=colors[0])
    plt.plot(output['time'], np.divide(np.divide(
        array_from(output['nucleoid']['molecules'])[-1],
        nodes['nucleoid']['volume']), large_total), color=colors[1])
    plt.plot(output['time'], np.divide(np.divide(
        array_from(output['cytosol_rear']['molecules'])[-1],
        nodes['cytosol_rear']['volume']), large_total), color=colors[2])
    plt.plot(output['time'], np.divide(np.divide(
        array_from(output['cytosol_front']['molecules'])[0],
        nodes['cytosol_front']['volume']), small_total),
        color=colors[0], linestyle='dashed')
    plt.plot(output['time'], np.divide(np.divide(
        array_from(output['nucleoid']['molecules'])[0],
        nodes['nucleoid']['volume']), small_total),
        color=colors[1], linestyle='dashed')
    plt.plot(output['time'], np.divide(np.divide(
        array_from(output['cytosol_rear']['molecules'])[0],
        nodes['cytosol_rear']['volume']), small_total),
        color=colors[2], linestyle='dashed')
    plt.xlabel('time (s)')
    plt.ylabel('Molecule counts')
    plt.title('Diffusion over compartments')
    plt.legend(['Cytosol front: large molecule', 'Nucleoid: large molecule',
                'Cytosol rear: large molecule', 'Cytosol front: small molecule',
                'Nucleoid: small molecule', 'Cytosol rear: small molecule'])
    out_file = out_dir + '/diffusion_large_small.png'
    plt.savefig(out_file)


# This function is modified from spatial_tool.py from WCM
def calculate_rp_from_mw(molecule_ids, mw):
    '''
        This function compute the hydrodynamic radius of a macromolecules from
        its molecular weight. It is important to note that the hydrodynamic
        diameter is mainly used for computation of diffusion constant, and can
        be different from the observed diameter under microscopes or the radius
        of gyration, especially for loose polymers such as RNAs. This function
        is not E coli specific.

        References: Bioinformatics (2012). doi:10.1093/bioinformatics/bts537

        Args:
            molecule_ids: List of molecule ids.
            mw: molecular weight of the macromolecules, units: fg.

        Returns: the hydrodynamic radius (in unit of nm) of the macromolecules
        using the following formula:
            - rp = 0.0515*MW^(0.392) nm (Hong & Lei 2008) (protein)

        These parameters are also possible for other macromolecule types,
        however all molecules are currently assumed to be proteins.
            - rp = 0.0566*MW^(0.38) nm (Werner 2011) (RNA)
            - rp = 0.024*MW^(0.57) nm (Robertson et al 2006) (linear DNA)
            - rp = 0.0125*MW^(0.59) nm (Robertson et al 2006) (circular DNA)
            - rp = 0.0145*MW^(0.57) nm (Robertson et al 2006) (supercoiled DNA)
        '''

    dic_rp = {'protein': (0.0515, 0.392),
              'RNA': (0.0566, 0.38),
              'linear_DNA': (0.024, 0.57),
              'circular_DNA': (0.0125, 0.59),
              'supercoiled_DNA': (0.0145, 0.57),
              }

    r_p0, rp_power = dic_rp['protein']
    fg_to_kDa = 602217364.34

    mw_subset = {
        key: value for key, value in mw.items() if key in molecule_ids
    }

    r_p = np.multiply(
            r_p0, np.power(np.multiply(array_from(mw_subset), fg_to_kDa), rp_power))

    return r_p


# This function is modified from spatial_tool.py from WCM
def compute_diffusion_constants_from_rp(molecule_ids, rp, mesh_size, edges,
                                       temp):
    '''
        Warning: The default values of the 'parameters' are E coli specific.

        This function computes the hypothesized diffusion constant of
        macromolecules within the nucleoid and the cytoplasm region.
        In literature, there is no known differentiation between the diffusion
        constant of a molecule in the nucleoid and in the cytoplasm up to the
        best of our knowledge in 2020. However, there is a good reason why we
        can assume that previously reported diffusion constant are in fact the
        diffusion constant of a protein in the nucleoid region:
        (1) The image traces of a protein within a bacteria usually cross the
            nucleoid regions.
        (2) The nucleoid region, compared to the cytoplasm, should be the main
        limiting factor restricting the magnitude of diffusion constant.
        (3) The same theory of diffusion constant has been implemented to
        mammalian cells, and the term 'rh', the average hydrodynamic radius of
        the biggest crowders, are different in mammalian cytoplasm, and it seems
        to reflect the hydrodynamic radius of the actin filament (note: the
        hydrodynamic radius of actin filament should be computed based on the
        average length of actin fiber, and is not equal to the radius of the
        actin filament itself.) (ref: Nano Lett. 2011, 11, 2157-2163).
        As for E coli, the 'rh' term = 40nm, which may correspond to the 80nm
        DNA fiber. On the other hand, for the diffusion constant of E coli in
        the true cytoplasm, we will expect the value of 'rh' term to be
        approximately 10 nm, which correspond to the radius of active ribosomes.

        Using these terms for scaling a baseline diffusion constant (calculated
        from Enstein-Stokes equation), a cytosol-specific diffusion calculation
        can be obtained).

        Ref: Kalwarczyk, T., Tabaka, M. & Holyst, R.
        Bioinformatics (2012). doi:10.1093/bioinformatics/bts537

        This function computes the hypothesized diffusion constant of
        macromolecules within the nucleoid region by scaling the hypothesized
        cytoplasm diffusion constant. There is evidence that as molecules
        grow in size, they become more excluded from E. coli's nucleoid because
        the DNA polymers form a meshgrid with an estimated mesh size of 50
        nm (ref: Xiang et al., bioRxiv (2020)). The equation implemented here
        assumes that molecules move through the network by encountering openings
        between the DNA meshgrid greater than the hydrodynamic radius.

        Ref: Brian Amsden
        Macromolecules (1999). doi:10.1021/ma980922a

        D_0 = K_B*T/(6*pi*eta_0*rp)
        ln(D_0/D_cyto) = ln(eta/eta_0) = (xi^2/Rh^2 + xi^2/rp^2)^(-a/2)
        D_0 = the diffusion constant of a macromolecule in pure solvent
        eta_0 = the viscosity of pure solvent, in this case, water
        eta = the size-dependent viscosity experienced by the macromolecule.
        xi = average distance between the surface of proteins
        rh = average hydrodynamic radius of the biggest crowders
        a = some constant of the order of 1
        rp = hydrodynamic radius of probed molecule

        In this formula, since we allow the changes in temperature, we also
        consider the viscosity changes of water under different temperature:
        Ref: Dortmund Data Bank
        eta_0 = A*10^(B/(T-C))
        A = 2.414*10^(-5) Pa*sec
        B = 247.8 K
        C = 140 K

        Args:
            molecule_ids: List of molecule ids.
            rp: List of radii corresponding to each molecule id. unit: nm
            mesh_size: Size of meshgrid openings. unit: nm
            edges: Dictionary of edges.
            temp: The temperature of interest. unit: K.

        Returns:
            dc: the diffusion constant of the macromolecule, units: um**2/sec
        '''
    if temp is None:
        temp = 310.15
    parameters = (0.51, 0.53, 10)

    # unpack constants required for the calculation
    xi, a, rh_cyto = parameters  # unit: nm, 1, nm

    ro = mesh_size/2
    rh = rh_cyto
    diffusion_constants = {}
    K_B = scipy.constants.Boltzmann  # Boltzmann constant, unit: J/K

    # calculate viscosity of water based on temperature
    a_visc = 2.414*10**(-5)  # unit: Pa*sec
    b_visc = 247.8  # unit: K
    c_visc = 140  # unit: K
    eta_0 = a_visc*10**(b_visc/(temp - c_visc))  # unit: Pa*sec

    # conversions
    m2_to_um2 = 1E12
    nm_to_m = 1E-9

    # compute dc (diffusion constant)
    dc_0 = np.multiply(
        np.divide(K_B*temp, np.multiply(6*np.pi*eta_0*nm_to_m, rp)), m2_to_um2)
    dc_cyto = np.multiply(dc_0, np.exp(
        -np.power(np.add(xi**2/rh**2, np.divide(xi**2, np.square(rp))), (-a/2))))

    # compute impact from mesh when nucleoid is a node
    dc_nuc = np.multiply(
        dc_cyto, np.exp(np.multiply((-np.pi / 4), np.square(np.divide(rp, ro)))))

    for edge_id, edge in edges.items():
        scaling_factor = 1
        mesh = False
        if 'mesh' in edge:
            mesh = edge['mesh']
        if 'diffusion_scaling_constant' in edge:
            scaling_factor = edge['diffusion_scaling_constant']
        if mesh:
            diffusion_constants[edge_id] = array_to(
                molecule_ids, dc_nuc * scaling_factor)
        else:
            diffusion_constants[edge_id] = array_to(
                molecule_ids, dc_cyto * scaling_factor)
        if 'diffusion_constants' in edge:
            for mol_id, dc in edge['diffusion_constants'].items():
                diffusion_constants[edge_id][mol_id] = dc

    return diffusion_constants


# Helper function
def array_from(d):
    return np.array(list(d.values()))

# Helper function
def array_to(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)}


# run module is run as the main program with python vivarium/process/template_process.py
if __name__ == '__main__':
    # make an output directory to save plots
    out_dir = os.path.join(PROCESS_OUT_DIR, NAME)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    test_diffusion_network_process(out_dir)



