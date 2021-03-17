from ecoli.processes.diffusion_network import calculate_rp_from_mw
from ecoli.processes.diffusion_network import compute_diffusion_constants_from_rp
import pickle
import matplotlib.pyplot as plt
import numpy as np

def check_fig_pickle(path):
    try:
        fig = pickle.load(open(path, 'rb'))
    except (OSError, IOError) as e:
        fig = plt.figure()
        pickle.dump(fig, open(path, 'wb'))
    return fig


'''
These plots are all intended to be run on `ecoli_spatial.py` and can be called
from `run_spatial_ecoli()`. It is important to note that these plots are
specific to the configuration defined in `test_spatial_ecoli.py`. A few of
of these plotting functions that compare output for different polyribosome
assumptions should be run in a specific way to achieve the desired output plot.
The simulation needs to be run three times each time with a different polyribosome
assumption declared in the configuration. For intended output, start with the
spherical assumption, then the mrna assumption and lastly the linear assumption.
'''

# Plots the diffusion of a single molecule
def plot_single_molecule_diff(output, mol_id, mesh_size, nodes):
    plt.figure()
    plt.plot(output['time'], np.divide(
        output['cytosol_front']['molecules'][mol_id],
        nodes['cytosol_front']['volume']), color='#d8b365')
    plt.plot(output['time'], np.divide(
        output['nucleoid']['molecules'][mol_id],
        nodes['nucleoid']['volume']), color='#5ab4ac')
    plt.xlabel('time (s)')
    plt.ylabel(r'Concentration (molecules / $\mu m^3$)')
    plt.title(f'Diffusion of {mol_id} over compartments with {mesh_size} nm mesh')
    plt.legend(['Cytosol front', 'Nucleoid'])
    out_file = 'out/single_molecule.png'
    plt.savefig(out_file)


# Plots the diffusion of a set of three large molecules
def plot_large_molecules(output, mol_ids, mesh_size, nodes):
    plt.figure()
    linestyle = ['solid', 'dashed', 'dotted']
    legend = []
    for i, mol_id in enumerate(mol_ids):
        plt.plot(output['time'], np.divide(np.divide(
            output['cytosol_front']['molecules'][mol_id],
            nodes['cytosol_front']['volume']),
            output['cytosol_front']['molecules'][mol_id][0]), color='#d8b365',
                 linestyle=linestyle[i])
        plt.plot(output['time'], np.divide(np.divide(
            output['nucleoid']['molecules'][mol_id],
            nodes['nucleoid']['volume']),
            output['cytosol_front']['molecules'][mol_id][0]), color='#5ab4ac',
                 linestyle=linestyle[i])
        legend.append(f'{mol_id} in pole')
        legend.append(f'{mol_id} in nucleoid')
    plt.xlabel('time (s)')
    plt.ylabel('Normalized concentration (% total concentration)')
    plt.title(f'Diffusion of large molecules over compartments with {mesh_size} nm mesh')
    plt.legend(legend)
    out_file = 'out/large_molecules.png'
    plt.savefig(out_file)


# Plots nucleoid diffusion of polyribosomes on same plot for different assumptions/runs
def plot_nucleoid_diff(output, nodes, polyribosome_assumption):
    x = np.arange(1, 11)
    total_molecules = array_from(
        output['nucleoid']['molecules'])[:, 0] + array_from(
        output['cytosol_front']['molecules'])[:, 0] + array_from(
        output['cytosol_rear']['molecules'])[:, 0]
    if polyribosome_assumption == 'mrna':
        fig = check_fig_pickle('out/nucleoid_diff.pickle')
        plt.plot(
            x, np.average(array_from(output['nucleoid']['molecules']), axis=1) /
               total_molecules *
               nodes['nucleoid']['volume'] / nodes['cytosol_front']['volume'],
            color='#5ab4ac')
        pickle.dump(fig, open('out/nucleoid_diff.pickle', 'wb'))
    elif polyribosome_assumption == 'linear':
        fig = check_fig_pickle('out/nucleoid_diff.pickle')
        plt.plot(
            x, np.average(array_from(output['nucleoid']['molecules']), axis=1) /
               total_molecules *
               nodes['nucleoid']['volume'] / nodes['cytosol_front']['volume'],
            color='#018571')
        pickle.dump(fig, open('out/nucleoid_diff.pickle', 'wb'))
    else:  # spherical assumption
        fig = check_fig_pickle('out/nucleoid_diff.pickle')
        plt.plot(
            x, np.average(array_from(output['nucleoid']['molecules']), axis=1) /
            total_molecules *
            nodes['nucleoid']['volume'] / nodes['cytosol_front']['volume'],
            color='#d8b365')
        labels = [str(val) if val < 10 else str(val) + '+' for val in np.arange(1, 11)]

        plt.xticks(np.arange(1, 11), labels)
        plt.xlabel('Number of ribosomes on polyribosome')
        plt.ylabel('Ratio of nucleoid localization to pole localization')
        handles = [plt.Line2D([0], [0], color=c) for c in ['#d8b365', '#5ab4ac', '#018571']]
        plt.legend(
            handles, ['Spherical protein assumption', 'mRNA assumption', 'Linear ribosomes assumption'])
        plt.title('Polyribosome nucleoid localization over 5 min')
        pickle.dump(fig, open('out/nucleoid_diff.pickle', 'wb'))
    out_file = 'out/nucleoid_diff.png'
    plt.savefig(out_file, dpi=300)


# Plots diffusion of polyribosomes in a pole and the nucleoid
def plot_polyribosomes_diff(output, mesh_size, nodes, filename):
    fig = plt.figure()
    groups = ['polyribosome_1[c]', 'polyribosome_2[c]', 'polyribosome_3[c]',
                  'polyribosome_4[c]','polyribosome_5[c]', 'polyribosome_6[c]',
                  'polyribosome_7[c]', 'polyribosome_8[c]', 'polyribosome_9[c]',
                  'polyribosome_>=10[c]']
    colors = ['#543005', '#8c510a', '#bf812d', '#dfc27d', '#f6e8c3',
              '#c7eae5', '#80cdc1', '#35978f', '#01665e', '#003c30']
    time = np.divide(output['time'], 60)
    for i, mol_id in enumerate(groups):
        plt.plot(time, np.divide(
            output['cytosol_front']['molecules'][mol_id],
            output['cytosol_front']['molecules'][mol_id][0]), linestyle='dashed',
            color=colors[i], label=str(mol_id + ' in pole'))
        plt.plot(time, np.divide(np.divide(
            output['nucleoid']['molecules'][mol_id],
            output['cytosol_front']['molecules'][mol_id][0]),
            nodes['nucleoid']['volume']/nodes['cytosol_front']['volume']),
                 color=colors[i], label=str(mol_id + ' in nucleoid'))
    plt.xlabel('time (min)')
    plt.ylabel('Normalized concentration (% total concentration)')
    plt.title(f'Diffusion of polyribosomes with mesh of {str(mesh_size)} nm')
    out_file = filename or 'out/polyribosomes.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


# Plots the average nucleotide availability
def plot_NT_availability(avg_NT_per_ribosome):
    '''
    Note that ribosomes are allowed to overlap with each other in wcEcoli, which
    is why this is an important analysis to look at. The average footprint of
    ribosomes in E. coli on mRNA is assumed to be about 25 on average with a
    range of 15-40 NT.

        - Ref: Mohammad et al., eLife (2019)
    '''
    fig = plt.figure()
    tot = len(avg_NT_per_ribosome)
    (counts, bins) = np.histogram(avg_NT_per_ribosome, bins=np.arange(25, 3500, 50))
    plt.hist(bins[:-1], bins, weights=counts / tot, color='#5ab4ac')
    plt.xlabel('Average number of available NTs per ribosome')
    plt.ylabel('Percentage of total number of polyribosomes (%)')
    plt.title(f'Available NTs per ribosome on polyribosomes (n = {tot})')
    out_file = 'out/avg_NT_per_ribosomes.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

    fig = plt.figure()
    tot = len(avg_NT_per_ribosome)
    (counts, bins) = np.histogram(avg_NT_per_ribosome, bins=np.arange(25, 100))
    plt.hist(bins[:-1], bins, weights=counts / tot, color='#5ab4ac')
    plt.axvline(x=15, color='k', linestyle='dashed')
    plt.axvline(x=40, color='k', linestyle='dashed')
    plt.xlabel('Average number of available NTs per ribosome')
    plt.ylabel('Percentage of total number of polyribosomes (%)')
    plt.title(f'Available NTs per ribosome on polyribosomes (n = {tot})')
    out_file = 'out/avg_NT_per_ribosomes_zoom.png'
    handles = [plt.Line2D([0], [0], color=c, linestyle='dashed') for c in ['k']]
    plt.legend(handles, ['Minimum and maximum ribosome footprints'])
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


# Performs calculations for and calls bulk and polyribosome molecules characterization plots
def plot_molecule_characterizations(ecoli, initial_config):
    molecule_ids = ecoli.config['nodes']['cytosol_front']['molecules'].keys()
    mesh_size = ecoli.config['mesh_size']
    rp = calculate_rp_from_mw(molecule_ids, ecoli.bulk_molecular_weights)
    ecoli.config['edges']['3'] = {
        'nodes': ['cytosol_front', 'cytosol_rear'],
        'cross_sectional_area': np.pi * 0.3 ** 2,
    }
    molecule_ids = np.asarray(list(molecule_ids))
    for mol_id, r in ecoli.config['radii'].items():
        rp[np.where(molecule_ids == mol_id)[0][0]] = r
    dc = compute_diffusion_constants_from_rp(
        ecoli.config['nodes']['cytosol_front']['molecules'].keys(),
        rp, ecoli.config['mesh_size'], ecoli.config['edges'], ecoli.config['temp'])

    if initial_config['include_bulk']:
        cytosol_mask = [i for i, mol_id in enumerate(
        ecoli.config['nodes']['cytosol_front']['molecules'].keys())
                    if '[c]' in mol_id]
        rp_cytosol = np.asarray(rp)[cytosol_mask]
        dc_cytosol = array_from(dc['1'])[cytosol_mask]
        dc_cytosol_no_mesh = array_from(dc['3'])[cytosol_mask]
        plot_bulk_molecules(rp_cytosol, dc_cytosol, dc_cytosol_no_mesh, mesh_size)

    if initial_config['include_polyribosomes']:
        polyribosome_mask = [i for i, mol_id in enumerate(
            ecoli.config['nodes']['cytosol_front']['molecules'].keys())
                             if 'polyribosome' in mol_id]
        rp_polyribosome = np.asarray(rp)[polyribosome_mask]
        dc_polyribosome = array_from(dc['1'])[polyribosome_mask]
        dc_polyribosome_no_mesh = array_from(dc['3'])[polyribosome_mask]
        total_molecules_polyribosome = np.add(
            array_from(ecoli.config['nodes']['cytosol_front']['molecules']
                       )[polyribosome_mask],
            array_from(ecoli.config['nodes']['cytosol_rear']['molecules']
                       )[polyribosome_mask])
        polyribosome_assumption = initial_config['polyribosome_assumption']
        plot_polyribosomes(rp_polyribosome, rp_polyribosome,
                           dc_polyribosome, dc_polyribosome_no_mesh,
                           total_molecules_polyribosome, ecoli.config['mesh_size'],
                           polyribosome_assumption)


# Plots characteristics of polyribosomes: counts, sizes, and diffusion constants
def plot_polyribosomes(rp, radii, dc, dc_no_mesh, total_molecules, mesh_size,
                       polyribosome_assumption):
    x = np.arange(1, 11)
    labels = [str(val) if val < 10 else str(val) + '+' for val in np.arange(1, 11)]
    if polyribosome_assumption == 'mrna':
        fig = check_fig_pickle('out/polyribosomes_sizes.pickle')
        plt.plot(x, np.multiply(radii, 2), color='#5ab4ac')
        pickle.dump(fig, open('out/polyribosomes_sizes.pickle', 'wb'))
    elif polyribosome_assumption == 'linear':
        fig = check_fig_pickle('out/polyribosomes_sizes.pickle')
        plt.plot(x, np.multiply(radii, 2), color='#018571')
        pickle.dump(fig, open('out/polyribosomes_sizes.pickle', 'wb'))
    else:  # spherical assumption
        fig = check_fig_pickle('out/polyribosomes_sizes.pickle')
        mesh = np.full(len(x), mesh_size)
        plt.plot(x, np.multiply(rp, 2), color='#d8b365')

        plt.plot(x, mesh, linestyle='dashed', color='k')
        plt.xticks(np.arange(1, 11), labels)
        plt.xlabel('Number of ribosomes')
        plt.ylabel('Polyribosome size (nm)')
        plt.title('Sizes of polyribosomes')
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ['#d8b365', '#5ab4ac', '#018571']]
        plt.legend(handles, ['spherical protein assumption', 'mRNA assumption',
                             'linear ribosomes assumption', 'mesh size'])
        pickle.dump(fig, open('out/polyribosomes_sizes.pickle', 'wb'))

    out_file = 'out/polyribosomes_sizes.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

    fig = plt.figure()
    # Note: this value is hardcoded
    avg_num_ribosomes = [1,  2,  3,  4,  5, 6,  7,  8,  9, 11.87]
    tot = sum(total_molecules)
    total_ribosomes = np.multiply(total_molecules, avg_num_ribosomes)
    tot_rib = np.sum(total_ribosomes)
    plt.bar(x-0.2, total_molecules/tot, width=0.4, color='#5ab4ac', align='center')
    plt.bar(x+0.2, total_ribosomes/tot_rib, width=0.4, color='#d8b365', align='center')
    plt.xticks(np.arange(1, 11), labels)
    plt.xlabel('Number of ribosomes')
    plt.ylabel('Percentage of total count (%)')
    plt.legend(['Polyribosomes', '70S ribosomes'])
    plt.title(f'Polyribosome counts (n = {tot})')
    out_file = 'out/polyribosomes_counts.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

    if polyribosome_assumption == 'linear':
        fig = check_fig_pickle('out/polyribosomes_dc.pickle')
        plt.plot(x, dc, color='#018571')
        plt.plot(x, dc_no_mesh, color='#018571', linestyle='dashed')
        plt.legend(['spherical protein assumption: 50 nm mesh',
                    'spherical protein assumption: no mesh',
                    'mRNA assumption: 50 nm mesh',
                    'mRNA assumption: no mesh',
                    'linear ribosomes assumption: 50 nm mesh',
                    'linear ribosomes assumption: no mesh'])
    elif polyribosome_assumption == 'mrna':
        fig = check_fig_pickle('out/polyribosomes_dc.pickle')
        plt.plot(x, dc, color='#5ab4ac')
        plt.plot(x, dc_no_mesh, color='#5ab4ac', linestyle='dashed')
    else:
        fig = check_fig_pickle('out/polyribosomes_dc.pickle')
        tot = len(dc)
        plt.plot(x, dc, color='#d8b365')
        plt.plot(x, dc_no_mesh, color='#d8b365', linestyle='dashed')
        plt.xticks(np.arange(1, 11), labels)
        plt.xlabel('Number of ribosomes')
        plt.ylabel(r'Diffusion constant ($\mu m^2 / s$)')
        plt.yscale('log')
        plt.title(f'Diffusion constants of polyribosomes (n = {tot})')
        plt.legend(['spherical protein assumption: 50 nm mesh',
                    'spherical protein assumption: no mesh',
                    'mRNA assumption: 50 nm mesh',
                    'mRNA assumption: no mesh',
                    'linear ribosomes assumption: 50 nm mesh',
                    'linear ribosomes assumption: no mesh'])
        pickle.dump(fig, open('out/polyribosomes_dc.pickle', 'wb'))
    out_file = 'out/polyribosomes_dc.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


# Plots characteristics of bulk molecules: sizes and diffusion constants
def plot_bulk_molecules(rp, dc, dc_no_mesh, mesh_size):
    fig = plt.figure()
    tot = len(rp)
    size = np.round(np.multiply(rp, 2))
    (counts, bins) = np.histogram(size, bins=range(int(max(size))))
    plt.hist(bins[:-1], bins, weights=counts/tot, color='#5ab4ac')
    plt.xlabel('Molecule size (nm)')
    plt.ylabel('Percentage of total number of molecules (%)')
    plt.title(f'Sizes of bulk molecules (n = {tot})')
    out_file = 'out/bulk_molecules_sizes.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

    fig = plt.figure()
    new_size, new_dc, new_dc_no_mesh = zip(*sorted(zip(size, dc, dc_no_mesh)))
    new_size, idx = np.unique(new_size, return_index=True)
    plt.plot(new_size, np.asarray(new_dc)[idx], color='#d8b365')
    plt.plot(new_size, np.asarray(new_dc_no_mesh)[idx], color='#5ab4ac')
    plt.ylabel(r'Diffusion constant ($\mu m^2 / s$)')
    plt.yscale('log')
    plt.xlabel('Molecule size (nm)')
    plt.title(f'Diffusion constants of bulk molecules (n = {tot})')
    plt.legend([f'with {str(mesh_size)} nm mesh', 'without mesh'])
    out_file = 'out/bulk_molecules_dc.png'
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)


# Helper functions
def array_from(d):
    return np.array(list(d.values()))


def array_to(keys, array):
    return {
        key: array[index]
        for index, key in enumerate(keys)}