import os
import numpy as np
import pickle
import time
from copy import deepcopy
import matplotlib.pyplot as plt

from vivarium.core.process import Process
from vivarium.core.engine import Engine
from ecoli.library.schema import (add_elements, numpy_schema, attrs, 
    counts, dict_value_schema, NUMPY_DEFAULTS,
    arrays_from, array_from, arrays_to, array_to)


def generate_initial_states():
    a, z = np.array(['a', 'z']).view('int32')
    random_names = np.random.randint(low=a, high=z, size=500000, dtype='int32').view(f'U10')

    initial_numpy_state = {
        'bulk': np.array([(
                    random_name,
                    np.random.random(9),
                    np.random.randint(1000000))
                for random_name in random_names],
                dtype=NUMPY_DEFAULTS['bulk'].dtype),
        'unique': {
            'RNA': np.array([(
                    np.random.randint(1000000),
                    np.random.randint(1000000),
                    np.random.randint(1000000),
                    i,
                    np.random.randint(2),
                    np.random.randint(2),
                    np.random.randint(2),
                    np.random.random(9),
                    1)
                for i in range(15000)],
                dtype=NUMPY_DEFAULTS['RNAs'].dtype),
            'active_ribosome': np.array([(
                    np.random.randint(1000000),
                    np.random.randint(1000000),
                    np.random.randint(1000000),
                    i,
                    np.random.randint(1000000),
                    np.random.random(9),
                    1)
                for i in range(15000)],
                dtype=NUMPY_DEFAULTS['active_ribosome'].dtype),
        }
    }
    
    with open('out/numpy_init.pkl', 'wb') as f:
        pickle.dump(initial_numpy_state, f)

    initial_default_state = {
        'bulk': {
            name: count for (name, _, count) in initial_numpy_state['bulk'].tolist()
        },
        'unique': {
            'RNA': {
                unique_index: {
                    'TU_index': tu_index,
                    'transcript_length': transcript_length,
                    'RNAP_index': RNAP_index,
                    'unique_index': unique_index,
                    'is_mRNA': is_mRNA,
                    'is_full_transcript': is_full_transcript,
                    'can_translate': can_translate,
                    'submass': submass
                } for (tu_index, transcript_length, RNAP_index, 
                    unique_index, is_mRNA, is_full_transcript, 
                    can_translate, submass, _) in initial_numpy_state[
                        'unique']['RNA'].tolist()
            },
            'active_ribosome': {
                unique_index: {
                    'protein_index': protein_index,
                    'peptide_length': peptide_length,
                    'mRNA_index': mRNA_index,
                    'unique_index': unique_index,
                    'pos_on_mRNA': pos_on_mRNA,
                    'submass': submass
                } for (protein_index, peptide_length, mRNA_index, 
                    unique_index, pos_on_mRNA, submass, _
                    ) in initial_numpy_state['unique'][
                        'active_ribosome'].tolist()
            }
        }
    }
    
    with open('out/dict_init.pkl', 'wb') as f:
        pickle.dump(initial_default_state, f)
        
    return initial_numpy_state, initial_default_state

class NumpyProcess(Process):
    name = 'NumpyProcess'
    topology = {
        'bulk': ('bulk',),
        'RNAs': ('unique', 'RNA'),
        'active_ribosome': ('unique', 'active_ribosome')
    }
    
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.bulk_ports = self.parameters['bulk_ports']
        self.num_to_add = self.parameters['num_to_add']
        self.num_to_del = self.parameters['num_to_del']
        self.unique_index = self.parameters['unique_index']
    
    def ports_schema(self):
        return {
            'bulk': numpy_schema('bulk'),
            'RNAs': numpy_schema('RNAs'),
            'active_ribosome': numpy_schema('active_ribosome')
        }
    
    def next_update(self, timestep, states):
        update = {'bulk': [], 'RNAs': [], 'active_ribosome': []}
        for idx, actions in self.bulk_ports:
            bulk_counts = counts(states[f'bulk'], idx)
            update['bulk'] += [(action, idx, {'count': 1}) for action in actions]
        
        active_rna_idx = np.where(states['RNAs']['_entryState'])[0]
        TU_index, can_translate, unique_index = attrs(states['RNAs'], np.array([
            'TU_index', 'can_translate', 'unique_index']))
        active_ribosome_idx = np.where(states['active_ribosome']['_entryState'])[0]
        protein_index, peptide_length, pos_on_mRNA = attrs(
            states['active_ribosome'], np.array(['protein_index', 
                'peptide_length', 'pos_on_mRNA']))
        update['RNAs'].append(('new', [], {
            'can_translate': np.ones(self.num_to_add), 
            'unique_index': np.arange(self.unique_index, self.unique_index+self.num_to_add)}))
        update['RNAs'].append(('del', active_rna_idx[:self.num_to_del], {}))
        update['active_ribosome'].append(('new', [], {
            'peptide_length': np.ones(self.num_to_add), 
            'mRNA_index': np.ones(self.num_to_add),
            'unique_index': np.arange(self.unique_index, self.unique_index+self.num_to_add)}))
        self.unique_index += self.num_to_add
        update['active_ribosome'].append(('del', active_ribosome_idx[:self.num_to_del], {}))
        
        return update
    
class LastNumpyProcess(NumpyProcess):
    def next_update(self, timestep, states):
        update = {'bulk': [], 'RNAs': [], 'active_ribosome': []}
        for idx, actions in self.bulk_ports:
            bulk_counts = counts(states[f'bulk'], idx)
            update['bulk'] += [(action, idx, {'count': 1}) for action in actions]
        
        active_rna_idx = np.where(states['RNAs']['_entryState'])[0]
        TU_index, can_translate, unique_index = attrs(states['RNAs'], np.array([
            'TU_index', 'can_translate', 'unique_index']))
        active_ribosome_idx = np.where(states['active_ribosome']['_entryState'])[0]
        protein_index, peptide_length, pos_on_mRNA = attrs(
            states['active_ribosome'], np.array(['protein_index', 
                'peptide_length', 'pos_on_mRNA']))
        update['RNAs'].append(('new', [], {
            'can_translate': np.ones(self.num_to_add), 
            'unique_index': np.arange(self.unique_index, self.unique_index+self.num_to_add)}))
        update['RNAs'].append(('del', active_rna_idx[:self.num_to_del], {}))
        update['RNAs'].append('last')
        update['active_ribosome'].append(('new', [], {
            'peptide_length': np.ones(self.num_to_add), 
            'mRNA_index': np.ones(self.num_to_add),
            'unique_index': np.arange(self.unique_index, self.unique_index+self.num_to_add)}))
        self.unique_index += self.num_to_add
        update['active_ribosome'].append(('del', active_ribosome_idx[:self.num_to_del], {}))
        update['active_ribosome'].append('last')
        
        return update

class DictProcess(Process):
    name = 'DictProcess'
    topology = {}
    
    def __init__(self, parameters=None):
        super().__init__(parameters)
        self.bulk_ports = self.parameters['bulk_ports']
        self.num_to_add = self.parameters['num_to_add']
        self.num_to_del = self.parameters['num_to_del']
        self.unique_index = self.parameters['unique_index']
        self.topology = {
            'RNAs': ('unique', 'RNA'),
            'active_ribosome': ('unique', 'active_ribosome')
        }
        for i in range(len(self.bulk_ports)):
            self.topology[f'bulk_{i}'] = ('bulk',)
    
    def ports_schema(self):
        bulk_ports = {
            f'bulk_{i}': {idx: {'_default': 0, '_emit': True} for idx in indices}
            for i, (indices, _) in enumerate(self.bulk_ports)
        }
        unique_ports = {
            'RNAs': dict_value_schema('RNAs'),
            'active_ribosome': dict_value_schema('active_ribosome')
        }
        return {**bulk_ports, **unique_ports}
    
    def next_update(self, timestep, states):
        update = {}
        for i, (_, actions) in enumerate(self.bulk_ports):
            port_name = f'bulk_{i}'
            bulk_counts = array_from(states[port_name])
            final_update = np.zeros_like(bulk_counts, dtype=np.int64)
            # Mixing in 'set' updates causes different behavior from
            # Numpy update model
            for action in actions:
                if action == 'inc':
                    final_update += 1
                elif action == 'dec':
                    final_update -= 1
            update[f'bulk_{i}'] = array_to(states[port_name], final_update)
        
        TU_index, can_translate, unique_index = arrays_from(
            states['RNAs'].values(), ['TU_index', 'can_translate', 'unique_index'])
        protein_index, peptide_length, pos_on_mRNA = arrays_from(
            states['active_ribosome'].values(), ['protein_index', 
                'peptide_length', 'pos_on_mRNA'])
        rnas_update = arrays_to(self.num_to_add, {
            'can_translate': np.ones(self.num_to_add), 
            'unique_index': np.arange(self.unique_index, self.unique_index+self.num_to_add)
        })
        update['RNAs'] = add_elements(rnas_update, 'unique_index')
        update['RNAs']['_delete'] = list(states['RNAs'].keys())[:self.num_to_del]
        ribosome_update = arrays_to(self.num_to_add, {
            'peptide_length': np.ones(self.num_to_add), 
            'mRNA_index': np.ones(self.num_to_add),
            'unique_index': np.arange(self.unique_index, self.unique_index+self.num_to_add)
        })
        update['active_ribosome'] = add_elements(ribosome_update, 'unique_index')
        update['active_ribosome']['_delete'] = list(states['active_ribosome'].keys())[:self.num_to_del]
        self.unique_index += self.num_to_add
        
        return update 


def run_processes(add_del_ratio=1, scale=1000, bulk_ops=[(5000, ('dec',)), 
    (5000, ('inc',))], n_proc=1, validate=True, total_time=500):
    """Run two functionally identical experiments, first using Numpy structured 
    array stores and next using dictionary or traditional stores.

    Args:
        add_del_ratio (float, optional): Ratio of unique molecules added to 
            deleted each timestep. Defaults to 1.
        scale (int, optional): Number of unique molecules to delete each 
            timestep. Defaults to 1000.
        bulk_ops (list, optional): List of tuples where each tuple represents
            a port to the bulk store. The first element in each tuple is the
            number of molecules in that store and the second is a tuple containing
            the types of operations to perform on that store: 'inc' to increment
            or 'dec' to decrement. Defaults to [(5000, ('dec',)), (5000, ('inc',))].
        n_proc (int, optional): Number of processes (all identical). Defaults to 1.
        validate (bool, optional): Compare bulk and unique molecule counts at 
            end of both experiments. Defaults to True.
        total_time (int, optional): Simulation time (timestep=1). Defaults to 500.

    Returns:
        numpy_time (float): wallclock time of Numpy-based experiment
        dict_time (float): wallclock time of dictionary-based experiment
    """
    if not os.path.exists('out/numpy_init.pkl') or (
        not os.path.exists('out/dict_init.pkl')):
        numpy_init, dict_init = generate_initial_states()
    else:
        with open('out/numpy_init.pkl', 'rb') as f:
            numpy_init = pickle.load(f)
        with open('out/dict_init.pkl', 'rb') as f:
            dict_init = pickle.load(f) 
    
    bulk_keys = np.array(list(dict_init['bulk'].keys()))
    
    numpy_processes = {}
    numpy_topologies = {}
    dict_processes = {}
    dict_topologies = {}
    for proc_idx in range(n_proc):
        bulk_indices = np.arange(numpy_init['bulk'].size)
        bulk_ports = [(np.random.choice(bulk_indices, n_molecules, replace=False), 
                   action) for n_molecules, action in bulk_ops]
        numpy_config = {
            'bulk_ports': bulk_ports,
            'num_to_add': int(scale*add_del_ratio),
            'num_to_del': scale,
            'name': str(proc_idx),
            'unique_index': int(15000 + proc_idx*1000000000)
        }
        if n_proc - proc_idx == 1:
            # Cap off unique updates
            numpy_processes[str(proc_idx)] = LastNumpyProcess(numpy_config)
        else:
            numpy_processes[str(proc_idx)] = NumpyProcess(numpy_config)
        numpy_topologies[str(proc_idx)] = numpy_processes[str(proc_idx)].topology
        
        dict_config = deepcopy(numpy_config)
        dict_config['bulk_ports'] = [(bulk_keys[idx], action) 
            for idx, action in dict_config['bulk_ports']]
        dict_processes[str(proc_idx)] = DictProcess(dict_config)
        dict_topologies[str(proc_idx)] = dict_processes[str(proc_idx)].topology
    
    experiment = Engine(
        processes=numpy_processes,
        topology=numpy_topologies,
        initial_state=numpy_init,
        progress_bar=True
    )
    start_time = time.time()
    experiment.update(total_time)
    np_time = time.time() - start_time
    numpy_data = experiment.emitter.saved_data[total_time]
    
    
    experiment = Engine(
        processes=dict_processes,
        topology=dict_topologies,
        initial_state=dict_init,
        progress_bar=True
    ) 
    start_time = time.time()
    experiment.update(total_time)
    dict_time = time.time() - start_time
    dict_data = experiment.emitter.saved_data[total_time]
    
    if validate:
        validate_data(dict_data, numpy_data)

    return np_time, dict_time

def validate_data(dict_data, np_data):
    bulk_keys = list(dict_data['bulk'].keys())
    bulk_values = [dict_data['bulk'][key] for key in bulk_keys]
    np_count = [np_data['bulk']['count'][np_data['bulk']['id'] == key][0] for key in bulk_keys]
    assert np.array_equal(bulk_values, np_count)
    assert len(dict_data['unique']['RNA']) == np_data['unique']['RNA']['_entryState'].sum()
    assert len(dict_data['unique']['active_ribosome']) == np_data['unique']['active_ribosome']['_entryState'].sum()
    
def scan_ratio(lower, upper):
    np_times = np.zeros(50)
    dict_times = np.zeros(50)
    add_del_ratio = np.linspace(lower, upper, 50)
    for i, ratio in enumerate(add_del_ratio):
        np_times[i], dict_times[i] = run_processes(add_del_ratio=ratio, total_time=100)
    plt.plot(add_del_ratio, np_times, label="Numpy")
    plt.plot(add_del_ratio, dict_times, label="Dict")
    plt.xlabel('Ratio of add to delete ops.')
    plt.ylabel('Runtime (s)')
    plt.tight_layout()
    plt.savefig('out/np_dict/test_ratio.png')
    plt.close()
    
def scan_scale(lower, upper):
    np_times = np.zeros(50)
    dict_times = np.zeros(50)
    scales = np.linspace(lower, upper, 50, dtype=int)
    for i, scale in enumerate(scales):
        np_times[i], dict_times[i] = run_processes(scale=scale, total_time=100)
    plt.plot(scales, np_times, label="Numpy")
    plt.plot(scales, dict_times, label="Dict")
    plt.xlabel('Approx. scale of add/delete ops.')
    plt.ylabel('Runtime (s)')
    plt.tight_layout()
    plt.savefig('out/np_dict/test_scale.png')
    plt.close()
    
def scan_bulk_ops(port_sizes=[1], n_ops=[1], n_ports=[1]):
    np_times = np.zeros((len(port_sizes), len(n_ops), len(n_ports)))
    dict_times = np.zeros((len(port_sizes), len(n_ops), len(n_ports)))
    for i, port_size in enumerate(port_sizes):
        for j, n_op in enumerate(n_ops):
            for k, n_port in enumerate(n_ports):
                n_op = int(n_op/2)
                bulk_ops = [(port_size, ('dec',)*n_op + ('inc',)*n_op)]*n_port
                np_times[i, j, k], dict_times[i, j, k] = run_processes(bulk_ops=bulk_ops, total_time=100)
    np.save('out/np_dict/np_bulk_ops.npy', np_times)
    np.save('out/np_dict/dict_bulk_ops.npy', dict_times)
    
def plot_heatmaps(dict_data, np_data, config):
    n_ports = config['n_ports'].tolist()
    n_ops = config['n_ops'].tolist()
    port_sizes = config['port_sizes'].tolist()
    fig, axs = plt.subplots(5, 1, sharey=True, figsize=(6, 25))
    for plot_idx, n_op in enumerate(n_ops):
        dict_i = dict_data[:, plot_idx, :]
        np_i = np_data[:, plot_idx, :]
        ratio = dict_i/np_i
        axs[plot_idx].imshow(ratio)
        axs[plot_idx].set_xticks(np.arange(len(n_ports)), labels=n_ports)
        axs[plot_idx].set_yticks(np.arange(len(port_sizes)), labels=port_sizes)
        axs[plot_idx].set_ylabel('Size of each port')
        axs[plot_idx].set_xlabel('# of ports')
        
        for i in range(len(port_sizes)):
            for j in range(len(n_ops)):
                axs[plot_idx].text(j, i, np.around(ratio[i, j], 2), ha='center', va='center', color='w')
                
        axs[plot_idx].set_title(f'Dict Runtime / Numpy Runtime: {n_op} discrete operation(s)')
        
    fig.tight_layout()
    plt.savefig('out/np_dict/bulk_ops.png', dpi=400)
        
        
    
if __name__ == '__main__':
    os.makedirs('out/np_dict/', exist_ok=True)
    bulk_ops_config = {
        'port_sizes': np.linspace(1, 11000, 5, dtype=int),
        'n_ops': np.linspace(1, 11000, 5, dtype=int),
        'n_ports': np.linspace(1, 110, 5, dtype=int)
    }
    scan_bulk_ops(**bulk_ops_config)
    dict_data = np.load('out/np_dict/dict_bulk_ops.npy')
    np_data = np.load('out/np_dict/np_bulk_ops.npy')
    plot_heatmaps(dict_data, np_data, bulk_ops_config)
    scan_scale(1, 100000)
    scan_ratio(0, 10)
