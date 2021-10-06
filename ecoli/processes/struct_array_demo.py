import time
import numpy as np
import matplotlib.pyplot as plt

from vivarium.core.composition import simulate_process
from vivarium.core.process import Process

from ecoli.library.schema import add_elements, array_from


class ArrayDict():
    # Reference: see https://docs.python.org/3/reference/datamodel.html?emulating-container-types#emulating-container-types

    EMPTY = "__EMPTY__"

    def __init__(self, dtype, capacity=1000):
        if dtype[0][0] != "_key":
            dtype = [("key", "<U16")] + dtype
        self.struct_array = np.zeros([capacity],
                                     dtype=dtype)
        # Correct?
        self.struct_array["key"] = EMPTY
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass

    def __setitem__(self, key, value):
        pass

    def __delitem__(self, key):
        pass

    def __missing__(self, key):
        pass

    def __iter__(self):
        pass

    def __reversed__(self):
        pass

    def __contains__(self, item):
        pass


def dict_value_updater(current, update):
    '''
    Updater which translates add_item and delete_item -style updates
    into operations on a dictionary.

    Expects current to be a dictionary, with no restriction on the types of objects
    stored within it, and no defaults. For enforcing expectations/defaults, try
    make_dict_value_updater(**defaults).
    '''
    result = current

    if update.get("add_items"):
        for operation in update["add_items"]:
            result[operation["key"]] = operation["state"]

    for k in update.get("remove_items", {}):
        result.pop(k)

    return result

def make_dict_value_updater(**defaults):
    '''
    Returns an updater which translates add_item and delete_item -style updates
    into operations on a dictionary.

    The returned updater expects current to be a dictionary. Each added item
    can have a subset of the provided defaults as its keys;
    entries not provided will have values supplied by the defaults.
    '''

    def dict_value_updater(current, update):
        result = current

        for operation in update.get("add_items", {}):
            state = operation["state"]
            if not set(state.keys()).issubset(defaults.keys()):
                raise Exception(f"Attempted to write state with keys not included in defaults")
            state = {**defaults, **state}
            result[operation["key"]] = state

        for k in update.get("remove_items", {}):
            result.pop(k)

        return result

    return dict_value_updater

def struct_array_updater(current, update):
    '''
    Updater which translates _add and _delete -style updates
    into operations on a structured array.

    Expects current to be a numpy structured array, whose dtype will be
    maintained by the result returned. The first element of the dtype must be
    ("_key", "<U16"), an index targeted by _add and _delete operations.

    TODO:
      - inefficient use of numpy arrays, should instead pre-allocate and expand when necessary
        (cf. array list data structure)
      - in the next_update method, retrieving data from state requires different code, since
        dict and structured array interfaces are different. To avoid having to change code in each
        process, we could, instead of using an updater, make a custom data structure with dict
        interface but internally using a struct array.
    '''
    result = current

    if update.get("add_items"):
        added = np.array([((v['key'],)
                           + tuple(v['state'][name]
                                   for name in current.dtype.names
                                   if name != "_key"))
                         for v in update["add_items"]],
                         dtype=current.dtype)
        result = np.append(current, added)

    if update.get("remove_items"):
        mask = np.isin(result['_key'], update['remove_items'])
        result = result[~mask]

    return result


class StructArrayDemo(Process):
    '''
    Process allowing for easy comparison of _add and _delete operations
    with/without using a struct array.

    Adds/deletes an active_RNAP molecule at exponentially distributed random intervals
    with the rate parameters specified.
    '''

    name = "StructArrayDemo"
    defaults = {
        'mode': "default",
        'rate_add': 0.5,
        'rate_delete': 0.2,
        'seed': 0
    }

    DTYPE = np.dtype([("_key", '<U16'),  # necessary for _add, _delete operations. We restrict keys to be 16 characters or less.
                      ('unique_index', 'int'),
                      ('domain_index', 'int'),
                      ('coordinates', 'int'),
                      ('direction', 'bool')])

    def __init__(self, parameters=None):
        super().__init__(parameters)

        self.mode = self.parameters['mode']
        self.rate_add = self.parameters['rate_add']
        self.rate_delete = self.parameters['rate_delete']
        self.seed = self.parameters['seed']

        self.random = np.random.default_rng(self.seed)

        self.next_id = 0
        self.time_til_next_add = 0
        self.time_til_next_delete = 0

    def ports_schema(self):
        if self.mode == "default":
            return {
                'active_RNAPs': {
                    '*': {
                        'unique_index': {'_default': 0, '_updater': 'set', '_emit': True},
                        'domain_index': {'_default': 0, '_updater': 'set', '_emit': True},
                        'coordinates': {'_default': 0, '_updater': 'set', '_emit': True},
                        'direction': {'_default': 0, '_updater': 'set', '_emit': True}
                    }
                }
            }
        elif self.mode == "struct_array":
            return {
                'active_RNAPs': {
                    '_default': np.array([], dtype=StructArrayDemo.DTYPE),
                    '_updater': struct_array_updater,
                    '_emit': True
                }
            }
        elif self.mode == "dict_value":
            return {
                'active_RNAPs' : {
                    '_default' : {},
                    '_updater' : make_dict_value_updater(
                        unique_index=0,
                        domain_index=0,
                        coordinates=0,
                        direction=True),
                    "_emit" : True
                    }
            }
        

    def next_update(self, timestep, states):
        update = {'active_RNAPs': {}}

        self.time_til_next_add -= timestep
        self.time_til_next_delete -= timestep

        if self.time_til_next_add <= 0:
            key = "_add"  if self.mode == "default" else "add_items"
            update['active_RNAPs'][key] = [{
                'key': f'{self.next_id}',
                'state': {
                    'unique_index': self.next_id,
                    'domain_index': self.random.integers(0, 2),
                    'coordinates': self.random.integers(-10000, 10000),
                    'direction': bool(self.random.integers(0, 2))
                }
            }]
            self.next_id += 1
            self.time_til_next_add = self.random.exponential(1/self.rate_add)

        if self.time_til_next_delete <= 0:
            if len(states['active_RNAPs']) > 0:
                key = "_delete" if self.mode == "default" else "remove_items"
                update['active_RNAPs'][key] = [self.random.choice(
                    (list(states['active_RNAPs'].keys())
                     if self.mode in {"default", "dict_value"}
                     else states['active_RNAPs']['_key']),
                    replace=False)]
            self.time_til_next_delete = self.random.exponential(
                1/self.rate_delete)

        return update


def test_struct_array_updater(mode="struct_array",
                              rate_add=0.5,
                              rate_delete=0.2,
                              total_time=10):
    process_config = {
        'mode': mode,
        'rate_add': rate_add,
        'rate_delete': rate_delete,
        'seed': 0
    }
    process = StructArrayDemo(process_config)

    initial_state = {}
    if mode == "default" or mode == "dict_value":
        initial_state['active_RNAPs'] = {}
    elif mode == "struct_array":
        initial_state['active_RNAPs'] = np.empty((0, len(StructArrayDemo.DTYPE)),
                                                 dtype=StructArrayDemo.DTYPE)

    settings = {
        'total_time': total_time,
        'initial_state': initial_state,
        'return_raw_data' : True}

    tick = time.perf_counter()
    data = simulate_process(process, settings)
    tock = time.perf_counter()

    print(f'Run in mode "{mode}" took {tock - tick} seconds.')

    return (tock - tick), data


def main():
    # Parameter values to sweep through.
    sweep = {
        'mode': ["default", "struct_array", "dict_value"],
        'add_to_delete_rate_ratio': [1, 5, 10, 100],
        'total_time': [10, 50, 100, 200, 500, 750, 1000]
    }

    # Sweep through parameters, saving runtime.
    result = np.zeros((len(sweep['mode']),
                       len(sweep['add_to_delete_rate_ratio']),
                       len(sweep['total_time']),
                       1))

    rate_delete = 0.1
    for i, rate_ratio in enumerate(sweep['add_to_delete_rate_ratio']):
        for j, total_time in enumerate(sweep['total_time']):
            data = {}
            for k, mode in enumerate(sweep['mode']):
                time, data[mode] = test_struct_array_updater(mode=mode,
                                                             rate_add=rate_ratio * rate_delete,
                                                             rate_delete=rate_delete,
                                                             total_time=total_time)
                result[k, i, j, 0] = time

            # Assertions to make sure data matches across modes: ===============================

            # Make default data conformable to dict_value data for comparison
            data['default'] = {
                k : v if v != {} else {'active_RNAPs' : {}}
                for k, v in data['default'].items()
                }
            
            # default vs. dict_value updater
            assert data['dict_value'] == data['default']

            # default vs. struct_array updater
            for time, snapshot in data['default'].items():
                struct_array_snapshot = data['struct_array'][time]['active_RNAPs']
                for idx, rnap in enumerate(snapshot['active_RNAPs'].values()):
                    assert all(array_from(rnap) == np.array(struct_array_snapshot[idx][1:]))

    # Plots:
    # One plot for with-struct-arrays, one without
    # Y axis: time elapsed
    # X axis: total simulation time
    # Line color gradient: ratio of add rate to delete rate

    fig, axs = plt.subplots(1, len(sweep['mode']), sharey=True)
    n_lines = len(sweep['add_to_delete_rate_ratio'])
    for i, rate_ratio in enumerate(sweep['add_to_delete_rate_ratio']):
        for j, mode in enumerate(sweep['mode']):
            axs[j].plot(sweep['total_time'], result[j, i, :],
                        label=rate_ratio, color=(1-i/n_lines, i/n_lines, 0.25))
            axs[j].set_title(
                f'Runtime using\n"{mode}"')
            axs[j].set_xlabel("Simulation length (s)")
            axs[j].set_xticks(sweep['total_time'])
            axs[j].tick_params(axis='x', labelrotation=45)
            axs[j].legend(title="add : delete\nrate ratio")

    axs[0].set_ylabel("Runtime (s)")
    plt.tight_layout()
    plt.savefig('out/struct_array_sweep.png')


if __name__ == "__main__":
    main()
