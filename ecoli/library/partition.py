# logging
from ecoli.library.logging import make_logging_process

# partitioning
from ecoli.processes.partition import Partition

def generate_partition_proc(blame, proc_conf, ECOLI_PROCESSES, timestep, 
                            random_state):
    """
    Replaces default generate_processes function in composer to
    utilize partition process wrapper make_partition_proc

    Args:
        blame (bool): Whether to enable logging
        proc_conf (dict): Configuration paramaters for all processes
        ECOLI_PROCESSES (dict): Map process names to process classes
        timestep (float): Base time to increment between updates
        random_state (numpy.random.RandomState): Used to generate seed
            for partitioning

    Returns:
        Dict: All instantiated and configured processes
    """
    partition_conf = {'time_step': timestep/2,
                      'molecule_names': list(proc_conf['mass'][
                        'molecular_weights'].keys()),
                      'proc_names': list(proc_conf.keys()),
                      'seed': random_state.randint(2**31)}
    partition = {
        'partition': make_partition_proc(Partition)(
            partition_conf)}
    for process in proc_conf.keys():
        proc_conf[process]['time_step'] = timestep/2
    procs = {
        proc_name: make_partition_proc(proc)(proc_conf[proc_name])
        for (proc_name, proc) in ECOLI_PROCESSES.items()
        # Removing polypeptide elongation makes dry mass updates reasonable
        if proc_name != 'polypeptide_elongation'}
        
    if blame:
        procs = {
            proc_name: make_logging_process(proc)
            for (proc_name, proc) in procs.items()
        }
    all_procs = {**partition, **procs}
    return all_procs

def make_partition_proc(process_class):
    """
    Wrapper function to add partitioning specific ports and logic to
    all instantiated processes

    Args:
        process_class (Process): Instantiated process to be inherited
        and modified

    Returns:
        Process: Modified Process with partitioning configured
    """
    partition_process = type(f"Partition_{process_class.__name__}",
                           (process_class,),
                           {})
    # set __class__ manually for super()
    __class__ = partition_process 

    def ports_schema(self):
        """
        Adds the following ports
        
            'totals': View of total counts of molecules requested by procs.
            'requested': View of molecule counts requested by each process
            'allocated': View of molecule counts partitioned to each process
            'timesteps': Number of half-timesteps (hTS) that have elapsed

        Returns:
            Dict: Updated ports schema
        """ 
        ports = {}
        ports['requested'] = {'_default' : {}, '_updater': 'set', '_emit': True}
        ports['allocated'] = {'_default' : {}, '_updater': 'set', '_emit': True}
        ports['timesteps'] = {'_default' : 0, '_emit': True}
        # get the original port structure
        ports.update(super().ports_schema())
        return ports

    def next_update(self, timestep, states):
        """
        Run Processes with halved timestep (hTS) to enable staggered 
        calculate_request, partitioning, and evolve_state calls. 
        Processes and Derivers run according to the following time table.
        
        Partition: partition every 2 hTS starting from 1 hTS
        Metabolism: run every 2 hTS starting from 2 hTS
        Other derivers: update states every 1 hTS starting from 0
        Processes: calculate requests every 2 hTS starting from 1 hTS,
            evolve state every 2 hTS starting from 2 hTS

        Args:
            timestep (float): Base time to increment between process updates
            states (dict): View of all connected stores

        Returns:
            Dict: Update to apply to states (derivers apply immediately,
                processes do not)
        """
        # Derivers have 0 timestep, run at end of every timestep (inc. t=0),
        # and update states immediately after running
        if self.name in ['ecoli-mass', 'divide_condition']:
            return super().next_update(2, states)
        if states['timesteps']==0:
            return super().calculate_request(timestep*2, states)
        if states['timesteps']%2:
            return super().calculate_request(timestep*2, states)
            # IMPORTANT: Run update with full timestep (not halved)
        return super().evolve_state(timestep*2, states)
        
    partition_process.ports_schema = ports_schema
    partition_process.next_update = next_update

    return partition_process

def generate_partition_topology(blame, ECOLI_TOPOLOGY):
    """
    Replaces default generate_topology function in composer to
        utilize connect stores required for partitioning

    Args:
        blame (bool): Whether to enable logging
        ECOLI_TOPOLOGY (dict): Map process names to stores (and paths)

    Returns:
        Dict: Completely wired process/store topology for partitioning
    """
    proc_topo = {}
    for proc_id in ECOLI_TOPOLOGY:
        proc_topo[proc_id] = {}
    for proc_id, ports in ECOLI_TOPOLOGY.items():
        proc_topo[proc_id] = ports
        proc_topo[proc_id]['timesteps'] = ('partitioning', 'timesteps')
        if proc_id not in ['mass', 'divide_condition']:
            if blame:
                proc_topo[proc_id]['log_update'] = ('log_update', proc_id,)
            proc_topo[proc_id]['requested'] = ('partitioning', 'requested', proc_id)
            proc_topo[proc_id]['allocated'] = ('partitioning', 'allocated', proc_id)
        
    partition_topo = {'totals': ('bulk',),
                      'requested': ('partitioning', 'requested',),
                      'allocated': ('partitioning', 'allocated',),
                      'timesteps': ('partitioning', 'timesteps')}
    
    total = {'partition': partition_topo,
             **proc_topo}
    return total
    