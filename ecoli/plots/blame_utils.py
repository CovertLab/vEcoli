def get_bulk_processes(topology):
    # Get relevant processes (those affecting bulk)
    bulk_processes = {}
    for process, ports in topology.items():
        # Only care about evolver molecule count changes
        if '_requester' in process:
            continue
        for port, path in ports.items():
            if 'bulk' in path:
                if process not in bulk_processes:
                    bulk_processes[process] = []

                bulk_processes[process].append(port)
    
    return bulk_processes
