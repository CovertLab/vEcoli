from vivarium.core.experiment import pp


def mrna_comparison_plot(data, out_dir='out'):
    # separate data by port
    bulk = data['bulk']
    unique = data['unique']
    listeners = data['listeners']
    process_state = data['process_state']
    environment = data['environment']

    # print(bulk)
    # print(unique.keys())
    pp(listeners['mass'])

