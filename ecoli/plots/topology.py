
def get_ecoli_master_topology_settings():
    """plot settings for ecoli_master's topology plot"""
    process_row = -4
    process_distance = 0.9
    settings = {
        'graph_format': 'hierarchy',
        'dashed_edges': True,
        'show_ports': False,
        'node_size': 12000,
        'font_size': 12,
        'coordinates': {
            'tf_binding': (1*process_distance, process_row),
            'transcript_initiation': (2*process_distance, process_row),
            'transcript_elongation': (3*process_distance, process_row),
            'rna_degradation': (4*process_distance, process_row),
            'polypeptide_initiation': (5*process_distance, process_row),
            'polypeptide_elongation': (6*process_distance, process_row),
            'complexation': (7*process_distance, process_row),
            'two_component_system': (8*process_distance, process_row),
            'equilibrium': (9*process_distance, process_row),
            'protein_degradation': (10*process_distance, process_row),
            'metabolism': (11*process_distance, process_row),
            'chromosome_replication': (12 * process_distance, process_row),
            'mass': (13*process_distance, process_row),
            'mrna_counts': (14 * process_distance, process_row),
            'divide_condition': (15*process_distance, process_row),
        },
        'node_labels': {
            # processes
            'tf_binding': 'tf\nbinding',
            'transcript_initiation': 'transcript\ninitiation',
            'transcript_elongation': 'transcript\nelongation',
            'rna_degradation': 'rna\ndegradation',
            'polypeptide_initiation': 'polypeptide\ninitiation',
            'polypeptide_elongation': 'polypeptide\nelongation',
            'complexation': 'complexation',
            'two_component_system': 'two component\nsystem',
            'equilibrium': 'equilibrium',
            'protein_degradation': 'protein\ndegradation',
            'metabolism': 'metabolism',
            'chromosome_replication': 'chromosome\nreplication',
            'mass': 'mass',
            'mrna_counts': 'mrna\ncounts',
            'divide_condition': 'division',
            # stores
            'unique\nchromosome_domain': 'unique\nchromosome\ndomain',
        },
        'remove_nodes': [
            'aa_enzymes',
            'listeners\nmass\ncell_mass',
            'process_state',
            'listeners\nfba_results',
            'listeners\nenzyme_kinetics',
            'listeners\nmass',
            'listeners\nribosome_data',
            'listeners\nfba_results',
            'listeners\nequilibrium_listener',
            'listeners\nrna_degradation_listener',
            'listeners\ntranscript_elongation_listener',
            'process_state\npolypeptide_elongation',
        ]
    }
    return settings
