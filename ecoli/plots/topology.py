
def get_ecoli_nonpartition_topology_settings():
    """plot settings for ecoli_master's topology plot"""
    process_row = -4
    process_x_offset = -1
    process_distance = 0.9
    settings = {
        'graph_format': 'hierarchy',
        'dashed_edges': True,
        'show_ports': False,
        'node_size': 12000,
        'font_size': 12,
        'coordinates': {
            'ecoli-tf-binding': (1*process_distance+process_x_offset, process_row),
            'ecoli-transcript-initiation': (2*process_distance+process_x_offset, process_row),
            'ecoli-transcript-elongation': (3*process_distance+process_x_offset, process_row),
            'ecoli-rna-degradation': (4*process_distance+process_x_offset, process_row),
            'ecoli-polypeptide-initiation': (5*process_distance+process_x_offset, process_row),
            'ecoli-polypeptide-elongation': (6*process_distance+process_x_offset, process_row),
            'ecoli-complexation': (7*process_distance+process_x_offset, process_row),
            'ecoli-two-component-system': (8*process_distance+process_x_offset, process_row),
            'ecoli-equilibrium': (9*process_distance+process_x_offset, process_row),
            'ecoli-protein-degradation': (10*process_distance+process_x_offset, process_row),
            'ecoli-metabolism': (11*process_distance+process_x_offset, process_row),
            'ecoli-chromosome-replication': (12*process_distance+process_x_offset, process_row),
            'ecoli-chromosome-structure': (13*process_distance+process_x_offset, process_row),
            'ecoli-mass-listener': (14*process_distance+process_x_offset, process_row),
            'mRNA_counts_listener': (15*process_distance+process_x_offset, process_row),
            'divide_condition': (16*process_distance+process_x_offset, process_row),
        },
        'node_labels': {
            # processes
            'ecoli-tf-binding': 'tf\nbinding',
            'ecoli-transcript-initiation': 'transcript\ninitiation',
            'ecoli-transcript-elongation': 'transcript\nelongation',
            'ecoli-rna-degradation': 'rna\ndegradation',
            'ecoli-polypeptide-initiation': 'polypeptide\ninitiation',
            'ecoli-polypeptide-elongation': 'polypeptide\nelongation',
            'ecoli-complexation': 'complexation',
            'ecoli-two-component-system': 'two component\nsystem',
            'ecoli-equilibrium': 'equilibrium',
            'ecoli-protein-degradation': 'protein\ndegradation',
            'ecoli-metabolism': 'metabolism',
            'ecoli-chromosome-replication': 'chromosome\nreplication',
            'ecoli-chromosome-structure': 'chromosome\nstructure',
            'ecoli-mass-listener': 'mass',
            'mRNA_counts_listener': 'mrna\ncounts',
            'divide_condition': 'division',
            # stores
            'unique\nfull_chromosome': 'unique\nchromosome',
            'unique\nchromosome_domain': 'unique\nchromosome\ndomain',
            'unique\nactive_replisome': 'unique\nactive\nreplisome',
            'unique\nactive_RNAP': 'unique\nactive\nRNAP',
            'unique\nactive_ribosome': 'unique\nactive\nribosome',
            'unique\nDnaA_box': 'unique\nDnaA',
        },
        'remove_nodes': [
            'aa_enzymes',
            'process_state',
            'process_state\npolypeptide_elongation',
            'environment\nexchange_data',
            'listeners\nmass\ncell_mass',
            'listeners\nfba_results',
            'listeners\nenzyme_kinetics',
            'listeners\nmass',
            'listeners\nribosome_data',
            'listeners\nfba_results',
            'listeners\nRnapData',
            'listeners\ntranscript_elongation_listener',
            'listeners\nrna_degradation_listener',
            'listeners\nequilibrium_listener',
            'listeners\nreplication_data',
            'listeners\nrnap_data',
        ]
    }
    return settings


def get_ecoli_partition_topology_settings():
    evolver_row = -6
    allocator_row = -7
    requester_row = -8
    process_distance = 0.9
    settings = {
        'graph_format': 'hierarchy',
        'dashed_edges': True,
        'show_ports': False,
        'node_size': 12000,
        'coordinates': {
            'ecoli-tf-binding_evolver': (1 * process_distance, evolver_row),
            'ecoli-tf-binding_requester': (1 * process_distance, requester_row),

            'ecoli-transcript-initiation_evolver': (2 * process_distance, evolver_row),
            'ecoli-transcript-initiation_requester': (2 * process_distance, requester_row),

            'ecoli-transcript-elongation_evolver': (3 * process_distance, evolver_row),
            'ecoli-transcript-elongation_requester': (3 * process_distance, requester_row),

            'ecoli-rna-degradation_evolver': (4 * process_distance, evolver_row),
            'ecoli-rna-degradation_requester': (4 * process_distance, requester_row),

            'ecoli-polypeptide-initiation_evolver': (5 * process_distance, evolver_row),
            'ecoli-polypeptide-initiation_requester': (5 * process_distance, requester_row),

            'ecoli-polypeptide-elongation_evolver': (6 * process_distance, evolver_row),
            'ecoli-polypeptide-elongation_requester': (6 * process_distance, requester_row),

            'ecoli-complexation_evolver': (7 * process_distance, evolver_row),
            'ecoli-complexation_requester': (7 * process_distance, requester_row),

            'ecoli-two-component-system_evolver': (8 * process_distance, evolver_row),
            'ecoli-two-component-system_requester': (8 * process_distance, requester_row),

            'ecoli-equilibrium_evolver': (9 * process_distance, evolver_row),
            'ecoli-equilibrium_requester': (9 * process_distance, requester_row),

            'ecoli-protein-degradation_evolver': (10 * process_distance, evolver_row),
            'ecoli-protein-degradation_requester': (10 * process_distance, requester_row),

            'ecoli-chromosome-replication_evolver': (11 * process_distance, evolver_row),
            'ecoli-chromosome-replication_requester': (11 * process_distance, requester_row),

            'ecoli-chromosome-structure_evolver': (12 * process_distance, evolver_row),
            'ecoli-chromosome-structure_requester': (12 * process_distance, requester_row),
            'ecoli-chromosome-structure': (12 * process_distance, evolver_row),

            'ecoli-metabolism_evolver': (13 * process_distance, evolver_row),
            'ecoli-metabolism_requester': (13 * process_distance, requester_row),
            'ecoli-metabolism': (13 * process_distance, evolver_row),

            'ecoli-mass-listener': (14 * process_distance, evolver_row),
            'mRNA_counts_listener': (15 * process_distance, evolver_row),
            'divide_condition': (16 * process_distance, evolver_row),

            'allocator': (6 * process_distance, allocator_row),
        },
        'node_labels': {
            # processes
            'ecoli-tf-binding_requester': 'tf\nbinding\nrequester',
            'ecoli-tf-binding_evolver': 'tf\nbinding\nevolver',

            'ecoli-transcript-initiation_requester': 'transcript\ninitiation\nrequester',
            'ecoli-transcript-initiation_evolver': 'transcript\ninitiation\nevolver',

            'ecoli-transcript-elongation_requester': 'transcript\nelongation\nrequester',
            'ecoli-transcript-elongation_evolver': 'transcript\nelongation\nevolver',

            'ecoli-rna-degradation_requester': 'rna\ndegradation\nrequester',
            'ecoli-rna-degradation_evolver': 'rna\ndegradation\nevolver',

            'ecoli-polypeptide-initiation_requester': 'polypeptide\ninitiation\nrequester',
            'ecoli-polypeptide-initiation_evolver': 'polypeptide\ninitiation\nevolver',

            'ecoli-polypeptide-elongation_requester': 'polypeptide\nelongation\nrequester',
            'ecoli-polypeptide-elongation_evolver': 'polypeptide\nelongation\nevolver',

            'ecoli-complexation_requester': 'complexation\nrequester',
            'ecoli-complexation_evolver': 'complexation\nevolver',

            'ecoli-two-component-system_requester': 'two component\nsystem\nrequester',
            'ecoli-two-component-system_evolver': 'two component\nsystem\nevolver',

            'ecoli-equilibrium_requester': 'equilibrium\nrequester',
            'ecoli-equilibrium_evolver': 'equilibrium\nevolver',

            'ecoli-protein-degradation_requester': 'protein\ndegradation\nrequester',
            'ecoli-protein-degradation_evolver': 'protein\ndegradation\nevolver',

            'ecoli-chromosome-replication_requester': 'chromosome\nreplication\nrequester',
            'ecoli-chromosome-replication_evolver': 'chromosome\nreplication\nevolver',

            'ecoli-chromosome-structure_requester': 'chromosome\nstructure\nrequester',
            'ecoli-chromosome-structure_evolver': 'chromosome\nstructure\nevolver',

            'ecoli-metabolism_requester': 'metabolism\nrequester',
            'ecoli-metabolism_evolver': 'metabolism\nevolver',

            'ecoli-mass-listener': 'mass',
            'mRNA_counts_listener': 'mrna\ncounts',
            'divide_condition': 'division',
        },
        'remove_nodes': [
            'allocate\necoli-polypeptide-elongation\nenvironment\namino_acids',
            'request\necoli-polypeptide-elongation\nenvironment\namino_acids',
            'aa_enzymes',
            'process_state',
            'process_state\npolypeptide_elongation',
            'environment\nexchange_data',
            'listeners\nmass\ncell_mass',
            'listeners\nfba_results',
            'listeners\nenzyme_kinetics',
            'listeners\nmass',
            'listeners\nribosome_data',
            'listeners\nfba_results',
            'listeners\nRnapData',
            'listeners\ntranscript_elongation_listener',
            'listeners\nrna_degradation_listener',
            'listeners\nequilibrium_listener',
            'listeners\nreplication_data',
            'listeners\nrnap_data',
        ],
    }
    return settings
