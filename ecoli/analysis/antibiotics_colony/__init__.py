import pandas as pd

METADATA = {'Time', 'Death', 'Agent ID', 'Boundary', 'Condition', 'Seed'}
CONCENTRATION_COLUMNS = {'OmpF monomer', 'ompF mRNA',
    'AcrA monomer', 'acrA mRNA', 'AcrB monomer', 'acrB mRNA', 'TolC monomer',
    'tolC mRNA', 'MicF RNA', 'Active MarR', 'Inactive MarR',
    'micF-ompF duplex', 'Inactive 30S subunit', 'Murein tetramer',
    'PBP1a complex', 'PBP1a mRNA', 'PBP1b alpha complex', 'PBP1b mRNA',
    'PBP1b gamma complex', 'AmpC monomer', 'ampC mRNA', 'marR mRNA',
    'marA mRNA', 'MarA monomer', 'Active ribosomes'}
CONDITION_GROUPINGS = [
    ['Glucose'],
    ['Ampicillin (2 mg/L)', 'Glucose'],
    ['Tetracycline (1.5 mg/L)', 'Glucose']
]
DE_GENES = pd.read_csv('data/marA_binding/model_degenes.csv')
SPLIT_TIME = 11550
