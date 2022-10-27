import pandas as pd

METADATA = {'Time', 'Death', 'Agent ID', 'Boundary', 'Condition', 'Seed',
    'Wall cracked', 'Volume'}
DE_GENES = pd.read_csv('data/marA_binding/model_degenes.csv')
CONCENTRATION_COLUMNS = ['Active MarR', 'Inactive MarR',
    'micF-ompF duplex', 'Inactive 30S subunit', 'Murein tetramer',
    'PBP1a complex', 'PBP1a mRNA', 'PBP1b alpha complex', 'PBP1b mRNA',
    'PBP1b gamma complex', 'AmpC monomer', 'ampC mRNA', 'Active ribosomes']
for gene_data in DE_GENES[['Gene name', 'id', 'monomer_ids']].values:
    if gene_data[0] != 'MicF':
        CONCENTRATION_COLUMNS.append(f'{gene_data[0]} mRNA')
    gene_data[2] = eval(gene_data[2])
    if len(gene_data[2]) > 0:
        monomer_name = gene_data[0][0].upper() + gene_data[0][1:]
        CONCENTRATION_COLUMNS.append(f'{monomer_name} monomer')
CONCENTRATION_COLUMNS = set(CONCENTRATION_COLUMNS)
CONDITION_GROUPINGS = [
    ['Glucose'],
    ['Ampicillin (2 mg/L)', 'Glucose'],
    ['Tetracycline (1.5 mg/L)', 'Glucose']
]
SPLIT_TIME = 11550
MAX_TIME = 26000
