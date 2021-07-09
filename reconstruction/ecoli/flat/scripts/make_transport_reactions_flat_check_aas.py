'''
Makes a tsv file with all transport reaction ids

goes through all reactions in reaction.tsv and compares substrate and product ids

'''

import os
import re
import csv
import json
from reconstruction.spreadsheets import JsonReader

TSV_DIALECT = csv.excel_tab

REACTIONS_FILE = os.path.join("reconstruction", "ecoli", "flat", "reactions.tsv")
OUT_FILE = os.path.join("out", "transport_reactions.tsv")




AA_TRANSPORT_FILE = os.path.join(
	'estimate_kinetics', 'data', 'aa_exchange_reactions.json'
	)

with open(AA_TRANSPORT_FILE, 'r') as f:
	aa_transport_reactions = json.loads(f.read())
aa_rxn_ids = aa_transport_reactions.keys()

# make list of transport reactions
transport_reactions = []
with open(REACTIONS_FILE, 'rU') as tsvfile:
	reader = JsonReader(tsvfile, dialect=TSV_DIALECT)
	for row in reader:
		reaction_id = row["reaction id"]
		stoichiometry = row["stoichiometry"]

		# get substrates and products
		substrates = [mol_id for mol_id, coeff in stoichiometry.iteritems() if coeff < 0]
		products = [mol_id for mol_id, coeff in stoichiometry.iteritems() if coeff > 0]
		substrates_no_loc = [re.sub("[[@*&?].*[]@*&?]", "", mol_id) for mol_id in substrates]
		products_no_loc = [re.sub("[[@*&?].*[]@*&?]", "", mol_id) for mol_id in products]

		overlap_no_loc = set(substrates_no_loc) & set(products_no_loc)

		# if overlap between substrate and product names with no location:
		for mol_id in list(overlap_no_loc):
			sub = [mol for mol in substrates if mol_id in mol]
			prod = [mol for mol in products if mol_id in mol]
			overlap = set(sub) & set(prod)

			# if there is no overlap between those substrates and products with locations included
			if len(overlap) == 0:
				# print('sub ' + str(sub))
				# print('prod ' + str(prod))
				transport_reactions.append(reaction_id.encode('ascii','ignore'))

# sort reactions to save them in ordered list
transport_reactions = list(set(transport_reactions))
transport_reactions = sorted(transport_reactions)



t_r = set(transport_reactions)  # 479
a_r = set(aa_rxn_ids)  # 72


aXr = a_r.intersection(t_r)  # 70
not_included = a_r - aXr  # set([u'TRANS-RXN0-537 (reverse)', u'RXN0-1924 (reverse)'])

import ipdb; ipdb.set_trace()





# save list of transport reactions
if os.path.exists(OUT_FILE):
	os.remove(OUT_FILE)

with open(OUT_FILE, 'a') as tsvfile:
	writer = csv.writer(tsvfile, quoting=csv.QUOTE_NONNUMERIC, delimiter='\t')
	writer.writerow(["reaction id"])
	for reaction_id in transport_reactions:
		append_line = [reaction_id]
		writer.writerow(append_line)
