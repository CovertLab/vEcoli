#!/usr/bin/env python

"""
mRNACounts Listener
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from vivarium.core.process import Deriver
from ecoli.library.schema import arrays_from


def test_mrna_counts_listener():
	from ecoli.composites.ecoli_master import run_ecoli
	data = run_ecoli(total_time=2)
	assert(type(data['listeners']['mRNA_counts'][0]) == list)
	assert(type(data['listeners']['mRNA_counts'][1]) == list)


class mRNACounts(Deriver):
	"""
	Listener for the counts of each mRNA species.
	"""
	name = 'mRNA_counts_listener'

	defaults = {
		'rna_ids': [],
		'mrna_indexes': [],
	}

	def __init__(self, parameters=None):
		super().__init__(parameters)

		# Get IDs and indexes of all mRNAs
		self.all_RNA_ids = self.parameters['rna_ids']
		self.mRNA_indexes = self.parameters['mrna_indexes']

	def ports_schema(self):
		return {
			'listeners': {
				'mRNA_counts': {
					'_default': [],
					'_updater': 'set',
					'_emit': True}
				},
			'RNAs': {
				'*': {
					'unique_index': {'_default': 0, '_updater': 'set'},
					'TU_index': {'_default': 0, '_updater': 'set'},
					'transcript_length': {'_default': 0, '_updater': 'set', '_emit': True},
					'is_mRNA': {'_default': 0, '_updater': 'set'},
					'is_full_transcript': {'_default': 0, '_updater': 'set'},
					'can_translate': {'_default': 0, '_updater': 'set'},
					'RNAP_index': {'_default': 0, '_updater': 'set'}}
			},
		}

	def next_update(self, timestep, states):
		# Get attributes of mRNAs
		tu_indexes, can_translate = arrays_from(states['RNAs'].values(), ['TU_index', 'can_translate'])

		# Get counts of all mRNAs
		mrna_counts = np.bincount(
			tu_indexes[can_translate],
			minlength=len(self.all_RNA_ids))[self.mRNA_indexes]

		return {
			'listeners': {
				'mRNA_counts': mrna_counts
			}
		}


if __name__ == '__main__':
	test_mrna_counts_listener()