"""
====================
RNA Counts Listener
====================
"""

import numpy as np
from ecoli.library.schema import numpy_schema, attrs, listener_schema
from vivarium.core.process import Step

from ecoli.processes.registries import topology_registry


NAME = "RNA_counts_listener"
TOPOLOGY = {
    "listeners": ("listeners",),
    "RNAs": ("unique", "RNA"),
    "global_time": ("global_time",),
    "timestep": ("timestep",),
}
topology_registry.register(NAME, TOPOLOGY)


class RNACounts(Step):
    """
    Listener for the counts of each mRNA and rRNA transcription units and
        cistrons. Includes the counts of both partial and full transcripts.

    mRNA counts include only mRNAs that have not been targeted for degradation.
    mRNAs that are being degraded in the 5' to 3' direction (see
    ``directional_mrna_decay`` in
    :py:class:`~ecoli.library.sim_data.LoadSimData`) are counted separately in
    ``decaying_mRNA_counts``, including those that can no longer bind
    ribosomes but are kept until transcription and ribosomes on them finish.
    Cistron counts are calculated from transcription unit counts, so all
    cistrons of a partial transcript are counted.
    ``available_mRNA_cistron_counts`` instead counts only the cistrons that can
    bind new ribosomes: cistrons that have been transcribed and have not been
    passed by the degradation front.
    """

    name = NAME
    topology = TOPOLOGY

    defaults = {
        "rna_ids": [],
        "mrna_indexes": [],
        "cistron_start_end_pos_in_tu": {},
        "directional_mrna_decay": False,
        "time_step": 1,
        "emit_unique": False,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)

        # Get IDs and indexes of all mRNA and rRNA transcription units
        self.all_TU_ids = self.parameters["all_TU_ids"]
        self.mRNA_indexes = self.parameters["mRNA_indexes"]
        self.mRNA_TU_ids = self.parameters["mRNA_TU_ids"]
        self.rRNA_indexes = self.parameters["rRNA_indexes"]
        self.rRNA_TU_ids = self.parameters["rRNA_TU_ids"]

        # Get IDs and indexes of all mRNA and rRNA cistrons
        self.all_cistron_ids = self.parameters["all_cistron_ids"]
        self.cistron_is_mRNA = self.parameters["cistron_is_mRNA"]
        self.mRNA_cistron_ids = self.parameters["mRNA_cistron_ids"]
        self.cistron_is_rRNA = self.parameters["cistron_is_rRNA"]
        self.rRNA_cistron_ids = self.parameters["rRNA_cistron_ids"]

        # Get mapping matrix between TUs and cistrons
        self.cistron_tu_mapping_matrix = self.parameters["cistron_tu_mapping_matrix"]

        self.directional_mrna_decay = self.parameters["directional_mrna_decay"]

        # Get start positions of each cistron within each TU, sorted by TU,
        # to count the cistrons available on partial and decaying mRNAs
        cistron_TU_pairs = sorted(
            (TU_index, cistron_index, start_position)
            for (cistron_index, TU_index), (start_position, _) in self.parameters[
                "cistron_start_end_pos_in_tu"
            ].items()
        )
        pair_TU_indexes = np.array([pair[0] for pair in cistron_TU_pairs], dtype=int)
        self.pair_cistron_indexes = np.array(
            [pair[1] for pair in cistron_TU_pairs], dtype=int
        )
        self.pair_start_positions = np.array(
            [pair[2] for pair in cistron_TU_pairs], dtype=int
        )
        self.n_cistrons_per_TU = np.bincount(
            pair_TU_indexes, minlength=len(self.all_TU_ids)
        )
        self.TU_pair_offsets = np.concatenate(
            ([0], np.cumsum(self.n_cistrons_per_TU)[:-1])
        )

    def ports_schema(self):
        return {
            "listeners": {
                "rna_counts": listener_schema(
                    {
                        "mRNA_counts": ([], self.mRNA_TU_ids),
                        "full_mRNA_counts": ([], self.mRNA_TU_ids),
                        "partial_mRNA_counts": ([], self.mRNA_TU_ids),
                        "decaying_mRNA_counts": ([], self.mRNA_TU_ids),
                        "mRNA_cistron_counts": ([], self.mRNA_cistron_ids),
                        "full_mRNA_cistron_counts": ([], self.mRNA_cistron_ids),
                        "partial_mRNA_cistron_counts": ([], self.mRNA_cistron_ids),
                        "available_mRNA_cistron_counts": ([], self.mRNA_cistron_ids),
                        "partial_rRNA_counts": ([], self.rRNA_TU_ids),
                        "partial_rRNA_cistron_counts": ([], self.rRNA_cistron_ids),
                    }
                )
            },
            "RNAs": numpy_schema("RNAs", emit=self.parameters["emit_unique"]),
            "global_time": {"_default": 0.0},
            "timestep": {"_default": self.parameters["time_step"]},
        }

    def update_condition(self, timestep, states):
        return (states["global_time"] % states["timestep"]) == 0

    def next_update(self, timestep, states):
        # Get attributes of mRNAs
        TU_indexes, can_translate, is_full_transcript, transcript_lengths = attrs(
            states["RNAs"],
            ["TU_index", "can_translate", "is_full_transcript", "transcript_length"],
        )
        if self.directional_mrna_decay:
            is_decaying, decay_front = attrs(
                states["RNAs"], ["is_decaying", "decay_front"]
            )
        else:
            is_decaying = np.zeros_like(can_translate)
            decay_front = np.zeros_like(TU_indexes)
        # mRNAs being degraded in the 5' to 3' direction are counted separately
        is_intact_mRNA = can_translate & ~is_decaying
        is_rRNA = np.isin(TU_indexes, self.rRNA_indexes)

        # Get counts of mRNA and rRNA transcription units
        all_TU_counts = np.bincount(
            TU_indexes[np.logical_or(is_intact_mRNA, is_rRNA)],
            minlength=len(self.all_TU_ids),
        )
        mRNA_counts = all_TU_counts[self.mRNA_indexes]
        full_TU_counts = np.bincount(
            TU_indexes[np.logical_and(is_intact_mRNA, is_full_transcript)],
            minlength=len(self.all_TU_ids),
        )
        full_mRNA_counts = full_TU_counts[self.mRNA_indexes]
        partial_TU_counts = all_TU_counts - full_TU_counts
        partial_mRNA_counts = mRNA_counts - full_mRNA_counts
        # Includes decaying mRNAs whose fronts have passed every cistron
        # (can_translate is False) but that are kept until transcription and
        # ribosomes on them finish
        decaying_mRNA_counts = np.bincount(
            TU_indexes[is_decaying],
            minlength=len(self.all_TU_ids),
        )[self.mRNA_indexes]
        # All unique rRNAs are partially transcribed
        partial_rRNA_counts = all_TU_counts[self.rRNA_indexes]

        # Calculate counts of mRNA cistrons from transcription unit counts
        # TODO (ggsun): Partial RNA cistron counts should take into account
        # 	the lengths of each RNA transcript.
        cistron_counts = self.cistron_tu_mapping_matrix.dot(all_TU_counts)
        mRNA_cistron_counts = cistron_counts[self.cistron_is_mRNA]
        full_mRNA_cistron_counts = self.cistron_tu_mapping_matrix.dot(full_TU_counts)[
            self.cistron_is_mRNA
        ]
        partial_mRNA_cistron_counts = self.cistron_tu_mapping_matrix.dot(
            partial_TU_counts
        )[self.cistron_is_mRNA]
        partial_rRNA_cistron_counts = cistron_counts[self.cistron_is_rRNA]

        # Calculate counts of mRNA cistrons that can bind new ribosomes. All
        # cistrons on full transcripts that are not being degraded are
        # available. Cistrons on partial transcripts and on transcripts being
        # degraded are available if they have been transcribed and have not
        # been passed by the degradation front.
        is_other_mRNA = can_translate & ~(is_full_transcript & ~is_decaying)
        other_decay_front = np.where(
            is_decaying[is_other_mRNA], decay_front[is_other_mRNA], -1
        )
        available_mRNA_cistron_counts = (
            full_mRNA_cistron_counts
            + self._count_available_cistrons(
                TU_indexes[is_other_mRNA],
                transcript_lengths[is_other_mRNA],
                other_decay_front,
            )[self.cistron_is_mRNA]
        )

        update = {
            "listeners": {
                "rna_counts": {
                    "mRNA_counts": mRNA_counts,
                    "full_mRNA_counts": full_mRNA_counts,
                    "partial_mRNA_counts": partial_mRNA_counts,
                    "decaying_mRNA_counts": decaying_mRNA_counts,
                    "partial_rRNA_counts": partial_rRNA_counts,
                    "mRNA_cistron_counts": mRNA_cistron_counts,
                    "full_mRNA_cistron_counts": full_mRNA_cistron_counts,
                    "partial_mRNA_cistron_counts": partial_mRNA_cistron_counts,
                    "available_mRNA_cistron_counts": available_mRNA_cistron_counts,
                    "partial_rRNA_cistron_counts": partial_rRNA_cistron_counts,
                }
            }
        }
        return update

    def _count_available_cistrons(self, TU_indexes, transcript_lengths, decay_front):
        """
        Counts the cistrons on a set of mRNAs that have been transcribed and
        have not been passed by the degradation front.

        Args:
            TU_indexes: TU index of each mRNA
            transcript_lengths: Current length of each mRNA
            decay_front: Position of the degradation front of each mRNA (-1
                for mRNAs that are not being degraded)

        Returns:
            Counts of each cistron
        """
        n_cistrons = self.n_cistrons_per_TU[TU_indexes]
        mRNA_rep = np.repeat(np.arange(len(TU_indexes)), n_cistrons)
        pair_indexes = (
            np.repeat(self.TU_pair_offsets[TU_indexes], n_cistrons)
            + np.arange(n_cistrons.sum())
            - np.repeat(np.cumsum(n_cistrons) - n_cistrons, n_cistrons)
        )
        start_positions = self.pair_start_positions[pair_indexes]
        is_available = (transcript_lengths[mRNA_rep] > start_positions) & (
            start_positions >= decay_front[mRNA_rep]
        )
        return np.bincount(
            self.pair_cistron_indexes[pair_indexes][is_available],
            minlength=len(self.all_cistron_ids),
        )


def test_rna_counts_listener():
    from ecoli.experiments.ecoli_master_sim import EcoliSim

    sim = EcoliSim.from_file()
    sim.max_duration = 2
    sim.raw_output = False
    sim.build_ecoli()
    sim.run()
    listeners = sim.query()["agents"]["0"]["listeners"]
    assert isinstance(listeners["rna_counts"]["mRNA_counts"][0], list)
    assert isinstance(listeners["rna_counts"]["mRNA_counts"][1], list)


if __name__ == "__main__":
    test_rna_counts_listener()
