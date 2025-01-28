"""
================
RNA Interference
================
Treats sRNA-mRNA binding as complexation events that create duplexes and free
bound ribosomes. Decreases ompF translation during micF overexpression.
"""

import numpy as np
import warnings

from vivarium.core.process import Process
from vivarium.core.engine import Engine

from ecoli.library.schema import numpy_schema, bulk_name_to_idx, counts, attrs
from ecoli.processes.registries import topology_registry
from ecoli.processes.unique_update import UniqueUpdate

# Register default topology for this process, associating it with process name
NAME = "ecoli-rna-interference"
TOPOLOGY = {
    "bulk": ("bulk",),
    "RNAs": ("unique", "RNA"),
    "active_ribosome": ("unique", "active_ribosome"),
}
topology_registry.register(NAME, TOPOLOGY)


class RnaInterference(Process):
    name = NAME
    topology = TOPOLOGY
    defaults = {
        "srna_ids": [],
        "target_tu_ids": [],
        "binding_probs": [],
        "ribosome30S": "ribosome30S",
        "ribosome50S": "ribosome50S",
        "duplex_ids": [],
        "seed": 0,
        "time_step": 2,
        "emit_unique": False,
    }

    def __init__(self, parameters=None):
        super().__init__(parameters)
        # Parameters are lists such that the nth element
        # of each list are grouped (e.g. the 1st sRNA ID
        # binds to the first sRNA target with the 1st
        # binding probability)
        self.srna_ids = self.parameters["srna_ids"]
        self.target_tu_ids = self.parameters["target_tu_ids"]
        self.binding_probs = self.parameters["binding_probs"]
        self.ribosome30S = self.parameters["ribosome30S"]
        self.ribosome50S = self.parameters["ribosome50S"]
        self.duplex_ids = list(self.parameters["duplex_ids"])
        self.bulk_rna_ids = self.srna_ids + self.duplex_ids
        self.random_state = np.random.RandomState(seed=self.parameters["seed"])

        self.srna_idx = None

    def ports_schema(self):
        return {
            "bulk": numpy_schema("bulk"),
            "active_ribosome": numpy_schema(
                "active_ribosome", emit=self.parameters["emit_unique"]
            ),
            "RNAs": numpy_schema("RNAs", emit=self.parameters["emit_unique"]),
        }

    def next_update(self, timestep, states):
        if self.srna_idx is None:
            bulk_ids = states["bulk"]["id"]
            self.srna_idx = bulk_name_to_idx(self.srna_ids, bulk_ids)
            self.subunit_idx = bulk_name_to_idx(
                [self.ribosome30S, self.ribosome50S], bulk_ids
            )
            self.duplex_idx = bulk_name_to_idx(self.duplex_ids, bulk_ids)

        update = {
            "bulk": [],
            "RNAs": {"delete": []},
            "active_ribosome": {"delete": []},
        }

        TU_index, can_translate, is_full_transcript, rna_indexes = attrs(
            states["RNAs"],
            ["TU_index", "can_translate", "is_full_transcript", "unique_index"],
        )

        mRNA_index, ribosome_indexes = attrs(
            states["active_ribosome"], ["mRNA_index", "unique_index"]
        )

        for srna_idx, mrna_index, binding_prob, duplex_idx in zip(
            self.srna_idx, self.target_tu_ids, self.binding_probs, self.duplex_idx
        ):
            # Get count of complete sRNAs
            srna_count = counts(states["bulk"], srna_idx)
            if srna_count == 0:
                continue

            # Get mask for translatable, complete target mRNAs
            mrna_mask = np.logical_and(TU_index == mrna_index, can_translate)
            mrna_mask = np.logical_and(mrna_mask, is_full_transcript)
            n_mrna = mrna_mask.sum()
            if n_mrna == 0:
                continue

            # Each sRNA has probability binding_prob of binding a target mRNA
            n_duplexed = np.min(
                [self.random_state.binomial(srna_count, binding_prob), mrna_mask.sum()]
            )

            # Choose n_duplexed mRNAs and sRNAs randomly to delete
            mrna_to_delete = self.random_state.choice(
                size=n_duplexed, a=np.nonzero(mrna_mask)[0], replace=False
            ).tolist()
            update["RNAs"]["delete"] += list(mrna_to_delete)
            update["bulk"].append((srna_idx, -n_duplexed))

            # Dissociate ribosomes attached to new duplexes
            ribosomes_to_delete = list(
                np.nonzero(np.isin(mRNA_index, rna_indexes[mrna_to_delete]))[0]
            )
            update["active_ribosome"]["delete"] += ribosomes_to_delete
            update["bulk"].append((self.subunit_idx, len(ribosomes_to_delete)))

            # Ensure that additional sRNAs cannot bind to mRNAs that have
            # already been duplexed
            remainder_mask = np.ones(TU_index.size, dtype=bool)
            remainder_mask[mrna_to_delete] = False
            TU_index = TU_index[remainder_mask]
            can_translate = can_translate[remainder_mask]
            is_full_transcript = is_full_transcript[remainder_mask]
            rna_indexes = rna_indexes[remainder_mask]

            # Add new RNA duplexes
            update["bulk"].append((duplex_idx, n_duplexed))

        return update


def test_rna_interference(return_data=False):
    test_config = {
        "time_step": 2,
        "ribosome30S": "CPLX0-3953[c]",
        "ribosome50S": "CPLX0-3962[c]",
        "srna_ids": ["MICF-RNA[c]"],
        "target_tu_ids": [661],
        "target_ids": ["EG10671_RNA[c]"],
        "duplex_ids": ["micF-ompF[c]"],
        "binding_probs": [0.5],
    }

    rna_inter = RnaInterference(test_config)
    unique_topology = {
        "active_ribosome": (
            "unique",
            "active_ribosome",
        ),
        "RNAs": (
            "unique",
            "RNA",
        ),
    }
    unique_update = UniqueUpdate({"unique_topo": unique_topology})

    initial_state = {
        "bulk": np.array(
            [
                ("CPLX0-3953[c]", 100),
                ("CPLX0-3962[c]", 100),
                ("MICF-RNA[c]", 4),
                ("micF-ompF[c]", 0),
            ],
            dtype=[("id", "U40"), ("count", int)],
        ),
        "unique": {
            "active_ribosome": np.array(
                [
                    (1, 1, 1),
                    (1, 2, 2),
                    (1, 1, 3),
                ],
                dtype=[
                    ("_entryState", np.int8),
                    ("mRNA_index", int),
                    ("unique_index", int),
                ],
            ),
            "RNA": np.array(
                [
                    (1, 661, True, True, 1089, 1),
                    (1, 661, True, True, 1089, 2),
                    (1, 661, True, True, 1089, 3),
                    (1, 661, True, True, 1089, 4),
                ],
                dtype=[
                    ("_entryState", np.int8),
                    ("TU_index", int),
                    ("can_translate", "?"),
                    ("is_full_transcript", "?"),
                    ("transcript_length", int),
                    ("unique_index", int),
                ],
            ),
        },
    }

    # Since unique numpy updater is an class method, internal
    # deepcopying in vivarium-core causes this warning to appear
    warnings.filterwarnings(
        "ignore",
        message="Incompatible schema "
        "assignment at .+ Trying to assign the value <bound method "
        r"UniqueNumpyUpdater\.updater .+ to key updater, which already "
        r"has the value <bound method UniqueNumpyUpdater\.updater",
    )
    experiment = Engine(
        processes={"rna-interference": rna_inter},
        steps={"unique-update": unique_update},
        topology={
            "rna-interference": rna_inter.topology,
            "unique-update": unique_topology,
        },
        initial_state=initial_state,
    )
    experiment.state.set_emit_values(
        [
            ("bulk",),
            (
                "unique",
                "active_ribosome",
            ),
            (
                "unique",
                "RNA",
            ),
        ],
        True,
    )
    experiment.update(4)
    data = experiment.emitter.get_data()

    if return_data:
        return data, test_config


def main():
    data, config = test_rna_interference(return_data=True)
    print(data)


if __name__ == "__main__":
    main()
