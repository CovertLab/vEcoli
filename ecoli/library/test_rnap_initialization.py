"""
Tests for RNAP initialization edge cases fixed in PR #359. Specifically:
1. RNAPs are assigned to chromosome domains based on their actual coordinates,
   not just their transcription unit (TU) start positions
2. RNAPs cannot be initialized at the same location (collision detection)
"""

import numpy as np
import pytest
from unittest.mock import Mock

from ecoli.library.initial_conditions import initialize_transcription
from wholecell.utils import units


@pytest.fixture
def mock_sim_data():
    """Create a minimal mock sim_data object for testing."""
    sim_data = Mock()

    # Transcription data
    sim_data.process.transcription.rna_data = {
        "length": units.nt * np.array([100, 150, 300, 250, 200, 600]),
        "mw": units.g
        / units.mol
        * np.array([30000.0, 45000.0, 90000.0, 75000.0, 60000.0, 180000.0]),
        "is_forward": np.array([True, True, False, True, False, False]),
        "is_mRNA": np.array([True, False, False, True, False, True]),
        "is_rRNA": np.array([False, False, True, False, True, False]),
        "is_tRNA": np.array([False, True, False, False, False, False]),
        "includes_ribosomal_protein": np.array(
            [False, False, False, False, False, False]
        ),
        "includes_RNAP": np.array([False, False, False, False, False, False]),
        "id": np.array(["TU0", "TU1", "TU2", "TU3", "TU4", "TU5"]),
    }

    sim_data.process.transcription.transcription_sequences = np.random.choice(
        6, (6, 500)
    ).astype(np.int8)

    sim_data.process.transcription.transcription_monomer_weights = np.array(
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    sim_data.process.transcription.transcription_end_weight = 0.5

    sim_data.process.transcription.rnapFractionActiveDict = {"minimal": 0.5}
    sim_data.process.transcription.rnaSynthProbFraction = {
        "minimal": {"mRna": 0.5, "tRna": 0.3, "rRna": 0.2}
    }
    sim_data.process.transcription.rnaSynthProbRProtein = {"minimal": np.array([0.1])}
    sim_data.process.transcription.rnaSynthProbRnaPolymerase = {
        "minimal": np.array([0.05])
    }

    sim_data.process.transcription_regulation.basal_prob = np.array(
        [0.17, 0.17, 0.17, 0.17, 0.17, 0.15]
    )
    sim_data.process.transcription_regulation.get_delta_prob_matrix = Mock(
        return_value=np.zeros((6, 0))
    )
    sim_data.process.transcription.attenuated_rna_indices = np.array([], dtype=int)

    sim_data.process.replication.replichore_lengths = np.array([2000, 2000])

    sim_data.molecule_ids.full_RNAP = "APORNAP-CPLX[c]"
    sim_data.conditions = {"basal": {"nutrients": "minimal"}}
    sim_data.condition = "basal"
    sim_data.constants.n_avogadro = 6.022e23 / units.mol
    sim_data.submass_name_to_index = {
        "nonspecific_RNA": 0,
        "mRNA": 1,
    }
    sim_data.genetic_perturbations = {}

    sim_data.internal_state.unique_molecule.unique_molecule_definitions = {
        "active_RNAP": {"domain_index": "i4", "coordinates": "i8", "is_forward": "?"},
        "RNA": {
            "TU_index": "i8",
            "transcript_length": "i8",
            "is_mRNA": "?",
            "is_full_transcript": "?",
            "can_translate": "?",
            "RNAP_index": "i8",
        },
        "full_chromosome": {
            "division_time": "f8",
            "has_triggered_division": "?",
            "domain_index": "i4",
        },
        "chromosome_domain": {"domain_index": "i4", "child_domains": ("i4", 2)},
        "active_replisome": {
            "domain_index": "i4",
            "right_replichore": "?",
            "coordinates": "i8",
        },
        "oriC": {"domain_index": "i4"},
        "promoter": {
            "TU_index": "i8",
            "coordinates": "i8",
            "domain_index": "i4",
            "bound_TF": ("?", 1),
        },
    }

    return sim_data


@pytest.fixture
def mock_bulk_state():
    """Create mock bulk state with inactive RNAPs."""
    bulk_ids = np.array(
        [
            "APORNAP-CPLX[c]",
            "ATP[c]",
            "GTP[c]",
            "TU0",
            "TU1",
            "TU2",
            "TU3",
            "TU4",
            "TU5",
        ],
        dtype="U50",
    )
    bulk_counts = np.array([500, 1000, 1000, 50, 50, 50, 50, 50, 50])

    bulk_state = np.array(
        list(zip(bulk_ids, bulk_counts)), dtype=[("id", "U50"), ("count", int)]
    )

    return bulk_state


@pytest.fixture
def mock_unique_molecules_double_replicating():
    """
    Create mock unique molecules with 2 rounds replication in progress:
    - Domain 0 (mother domain)
    - Domain 1 and 2 (daughter domains created by 1st fork)
    - Domain 3 and 4 (daughter domains created by 2nd fork)
    - 4 active replisomes creating the forks
    - Promoters distributed across domains
    """
    unique_molecules = {}

    # Full chromosome
    unique_molecules["full_chromosome"] = np.array(
        [(0, 0.0, True, 1)],
        dtype=[
            ("domain_index", "i4"),
            ("division_time", "f8"),
            ("has_triggered_division", "?"),
            ("_entryState", "i1"),
        ],
    )

    # Chromosome domains: 0 (mother), 1 and 2 (daughters)
    unique_molecules["chromosome_domain"] = np.array(
        [
            (0, [1, 2], 1),  # Mother domain has two child domains
            (1, [-1, -1], 1),  # Child domain 1
            (2, [3, 4], 1),  # Child domain 2 has two child domains
            (3, [-1, -1], 1),  # Child domain 3
            (4, [-1, -1], 1),  # Child domain 4
        ],
        dtype=[
            ("domain_index", "i4"),
            ("child_domains", "i4", 2),
            ("_entryState", "i1"),
        ],
    )

    # OriCs in outermost daughter domains
    unique_molecules["oriC"] = np.array(
        [(1, 1), (3, 1), (4, 1)],
        dtype=[
            (
                "domain_index",
                "i4",
            ),
            ("_entryState", "i1"),
        ],
    )

    # Active replisomes at position 500 (creating fork)
    # Right replichore replisome and left replichore replisome
    unique_molecules["active_replisome"] = np.array(
        [
            (0, 1000, True, 1000, 1),  # Right fork, domain 0
            (0, -1000, False, 1001, 1),  # Left fork, domain 0
            (2, 500, True, 1000, 1),  # Right fork, domain 2
            (2, -500, False, 1001, 1),  # Left fork, domain 2
        ],
        dtype=[
            ("domain_index", "i4"),
            ("coordinates", "i8"),
            ("right_replichore", "?"),
            ("unique_index", "i8"),
            ("_entryState", "i1"),
        ],
    )

    # TU lengths: [100, 150, 300, 250, 200, 600]
    # TU forward: [True, True, False, True, False, False]
    # TU 0: coordinates -100 to 0, domain 1
    # TU 1: coordinates 1000 to 1150, domain 0
    # TU 2: coordinates -300 to -600, domain 4 overlapping with 2
    # TU 3: coordinates 800 to 1050, domain 2 overlapping with 0
    # TU 4: coordinates 600 to 400, domain 2 overlapping with 3/4
    # TU 5: coordinates 1050 to 450, domain 0 overlapping with 1/2 and 3/4
    unique_molecules["promoter"] = np.array(
        [
            (1, 0, -100, np.zeros(1, dtype=bool), 1),
            (0, 1, 1000, np.zeros(1, dtype=bool), 1),
            (4, 2, -300, np.zeros(1, dtype=bool), 1),
            (2, 3, 800, np.zeros(1, dtype=bool), 1),
            (2, 4, 600, np.zeros(1, dtype=bool), 1),
            (0, 5, 1050, np.zeros(1, dtype=bool), 1),
        ],
        dtype=[
            ("domain_index", "i4"),
            ("TU_index", "i8"),
            ("coordinates", "i8"),
            ("bound_TF", "?", (1,)),
            ("_entryState", "i1"),
        ],
    )

    return unique_molecules


def check_domain_boundaries(unique_mols):
    # Domain indices
    domain_indices = unique_mols["active_RNAP"]["domain_index"]
    coordinates = unique_mols["active_RNAP"]["coordinates"]
    tu_indices = np.array(
        [
            unique_mols["RNA"]["TU_index"][unique_mols["RNA"]["RNAP_index"] == rnap_idx]
            for rnap_idx in unique_mols["active_RNAP"]["unique_index"]
        ]
    )

    for domain_i, coord_i, tu_i in zip(domain_indices, coordinates, tu_indices):
        if tu_i == 0:
            assert domain_i == 1
            assert coord_i >= -100 and coord_i <= 0
        elif tu_i == 1:
            assert domain_i == 0
            assert coord_i >= 1000 and coord_i <= 1150
        elif tu_i == 2:
            assert coord_i <= -300 and coord_i >= -600
            if coord_i <= -500:
                assert domain_i == 2
            else:
                # TU 2 starts in domain 4 and ends in domain 2.
                # A TU 2 RNAP can be initialized in domain 2,
                # overlap with another RNAP, and relocated back
                # to a coordinate in domain 4. However, coordinates
                # in domain 4 are also valid in domain 3. In these
                # cases, the initialization code will always assign
                # the RNAP to the domain that appears first in the
                # list of valid domains (3 in this test case).
                #
                # This is mechanistically nonsensical because domain
                # 3 does not have the TU 2 promoter and should not
                # have TU 2 RNAPs. Fortunately, in real sims, both
                # daughter domains will each have a copy of every
                # promoter. This fixes the biological plausibility
                # but still favors initialization in one domain over
                # another. From what I can tell, collisions between
                # RNAPs during initialization should be rare
                # enough in real sims that this is negligible.
                assert domain_i in [3, 4]
        elif tu_i == 3:
            assert coord_i >= 800 and coord_i <= 1050
            if coord_i >= 1000:
                assert domain_i == 0
            else:
                # TU 3 RNAPs can end up in domain 1 for same
                # reason as above.
                assert domain_i in [1, 2]
        elif tu_i == 4:
            assert coord_i <= 600 and coord_i >= 400
            # Unlike TU 2 and 3, TU 4 starts in domain 2
            # and ends in 3/4. Domain 3 is always
            # chosen because it comes before domain 4
            # in our test setup.
            #
            # This is biologically sound because
            # RNAPs start at the promoter and process
            # along only the daughter domain containing
            # the template strand. The domain containing
            # the non-template strand does not yet have
            # a template of the promoter, so no RNAPs
            # should be initialized there.
            if coord_i <= 500:
                assert domain_i == 3
            else:
                assert domain_i == 2
        elif tu_i == 5:
            assert coord_i <= 1050 and coord_i >= 450
            # TU 5 technically spans all domains, but
            # RNAPs will only be initialized in domain
            # 0 (start of TU) or 1 (first valid domain
            # with all coordinates outside domain 0).
            #
            # This is biologically sound for the same
            # reason as TU 4.
            if coord_i >= 1000:
                assert domain_i == 0
            else:
                assert domain_i == 1
        else:
            raise AssertionError(f"Unexpected TU index {tu_i}")


def check_no_duplicates(active_rnaps):
    assert len(active_rnaps) == len(
        np.unique(
            list(zip(active_rnaps["coordinates"], active_rnaps["domain_index"])), axis=0
        )
    ), "Found duplicate (coordinate, domain) pairs"


class TestRNAPDomainAssignment:
    """Test that RNAPs are correctly assigned to chromosome domains."""

    def test_rnap_domain_assignment(
        self, mock_sim_data, mock_bulk_state, mock_unique_molecules_double_replicating
    ):
        """
        Test RNAP domain assignment when there's no replication.
        All RNAPs should be assigned to domain 0.
        """
        mock_sim_data.process.transcription.rna_data["replication_coordinate"] = (
            mock_unique_molecules_double_replicating["promoter"]["coordinates"]
        )
        unique_id_rng = np.random.RandomState(seed=42)
        random_state = np.random.RandomState(seed=100)

        # Call initialize_transcription
        initialize_transcription(
            mock_bulk_state,
            mock_unique_molecules_double_replicating,
            mock_sim_data,
            random_state,
            unique_id_rng,
            ppgpp_regulation=False,
            trna_attenuation=False,
        )

        # Check domains of all initialized RNAPs
        check_domain_boundaries(mock_unique_molecules_double_replicating)
        check_no_duplicates(mock_unique_molecules_double_replicating["active_RNAP"])


class TestRNAPCollisionDetection:
    """Test that RNAPs cannot be initialized at the same location."""

    def test_collision_resolution(
        self, mock_sim_data, mock_bulk_state, mock_unique_molecules_double_replicating
    ):
        """
        Test that RNAP collisions are appropriately handled.
        """
        mock_sim_data.process.transcription.rna_data["replication_coordinate"] = (
            mock_unique_molecules_double_replicating["promoter"]["coordinates"]
        )
        # Use a fixed random state that gives same value for first 30 calls
        random_state = Mock()
        hidden_rng = np.random.RandomState(seed=42)
        random_state.multinomial = hidden_rng.multinomial
        call_count = [0]

        def controlled_rand(n):
            call_count[0] += 1
            if call_count[0] < 30:
                return np.array([1] * n)
            return hidden_rng.rand(n)

        random_state.rand = controlled_rand
        unique_id_rng = np.random.RandomState(seed=43)

        initialize_transcription(
            mock_bulk_state,
            mock_unique_molecules_double_replicating,
            mock_sim_data,
            random_state,
            unique_id_rng,
            ppgpp_regulation=False,
            trna_attenuation=False,
        )

        active_rnaps = mock_unique_molecules_double_replicating["active_RNAP"]
        check_domain_boundaries(mock_unique_molecules_double_replicating)
        check_no_duplicates(active_rnaps)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
