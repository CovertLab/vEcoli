"""
profile_polymerize.py

Profiles where the time goes in polymerize().

Instructions: From the wcEcoli directory, run
        kernprof -lv wholecell/tests/utils/profile_polymerize.py
to get a line-by-line profile of functions decorated @profile.

This is split out into a separate file and directory from polymerize.py because
Python puts the current script's directory on the sys.path, and that breaks for
polymerize.py since its directory contains random.py. Importing numpy tries to
import the system random which instead loads the local random.py, and that
fails. So keep polymerize.py's directory off the sys.path.

The normal solution is to cd to wcEcoli then run "python -m <module>" but
kernprof doesn't support that.
"""

import cProfile
from io import StringIO
import pstats
from time import monotonic as monotonic_seconds

import numpy as np

from wholecell.utils.polymerize import polymerize
from wholecell.utils.profiler import line_profile

PAD_VALUE = polymerize.PAD_VALUE

# Attach __iter__ method to preserve old interface
# TODO (John): migrate to new interface
polymerize.__iter__ = lambda self: iter(
    (self.sequenceElongation, self.monomerUsages, self.nReactions)
)


# Wrap methods in line-profiling decorator
# noinspection PyUnresolvedReferences
def setup_profiler():
    polymerize.__init__ = line_profile(polymerize.__init__)

    polymerize._setup = line_profile(polymerize._setup)
    polymerize._sanitize_inputs = line_profile(polymerize._sanitize_inputs)
    polymerize._gather_input_dimensions = line_profile(
        polymerize._gather_input_dimensions
    )
    polymerize._gather_sequence_data = line_profile(polymerize._gather_sequence_data)
    polymerize._prepare_running_values = line_profile(
        polymerize._prepare_running_values
    )
    polymerize._prepare_outputs = line_profile(polymerize._prepare_outputs)

    polymerize._elongate = line_profile(polymerize._elongate)
    polymerize._elongate_to_limit = line_profile(polymerize._elongate_to_limit)
    polymerize._finalize_resource_limited_elongations = line_profile(
        polymerize._finalize_resource_limited_elongations
    )
    polymerize._update_elongation_resource_demands = line_profile(
        polymerize._update_elongation_resource_demands
    )

    polymerize._finalize = line_profile(polymerize._finalize)
    polymerize._clamp_elongation_to_sequence_length = line_profile(
        polymerize._clamp_elongation_to_sequence_length
    )


def _setupRealExample():
    # Test data pulled from an actual sim at an early time point.
    monomerLimits = np.array(
        [
            11311,
            6117,
            4859,
            6496,
            843,
            7460,
            4431,
            8986,
            2126,
            6385,
            9491,
            7254,
            2858,
            3770,
            4171,
            5816,
            6435,
            1064,
            3127,
            0,
            8749,
        ]
    )

    randomState = np.random.RandomState()

    nMonomers = len(monomerLimits)  # number of distinct aa-tRNAs
    nSequences = 10000  # approximate number of ribosomes
    length = 16  # translation rate
    nTerminating = np.int64(
        length / 300 * nSequences
    )  # estimate for number of ribosomes terminating

    sequences = np.random.randint(nMonomers, size=(nSequences, length))

    sequenceLengths = length * np.ones(nSequences, np.int64)
    sequenceLengths[np.random.choice(nSequences, nTerminating, replace=False)] = (
        np.random.randint(length, size=nTerminating)
    )

    sequences[np.arange(length) > sequenceLengths[:, np.newaxis]] = PAD_VALUE

    reactionLimit = 10000000

    return sequences, monomerLimits, reactionLimit, randomState


def _setupExample():
    # Contrive a scenario which is similar to real conditions

    randomState = np.random.RandomState()

    nMonomers = 36  # number of distinct aa-tRNAs
    nSequences = 10000  # approximate number of ribosomes
    length = 16  # translation rate
    nTerminating = np.int64(
        length / 300 * nSequences
    )  # estimate for number of ribosomes terminating
    monomerSufficiency = 0.85
    energySufficiency = 0.85

    sequences = np.random.randint(nMonomers, size=(nSequences, length))

    sequenceLengths = length * np.ones(nSequences, np.int64)
    sequenceLengths[np.random.choice(nSequences, nTerminating, replace=False)] = (
        np.random.randint(length, size=nTerminating)
    )

    sequences[np.arange(length) > sequenceLengths[:, np.newaxis]] = PAD_VALUE

    maxReactions = sequenceLengths.sum()

    monomerLimits = (
        monomerSufficiency * maxReactions / nMonomers * np.ones(nMonomers, np.int64)
    )
    reactionLimit = energySufficiency * maxReactions

    return sequences, monomerLimits, reactionLimit, randomState


def _simpleProfile():
    np.random.seed(0)

    sequences, monomerLimits, reactionLimit, randomState = _setupExample()

    nSequences, length = sequences.shape
    nMonomers = monomerLimits.size
    sequenceLengths = (sequences != PAD_VALUE).sum(axis=1)
    elongation_rates = []  # TODO: What to use here?

    t = monotonic_seconds()
    sequenceElongation, monomerUsages, nReactions = polymerize(
        sequences, monomerLimits, reactionLimit, randomState, elongation_rates
    )
    eval_sec = monotonic_seconds() - t

    assert (sequenceElongation <= sequenceLengths + 1).all()
    assert (monomerUsages <= monomerLimits).all()
    assert nReactions <= reactionLimit
    assert nReactions == monomerUsages.sum()

    print(
        """
Polymerize function report:

For {} sequences of {} different monomers elongating by at most {}:

{:0.1f} ms to evaluate
{} polymerization reactions
{:0.1f} average elongations per sequence
{:0.1%} monomer utilization
{:0.1%} energy utilization
{:0.1%} fully elongated
{:0.1%} completion
""".format(
            nSequences,
            nMonomers,
            length,
            eval_sec * 1000,
            nReactions,
            sequenceElongation.mean(),
            monomerUsages.sum() / monomerLimits.sum(),
            nReactions / reactionLimit,
            (sequenceElongation == sequenceLengths).sum() / nSequences,
            sequenceElongation.sum() / sequenceLengths.sum(),
        )
    )


def _fullProfile():
    np.random.seed(0)

    sequences, monomerLimits, reactionLimit, randomState = _setupRealExample()

    # Recipe from https://docs.python.org/2/library/profile.html#module-cProfile
    pr = cProfile.Profile()
    pr.enable()

    elongation_rates = []  # TODO: What to use here?
    sequenceElongation, monomerUsages, nReactions = polymerize(
        sequences, monomerLimits, reactionLimit, randomState, elongation_rates
    )

    pr.disable()
    s = StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


if __name__ == "__main__":
    setup_profiler()
    _fullProfile()
