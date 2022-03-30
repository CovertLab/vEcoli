import os
from functools import reduce
from operator import __or__
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


def detect_holes(lattice):
    # Create "hole view" of lattice.
    # Each position contains a set of integers representing the id of the hole
    # containing that position (or an empty set if that position is not a hole).
    #
    # Ids are sets because two holes that are initially considered separate may
    # later be discovered to be the same hole.

    # Fill hole view initially with empty sets
    hole_view = np.full_like(lattice, frozenset(), dtype=object)
    hole_sizes = dict()

    next_hole_id = 1
    rows, cols = lattice.shape
    for r in range(rows):
        for c in range(cols):
            # Skip non-holes
            if lattice[r, c] == 1:
                continue

            # Get neighbors of the current hole.
            # The following diagram indicates which positions relative to the current position (X)
            # are considered neighbors (N):
            #
            # N N N
            # N X
            neighbor_pos = {
                (max(0, min(n_r, rows - 1)), max(0, min(n_c, cols - 1)))
                for n_r in range(r - 1, r + 1)
                for n_c in range(c - 1, c + 2)
            } - {(r, c), (r, c + 1)}
            neighbor_holes = {
                hole_view[n_r, n_c]
                for n_r, n_c in neighbor_pos
                if len(hole_view[n_r, n_c]) > 0
            }

            if len(neighbor_holes) == 0:

                # Creating new hole
                new_id = frozenset([next_hole_id])
                hole_sizes[new_id] = 1
                next_hole_id += 1
            else:

                # Fill current position with correct id, by merging with existing holes
                new_id = reduce(__or__, neighbor_holes)

                # If merging holes, need to re-route hole size information
                # (e.g. if combining {1}, {2} => {1, 2}, need to update hole_sizes
                # entries such that {1}, {2} say to go look at {1, 2}).
                if new_id not in neighbor_holes:

                    # Route everything to destination_id, initially new_id (may change later)
                    destination_id = new_id
                    hole_sizes[destination_id] = hole_sizes.get(destination_id, 0)

                    for id in neighbor_holes:

                        loc = hole_sizes[id]
                        while not isinstance(loc, int):
                            # May have found a new destination.
                            # Traverse until an integer is found,
                            # updating the destination along the way.
                            new_dest_id = destination_id | loc

                            # reroute current destination to new destination
                            size = hole_sizes[destination_id]
                            hole_sizes[destination_id] = new_dest_id
                            hole_sizes[new_dest_id] = size
                            destination_id = new_dest_id

                            # reroute current location to destination
                            hole_sizes[loc] = destination_id

                            # Traverse upwards
                            loc = hole_sizes[loc]

                        # Transfer size information to destination id,
                        # route to destination id. Note we are iterating over
                        # neighbor_holes, which is a set, so this will not
                        # result in double-counting.
                        hole_sizes[destination_id] += loc
                        hole_sizes[
                            id
                        ] = destination_id  # Shortcut - can we just use this routing
                        # without doing the in-traversal routing above?

                    hole_sizes[destination_id] += 1

                else:
                    loc = new_id
                    while not isinstance(hole_sizes[loc], int):
                        loc = hole_sizes[loc]
                    hole_sizes[loc] += 1

            hole_view[r, c] = new_id

    return hole_view, max(filter(lambda x: isinstance(x, int), hole_sizes.values()))


def main():
    # Create output directory
    os.makedirs("out/hole_detection", exist_ok=True)

    # Get tests
    test_files = os.listdir("user/hole_detection/test_cases")

    # Run tests
    n_passed = 0
    for test_case in test_files:
        print(f"Test case: {test_case}")

        # Load test case
        test_array = np.genfromtxt(
            f"user/hole_detection/test_cases/{test_case}", dtype=int, skip_header=1
        )
        expected_max_size = np.loadtxt(
            f"user/hole_detection/test_cases/{test_case}", dtype=int, max_rows=1
        )

        # Get hole view, size of largest hole
        hole_view, max_hole = detect_holes(test_array)

        # Prints and asserts
        print(f"> Size of largest hole: {max_hole} (Expected {expected_max_size})")

        passed = max_hole == expected_max_size
        n_passed += int(passed)
        print(f"> {'PASSED' if passed else 'FAILED'}")
        print()

        # Plot test case, hole view
        fig, ax = plt.subplots()
        ax.imshow(test_array, interpolation="nearest", aspect="auto")
        for r in range(hole_view.shape[0]):
            for c in range(hole_view.shape[1]):
                ax.text(
                    c,
                    r,
                    f"{set(hole_view[r, c]) if len(hole_view[r,c]) > 0 else ''}",
                    ha="center",
                    va="center",
                    color="w",
                )
        ax.set_title(f"Hole View (Max hole detected = {max_hole})")

        # Save image
        fig.tight_layout()
        fig.savefig(f"out/hole_detection/test_{test_case}.png")

    print("===============================================")
    print(f"Passed {n_passed}/{len(test_files)} tests.")
    print()

    # Runtime plot
    rng = np.random.default_rng(0)
    side_length = [10, 100, 500]
    density = np.arange(0, 1, 0.1)
    runtimes = np.zeros((len(side_length), len(density)))

    for r, s in enumerate(side_length):
        for c, d in enumerate(density):
            a = rng.binomial(1, d, size=s * s).reshape((s, s))
            tick = perf_counter()
            detect_holes(a)
            tock = perf_counter()

            runtimes[r, c] = tock - tick
            print(
                f"Runtime for side length {s}, density {d:.1f} : {runtimes[r, c]} seconds"
            )

    fig, ax = plt.subplot()
    ax.plot(runtimes)
    fig.tight_layout()
    fig.savefig("out/hole_detection/test_runtime.png")


if __name__ == "__main__":
    main()
