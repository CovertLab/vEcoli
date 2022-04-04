import os
from collections.abc import MutableMapping
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np


class HoleSizeDict(MutableMapping):
    def __init__(self, data={}):
        self.mapping = {}
        self.max = 0
        self.update(data)
        if len(data) > 0:
            self.max = max(v for v in data.values() if isinstance(v, int))

    def __getitem__(self, key: frozenset):
        result = self.mapping[key]
        while not isinstance(result, int):
            result = self.mapping[result]
        return result

    def __delitem__(self, key):
        del self.mapping[key]

    def __setitem__(self, key: frozenset, value: int):
        loc = key
        while not isinstance(self.mapping.get(loc, 0), int):
            loc = self.mapping[loc]

        self.mapping[loc] = value
        if value > self.max:
            self.max = value

    def merge(self, hole1: frozenset, hole2: frozenset):
        while not isinstance(self.mapping[hole1], int):
            hole1 = self.mapping[hole1]

        while not isinstance(self.mapping[hole2], int):
            hole2 = self.mapping[hole2]

        # Only merge if hole1 and hole2 do not already
        # map to the same location
        merged_hole = hole1 | hole2
        if hole1 != hole2:
            new_size = self.mapping[hole1] + self.mapping[hole2]
            self.mapping[merged_hole] = new_size
            self.mapping[hole1] = merged_hole
            self.mapping[hole2] = merged_hole

            if new_size > self.max:
                self.max = new_size

        return merged_hole

    def get_max(self):
        return self.max

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"


def detect_holes(lattice):
    # Create "hole view" of lattice.
    # Each position contains a set of integers representing the id of the hole
    # containing that position (or an empty set if that position is not a hole).
    #
    # Ids are sets because two holes that are initially considered separate may
    # later be discovered to be the same hole.

    # Fill hole view initially with empty sets
    hole_view = np.full_like(lattice, frozenset(), dtype=object)
    hole_sizes = HoleSizeDict()

    next_hole_id = 1
    rows, cols = lattice.shape
    for r in range(rows):
        for c in range(cols):
            # Skip non-holes
            if lattice[r, c] == 1:
                continue

            # Get neighbors of the current hole.
            # The following diagram indicates which positions
            # relative to the current position (X)
            # are considered neighbors (N):
            #
            # N N N
            # N X
            neighbor_pos = {
                (max(0, min(n_r, rows - 1)), max(0, min(n_c, cols - 1)))
                for n_r in range(r - 1, r + 1)
                for n_c in range(c - 1, c + 2)
            } - {(r, c), (r, c + 1)}
            neighbor_holes = [
                hole_view[n_r, n_c]
                for n_r, n_c in neighbor_pos
                if len(hole_view[n_r, n_c]) > 0
            ]

            if len(neighbor_holes) == 0:

                # Creating new hole
                new_id = frozenset([next_hole_id])
                hole_sizes[new_id] = 1
                next_hole_id += 1
            else:
                new_id = neighbor_holes[0]
                for id1, id2 in zip(neighbor_holes, neighbor_holes[1:]):
                    new_id = hole_sizes.merge(id1, id2)
                hole_sizes[new_id] += 1

            hole_view[r, c] = new_id

    return hole_view, hole_sizes.get_max()


def test_hole_size_dict():
    hsd = HoleSizeDict({frozenset([1]): 1, frozenset([2]): 2})

    hsd.merge(frozenset([1]), frozenset([2]))
    assert hsd[frozenset([1, 2])] == 3

    hsd[frozenset([1])] += 5
    assert hsd[frozenset([1, 2])] == 8


def test_detect_holes():
    # Create output directory
    os.makedirs("out/hole_detection", exist_ok=True)

    # Get tests
    test_files = os.listdir("ecoli/library/cell_wall/test_cases")

    # Run tests
    n_passed = 0
    for test_case in test_files:
        print(f"Test case: {test_case}")

        # Load test case
        test_array = np.genfromtxt(
            f"ecoli/library/cell_wall/test_cases/{test_case}", dtype=int, skip_header=1
        )
        expected_max_size = np.loadtxt(
            f"ecoli/library/cell_wall/test_cases/{test_case}", dtype=int, max_rows=1
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


def test_runtime():
    # Runtime plot
    fig, ax = plt.subplots()

    rng = np.random.default_rng(0)
    side_length = [10, 100, 200]
    density = np.arange(0, 1, 0.1)

    for d in density:
        runtimes = []
        for s in side_length:
            a = rng.binomial(1, d, size=s * s).reshape((s, s))

            tick = perf_counter()
            detect_holes(a)
            tock = perf_counter()

            runtimes.append(tock - tick)

            print(f"Runtime for side length {s}, density {d:.1f} : {tock-tick} seconds")

        ax.plot(
            side_length,
            runtimes,
            label=f"density={d:.1f}",
            color=((1 - d) * 0.75, 0, d),
        )

    ax.legend()
    ax.set_xlabel("Side length")
    ax.set_ylabel("Runtime (s)")
    fig.tight_layout()
    fig.savefig("out/hole_detection/test_runtime.png")


def main():
    test_hole_size_dict()
    test_detect_holes()
    test_runtime()


if __name__ == "__main__":
    main()
