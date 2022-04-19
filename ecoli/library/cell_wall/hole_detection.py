from cProfile import label
import os
from sys import getsizeof
from collections.abc import MutableMapping
from functools import reduce
from operator import __or__
from time import perf_counter
from attr import frozen

import matplotlib.pyplot as plt
import numpy as np


class HoleSizeDict(MutableMapping):
    def __init__(self, data={}):
        self.mapping = {}
        self.roots = set()
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
        if key in self.roots:
            self.prune_subtree(key)
            return

        # Need to remap anything that maps to this key
        destination = self.get_containing_hole(key)
        to_remap = set()
        for k, v in self.items():
            if v == key:
                to_remap.add(k)

        for k in to_remap:
            self.mapping[k] = destination

        del self.mapping[key]

    def __setitem__(self, key: frozenset, value: int):
        loc = key
        while not isinstance(self.mapping.get(loc, 0), int):
            loc = self.mapping[loc]

        # Store value, update maximum if necessary
        self.mapping[loc] = value
        self.roots.add(loc)
        if value > self.max:
            self.max = value

    def merge(self, holes):
        containing_holes = {self.get_containing_hole(hole) for hole in holes}

        # Only merge if some entries do not already map to same location
        merged_hole = reduce(__or__, containing_holes)

        if len(containing_holes) > 1:
            new_size = sum(self.mapping[hole] for hole in containing_holes)
            self.mapping[merged_hole] = new_size
            for hole in containing_holes:
                self.mapping[hole] = merged_hole

                # No longer a root node after merge
                self.roots.remove(hole)

            if new_size > self.max:
                self.max = new_size

            self.roots.add(merged_hole)

        return merged_hole

    def prune_subtree(self, root):
        if not isinstance(self.mapping[root], int):
            raise ValueError(f"Node {root} is not a root node of a subtree.")

        for leaf in root:
            leaf = frozenset([leaf])

            current = leaf
            while current != root and current in self.mapping:
                next_node = self.mapping[current]
                del self.mapping[current]
                current = next_node

        del self.mapping[root]
        self.roots.remove(root)

    def get_containing_hole(self, hole):
        containing_hole = hole
        while not isinstance(self.mapping[containing_hole], int):
            containing_hole = self.mapping[containing_hole]

        return containing_hole

    def get_max(self):
        return self.max

    def get_depth(self):
        # Returns the length of the longest branch in the tree.
        # Somewhat naive implementation, but then again, shouldn't need this often.
        depth = 1
        seen = set()
        for key in self.mapping:
            if key in seen:
                continue
            seen.add(key)

            key_depth = 1
            while not isinstance(self.mapping[key], int):
                key = self.mapping[key]
                key_depth += 1

            if key_depth > depth:
                depth = key_depth

        return depth

    def __iter__(self):
        return iter(self.mapping)

    def __len__(self):
        return len(self.mapping)

    def __repr__(self):
        return f"{type(self).__name__}({self.mapping})"


def detect_holes(lattice, critical_size=None):
    # Create "hole view" of lattice.
    # Each position contains a set of integers representing the id of the hole
    # containing that position (or an empty set if that position is not a hole).
    #
    # Ids are sets because two holes that are initially considered separate may
    # later be discovered to be the same hole.

    # Fill hole view initially with empty sets
    hole_view = np.full_like(lattice, frozenset(), dtype=object)
    hole_sizes = HoleSizeDict()

    # Subtree pruning to save memory will occur when
    # len(hole_sizes) exceeds this value
    next_prune_size = 100

    next_hole_id = 1
    rows, cols = lattice.shape
    for r in range(rows):
        ids_in_row = set()
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
            neighbor_pos = {(r - 1, c - 1), (r - 1, c), (r - 1, c + 1), (r, c - 1)}
            # list({...}) removes duplicates
            neighbor_holes = list(
                {
                    hole_view[n_r, n_c]
                    for n_r, n_c in neighbor_pos
                    if 0 <= n_r < rows
                    and 0 <= n_c < cols
                    and len(hole_view[n_r, n_c]) > 0
                }
            )

            if len(neighbor_holes) == 0:

                # Creating new hole
                new_id = frozenset([next_hole_id])
                hole_sizes[new_id] = 1
                next_hole_id += 1
            else:
                new_id = hole_sizes.merge(neighbor_holes)
                hole_sizes[new_id] += 1

            hole_view[r, c] = new_id
            for primitive_id in new_id:
                ids_in_row.add(frozenset([primitive_id]))

        # prune the tree for memory efficiency
        # if a subtree was not seen in this row,
        # that hole/subtree is not coming back
        # (except due to cylindrical wraparound)
        if len(hole_sizes) >= next_prune_size:
            # print(f"Pruning because {len(hole_sizes)} >= {next_prune_size}")
            subtrees_to_prune = set()
            for k in hole_sizes.roots:  # hole_sizes.mapping.items():
                # # look only at root nodes
                # if not isinstance(v, int):
                #     continue

                # prune if none of its leaves/branches were seen
                if not any(frozenset([primitive_id]) in ids_in_row for primitive_id in k):
                    subtrees_to_prune.add(k)

            for subtree in subtrees_to_prune:
                hole_sizes.prune_subtree(subtree)
            
            next_prune_size = int(np.exp(np.ceil(np.log(len(hole_sizes)))))

        # Early stopping if reached critical size
        if critical_size and hole_view.max >= critical_size:
            break

    return hole_sizes, hole_view


def test_hole_size_dict():
    hsd = HoleSizeDict({frozenset([1]): 1, frozenset([2]): 2})

    # Merging
    hsd.merge([frozenset([1]), frozenset([2])])
    assert hsd[frozenset([1, 2])] == 3

    # Mapping updates to correct destination
    hsd[frozenset([1])] += 5
    assert hsd[frozenset([1, 2])] == 8

    # Subtree pruning
    try:
        hsd.prune_subtree(frozenset([1]))
        assert False, "Expected ValueError (not pruning root node)"
    except ValueError:
        pass

    hsd[frozenset([3])] = 1
    hsd.prune_subtree(frozenset([1, 2]))
    assert len(hsd) == 1 and frozenset([3]) in hsd


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
        hole_sizes, hole_view = detect_holes(test_array)
        max_hole = hole_sizes.get_max()

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
    fig, axs = plt.subplots(nrows=4, ncols=1)

    rng = np.random.default_rng(0)
    side_length = [10, 100, 200, 300, 400, 500]
    density = np.arange(0, 1.1, 0.1)

    for d in density:
        runtimes = []
        dict_sizes = []
        tree_depths = []
        max_hole = []
        for s in side_length:
            a = rng.binomial(1, 1 - d, size=s * s).reshape((s, s))

            tick = perf_counter()
            hole_sizes, _ = detect_holes(a)
            tock = perf_counter()

            runtimes.append(tock - tick)
            dict_sizes.append(len(hole_sizes.mapping))
            tree_depths.append(hole_sizes.get_depth())
            max_hole.append(hole_sizes.get_max())

            print(f"Runtime for side length {s}, density {d:.1f} : {tock-tick} seconds")

        axs[0].plot(
            side_length,
            runtimes,
            label=f"Density={d:.1f}",
            color=(0, 1 - (2 * d - 1) ** 2, d),
        )

        axs[1].plot(
            side_length,
            dict_sizes,
            label=f"Density={d:.1f}",
            color=(1 - (2 * d - 1) ** 2, d, 0),
        )

        axs[2].plot(
            side_length,
            tree_depths,
            label=f"Density={d:.1f}",
            color=(d, 0, 1 - (2 * d - 1) ** 2),
        )

        axs[3].plot(
            side_length,
            max_hole,
            label=f"Density={d:.1f}",
            color=(d, (2 * d - 1) ** 2, (1 - d) / 2),
        )

    axs[0].set_title("Runtime vs. Side Length, Density")
    axs[0].set_xlabel("Side length")
    axs[0].set_ylabel("Runtime (s)")
    axs[0].legend()

    axs[1].set_title("Tree Size vs. Side Length, Density")
    axs[1].set_xlabel("Side Length")
    axs[1].set_ylabel("Size (nodes)")
    axs[1].legend()

    axs[2].set_title("Tree Depth vs. Side Length, Density")
    axs[2].set_xlabel("Side Length")
    axs[2].set_ylabel("Tree Depth")
    axs[2].legend()

    axs[3].plot(
        side_length,
        np.repeat(np.pi * 20**2, len(side_length)),
        "k--",
        label="Critical Size",
    )
    axs[3].set_title("Max Hole Size vs. Side Length, Density")
    axs[3].set_xlabel("Side Length")
    axs[3].set_ylabel("Maximum Hole Size")
    axs[3].legend()

    fig.set_size_inches(8, 24)
    fig.tight_layout()
    fig.savefig("out/hole_detection/test_runtime.png")


def test_case(side_length, density, rng=np.random.default_rng(0)):
    a = rng.binomial(1, 1 - density, size=side_length * side_length).reshape(
        (side_length, side_length)
    )
    detect_holes(a)


def test_merge_time():
    import cProfile
    import pstats

    os.makedirs("out/hole_detection/merge_profile", exist_ok=True)

    side_length = [10, 100, 200, 300, 400]
    density = np.arange(0, 1.1, 0.1)

    merge_times = np.zeros((len(side_length), len(density)))
    total_times = np.zeros((len(side_length), len(density)))

    for r, s in enumerate(side_length):
        for c, d in enumerate(density):
            f = f"out/hole_detection/merge_profile/prof_{s}_{int(d*10)}"
            cProfile.run(
                f"test_case({s}, {d})",
                f,
            )

            p = pstats.Stats(f)
            p.strip_dirs()
            try:
                merge_time = [v for k, v in p.stats.items() if k[2] == "merge"][0][3]
            except IndexError:
                merge_time = 0

            total_time = [v for k, v in p.stats.items() if k[2] == "detect_holes"][0][3]

            merge_times[r, c] = merge_time
            total_times[r, c] = total_time

    fig, axs = plt.subplots(nrows=len(side_length), ncols=1)

    # Time vs. Density plots
    for r, s in enumerate(side_length):
        axs[r].plot(density, merge_times[r, :], label="Merging time", color="b")
        axs[r].plot(density, total_times[r, :], label="Total time", color="k")
        axs[r].set_title(f"Runtime vs. Density (Side length={s})")
        axs[r].set_xlabel("Density")
        axs[r].set_ylabel("Runtime (s)")
        axs[r].legend()

        ax2 = axs[r].twinx()
        ax2.plot(
            density,
            merge_times[r, :] / total_times[r, :],
            "r--",
            label="% Time Merging",
        )
        ax2.set_ylabel("% Time Merging")
        ax2.set_ylim([0, 1])
        ax2.legend()

    fig.set_size_inches(6, 3 * len(side_length))
    fig.tight_layout()
    fig.savefig("out/hole_detection/merge_time.png")


def main():
    test_hole_size_dict()
    test_detect_holes()
    test_runtime()
    test_merge_time()


if __name__ == "__main__":
    main()
