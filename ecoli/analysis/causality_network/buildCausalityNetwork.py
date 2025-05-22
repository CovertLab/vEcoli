"""
Builds causality network for a given variant of a given sim.

Run with '-h' for command line help.
"""

import argparse
import datetime
import os
import pprint as pp
import subprocess
import time

from ecoli.analysis.causality_network import read_dynamics
from ecoli.analysis.causality_network.build_network import BuildNetwork
from wholecell.utils import filepath as fp
from time import monotonic as monotonic_seconds
from time import process_time as process_time_seconds


CAUSALITY_ENV_VAR = "CAUSALITY_SERVER"
SIM_DATA_PATH = "out/kb/simData.cPickle"
DYNAMICS_OUTPUT = "out/seriesOut"


class BuildCausalityNetwork:
    """Builds and runs a causality network for a given sim."""

    def description(self):
        # type: () -> str
        """Describe the command line program. This defaults to the class name."""
        return type(self).__name__

    def help(self):
        # type: () -> str
        """Return help text for the Command Line Interface. This defaults to a
        string constructed around `self.description()`.
        """
        return "Run {}.".format(self.description())

    def define_parameters(self, parser):
        parser.add_argument(
            "--check_sanity", action="store_true", help="Check network sanity."
        )
        parser.add_argument(
            "--show",
            action="store_true",
            help="If set, attempts to show the causality visualization after"
            " processing data.",
        )
        parser.add_argument(
            "--id",
            type=str,
            default="",
            help="If set, a causality network is built using a custom listener dataset.",
        )

    def parse_args(self):
        # type: () -> argparse.Namespace
        """Parse the command line args: Construct an ArgumentParser, call
        `define_parameters()` to define parameters including subclass-specific
        parameters, use it to parse the command line into an
        `argparse.Namespace`, and return that.

        (A `Namespace` is an object with attributes and some methods like
        `__repr__()` and `__eq__()`. Call `vars(args)` to turn it into a dict.)
        """
        parser = argparse.ArgumentParser(description=self.help())

        self.define_parameters(parser)

        return parser.parse_args()

    def run(self, args):
        start_real_sec = monotonic_seconds()
        print("\n{}: --- Starting {} ---".format(time.ctime(), type(self).__name__))

        print("{}: Building the Causality network".format(time.ctime()))
        causality_network = BuildNetwork(
            SIM_DATA_PATH, DYNAMICS_OUTPUT, args.check_sanity
        )
        node_list, edge_list = causality_network.build_nodes_and_edges()

        fp.makedirs(DYNAMICS_OUTPUT)

        print(
            "{}: Converting simulation results to a Causality series".format(
                time.ctime()
            )
        )

        read_dynamics.convert_dynamics(
            DYNAMICS_OUTPUT, causality_network.sim_data, node_list, edge_list, args.id
        )

        elapsed_real_sec = monotonic_seconds() - start_real_sec

        duration = datetime.timedelta(seconds=elapsed_real_sec)
        print(
            "{}: Completed building the Causality network in {}".format(
                time.ctime(), duration
            )
        )

        # Optionally show the causality visualization.
        server_dir = os.environ.get(CAUSALITY_ENV_VAR, os.path.join("..", "causality"))
        server_app = os.path.join("site", "server.py")
        server_path = os.path.join(server_dir, server_app)
        if args.show and os.path.isfile(server_path):
            # See #890 if running command fails due to differences in pyenv
            # versions - might need to cd to repo and activate pyenv
            cmd = ["python", server_path, DYNAMICS_OUTPUT]
            print(
                f"\nServing the Causality site via the command:\n  {cmd}\n"
                f"Ctrl+C to exit.\n"
            )
            subprocess.run(cmd)
        else:
            print(
                "\nNOTE: Use the --show flag to automatically open the"
                " Casuality viewer on this data. You'll first need to"
                " `export {0}=~/path/to/causality` project unless the default"
                " (../causality) is good.\n".format(CAUSALITY_ENV_VAR)
            )


def main():
    network = BuildCausalityNetwork()
    args = network.parse_args()

    location = getattr(args, "sim_path", "")
    if location:
        location = " at " + location

    start_real_sec = monotonic_seconds()
    print("{}: {}{}".format(time.ctime(), network.description(), location))
    pp.pprint({"Arguments": vars(args)})

    start_process_sec = process_time_seconds()
    network.run(args)

    elapsed_process = process_time_seconds() - start_process_sec
    elapsed_real_sec = monotonic_seconds() - start_real_sec
    print(
        "{}: Elapsed time {:1.2f} sec ({}); CPU {:1.2f} sec".format(
            time.ctime(),
            elapsed_real_sec,
            datetime.timedelta(seconds=elapsed_real_sec),
            elapsed_process,
        )
    )


if __name__ == "__main__":
    main()
