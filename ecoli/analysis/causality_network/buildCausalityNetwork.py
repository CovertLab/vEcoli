"""
Builds a causality network for a given experiment/variant of a Parquet-based
vEcoli simulation run.

Run with '-h' for command-line help.
"""

import argparse
import datetime
import os
import pprint as pp
import subprocess
import time
from time import monotonic as monotonic_seconds
from time import process_time as process_time_seconds
from urllib import parse

from ecoli.analysis.causality_network import read_dynamics
from ecoli.analysis.causality_network.build_network import BuildNetwork
from ecoli.library.parquet_emitter import create_duckdb_conn, dataset_sql
from wholecell.utils import filepath as fp


CAUSALITY_ENV_VAR = "CAUSALITY_SERVER"

# (arg_name, dtype) — narrower filters override wider ones.
CELL_FILTERS = [
    ("variant", int),
    ("lineage_seed", int),
    ("generation", int),
    ("agent_id", str),
]


def build_where_clause(experiment_id: str, args: argparse.Namespace) -> str:
    """Compose a DuckDB WHERE clause pinning to one experiment and any
    optional variant/seed/generation/agent_id the user supplied."""
    quoted_exp = parse.quote_plus(experiment_id)
    clauses = [f"experiment_id = '{quoted_exp}'"]
    for name, dtype in CELL_FILTERS:
        val = getattr(args, name)
        if val is None:
            continue
        if dtype is str:
            clauses.append(f"{name} = '{parse.quote_plus(str(val))}'")
        else:
            clauses.append(f"{name} = {val}")
    return " AND ".join(clauses)


class BuildCausalityNetwork:
    """Builds and (optionally) serves a causality network for one sim."""

    def define_parameters(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--out_dir",
            required=True,
            help="Parent of the <experiment_id>/{history,configuration,...} "
            "tree (i.e. the emitter's out_dir/out_uri, NOT the per-experiment "
            "subdirectory).",
        )
        parser.add_argument(
            "--experiment_id",
            required=True,
            help="Experiment ID to build the network for.",
        )
        parser.add_argument(
            "--sim_data_path",
            required=True,
            help="Path to the variant sim_data pickle "
            "(e.g. out/<exp>/variant_sim_data/0.cPickle).",
        )
        parser.add_argument(
            "--series_out",
            default=None,
            help="Directory to write seriesOut.zip into. "
            "Defaults to <out_dir>/<experiment_id>/seriesOut.",
        )
        for name, dtype in CELL_FILTERS:
            parser.add_argument(f"--{name}", type=dtype, default=None)
        parser.add_argument(
            "--cpus",
            type=int,
            default=None,
            help="DuckDB thread count.",
        )
        parser.add_argument(
            "--check_sanity",
            action="store_true",
            help="Check network for duplicate node IDs.",
        )
        parser.add_argument(
            "--show",
            action="store_true",
            help="Launch the Causality viewer against the output when done.",
        )

    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=type(self).__name__)
        self.define_parameters(parser)
        return parser.parse_args()

    def run(self, args: argparse.Namespace) -> None:
        start_real_sec = monotonic_seconds()
        print(f"\n{time.ctime()}: --- Starting {type(self).__name__} ---")

        series_out = args.series_out or os.path.join(
            args.out_dir, args.experiment_id, "seriesOut"
        )
        fp.makedirs(series_out)

        print(f"{time.ctime()}: Building the Causality network")
        network = BuildNetwork(args.sim_data_path, series_out, args.check_sanity)
        node_list, edge_list = network.build_nodes_and_edges()

        print(f"{time.ctime()}: Reading simulation dynamics from Parquet")
        gcs_bucket = parse.urlparse(args.out_dir).scheme in ("gcs", "gs")
        conn = create_duckdb_conn(args.out_dir, gcs_bucket, args.cpus)
        history_sql, config_sql, _ = dataset_sql(args.out_dir, [args.experiment_id])
        where = build_where_clause(args.experiment_id, args)
        history_sql = f"SELECT * FROM ({history_sql}) WHERE {where}"
        config_sql = f"SELECT * FROM ({config_sql}) WHERE {where}"

        print(f"{time.ctime()}: Converting simulation results to a Causality series")
        read_dynamics.convert_dynamics(
            series_out,
            network.sim_data,
            node_list,
            edge_list,
            args.experiment_id,
            conn,
            history_sql,
            config_sql,
        )

        duration = datetime.timedelta(seconds=monotonic_seconds() - start_real_sec)
        print(f"{time.ctime()}: Completed building the Causality network in {duration}")

        server_dir = os.environ.get(CAUSALITY_ENV_VAR, os.path.join("..", "causality"))
        server_path = os.path.join(server_dir, "site", "server.py")
        if args.show and os.path.isfile(server_path):
            cmd = ["python", server_path, series_out]
            print(f"\nServing the Causality site via:\n  {cmd}\nCtrl+C to exit.\n")
            subprocess.run(cmd)
        elif args.show:
            print(
                f"\nCannot find Causality server at {server_path}. "
                f"Set {CAUSALITY_ENV_VAR}=/path/to/causality or clone the "
                "Causality repo alongside this one.\n"
            )
        else:
            print(
                "\nNOTE: Use --show to auto-launch the Causality viewer on this "
                f"output. Set {CAUSALITY_ENV_VAR} if it lives outside ../causality.\n"
            )


def main() -> None:
    network = BuildCausalityNetwork()
    args = network.parse_args()

    start_real_sec = monotonic_seconds()
    print(f"{time.ctime()}: BuildCausalityNetwork")
    pp.pprint({"Arguments": vars(args)})

    start_process_sec = process_time_seconds()
    network.run(args)

    elapsed_process = process_time_seconds() - start_process_sec
    elapsed_real_sec = monotonic_seconds() - start_real_sec
    print(
        f"{time.ctime()}: Elapsed time {elapsed_real_sec:1.2f} sec "
        f"({datetime.timedelta(seconds=elapsed_real_sec)}); "
        f"CPU {elapsed_process:1.2f} sec"
    )


if __name__ == "__main__":
    main()
