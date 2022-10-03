import numpy as np
import matplotlib.pyplot as plt

from ecoli.analysis.mass_fraction_summary import Plot
from vivarium.core.emitter import DatabaseEmitter, data_from_database
from ecoli.analysis.tablereader import MAPPING, TableReader
from ecoli.library.sim_data import LoadSimData


class MureinPlots:
    def __init__(self):
        # get ids, masses of small molecules from sim data
        self.sim_data = LoadSimData()
        self.mass_config = self.sim_data.get_mass_listener_config()

        self.bulk_ids = self.mass_config["bulk_ids"]
        small_submass_idx = self.mass_config["submass_indices"]["smallMolecule"]
        self.small_molecule_masses = self.mass_config["bulk_masses"][
            :, small_submass_idx
        ]
        self.small_molecule_dict = dict(zip(self.bulk_ids, self.small_molecule_masses))

        # mongo client
        config = {"host": "localhost:27017", "database": "simulations"}
        self.emitter = DatabaseEmitter(config)

    def do_plots(self, experiment_id):
        # output directory
        subdir = experiment_id.replace("/", "_").replace(" ", "__").replace(":", "_")
        self.outdir = f"out/analysis/{subdir}"

        self.mass_fraction_summary(experiment_id)
        self.small_molecule_summary(experiment_id)

    def mass_fraction_summary(self, experiment_id):
        print(f"Doing mass fraction summary for experiment '{experiment_id}'...")

        # Get data, querying only what is necessary for massFractionSummary
        query = [
            v for v in {**MAPPING["Mass"], **MAPPING["Main"]}.values() if v is not None
        ]
        data, _ = data_from_database(experiment_id, self.emitter.db, query=query)

        # mass fraction summary plot
        Plot(data, out_dir=self.outdir)

        print("Done.")

    def small_molecule_summary(self, experiment_id):
        print(f"Doing small molecule summary for experiment '{experiment_id}'...")
        # Get counts of bulk molecules
        data, _ = data_from_database(experiment_id, self.emitter.db, query=[("bulk",)])
        tb = TableReader("BulkMolecules", data)
        bulk_counts = tb.readColumn("counts")
        bulk_names = tb.readColumn("objectNames")
        bulk_dict = dict(zip(bulk_names, bulk_counts.T))

        small_molecule_mass_plot = np.array([
                submass * bulk_dict[molecule]
                for molecule, submass in self.small_molecule_dict.items()
            ]).T

        # plot all small molecule counts
        fig, ax = plt.subplots()
        x = list(sorted(data.keys()))
        ax.plot(
            x,
            small_molecule_mass_plot,
            color="y",
        )
        ax.set_xticks(x)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Mass")

        ax2 = ax.twinx()
        ax2.plot(x, small_molecule_mass_plot.sum(axis=1), color="k")
        ax2.set_ylabel("Total Small Molecule Mass")

        fig.tight_layout()
        fig.savefig(self.outdir + "/smallMolecules.png")

        print("Done.")


def main():
    # Ids of experiments to plot
    experiment_ids = [
        'test_murein_28/06/2022 15:49:59',
    ]

    for exp_id in experiment_ids:
        MureinPlots().do_plots(exp_id)


if __name__ == "__main__":
    main()
