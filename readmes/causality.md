# Causality Networks

> **Note:** Broken as of June 25, 2024, due to new sim output format.

> **Note:** Must be a member of the `CovertLab` GitHub organization.

[The Causality visualization tool](https://github.com/CovertLab/causality) can be used to manually examine the model's
output (molecule counts, reaction fluxes, etc.). First, [run a simulation](#running-the-simulation) ensuring that data is set to be emitted to MongoDB (e.g. `emitter` is `database` in [configuration](readmes/ecoli_configurations.md)). Then, build the Causality network with the following command:

```
python ecoli/analysis/buildCausalityNetwork.py
```

Then, clone the Causality repository into the same folder as the vivarium-ecoli repository (e.g. `/dev/causality` and `/dev/vivarium-ecoli`) and follow the installation instructions.

Finally, start the Causality server, which will open an interactive webpage in your web browser:

```
cd /dev/causality
python site/server.py ../vivarium-ecoli/out/seriesOut/
```

> **Note:** The first timestep in the data is not included when building a causality network.
