# Causality Networks

To build a causality network, use the command `python ecoli/analysis/buildCausalityNetwork.py`.
Use the `--show` flag to open up the resulting visualization in your web browser.
Use the `--id` flag to pass in an experiment id that you want the network to be built from.
Without the `--id` flag, you will be asked to input an experiment id while the program is running.

The first timestep in the data is not included when building a causality network. 