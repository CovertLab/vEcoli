"""
ParCa pipeline stages as pure functions with explicit Input/Output dataclasses.

Each stage module provides three functions:
    extract_input(sim_data, cell_specs, **kwargs) -> StageInput
    compute_*(inp: StageInput) -> StageOutput
    merge_output(sim_data, cell_specs, out: StageOutput)
"""
