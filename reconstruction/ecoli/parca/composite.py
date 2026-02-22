"""
Composite builder for the ParCa pipeline as process-bigraph Steps.

All data flows through a registered ``'parca_state'`` type that wraps
sim_data + cell_specs.  No Step declares ports named ``sim_data`` or
``cell_specs``.

Pure stages (2 and 8) are decomposed into Extract → Compute → Merge
triplets.  The Compute step has only explicit named ports — no
``parca_state`` dependency.  Coupled stages (1, 3–7, 9) pass the full
``parca_state`` through and expose intermediate results as named outputs.

Store chain::

    state_0 → [Stage 1] → state_1
    state_1 → [Extract02] → s02_in/* → [InputAdj] → s02_out/*
              → [Merge02 + state_1] → state_2
    state_2 → [Stage 3] → state_3 + stage_03/*
    state_3 → [Stage 4] → state_4 + stage_04/*
    state_4 → [Stage 5] → state_5 + stage_05/*
    state_5 → [Stage 6] → state_6 + stage_06/*
    state_6 → [Stage 7] → state_7 + stage_07/*
    state_7 → [Extract08] → s08_in/* → [SetCond] → s08_out/*
              → [Merge08 + state_7] → state_8
    state_8 → [Stage 9] → state_9

Provides:
    ``register_parca_steps(core)`` — register types and all Step links
    ``build_parca_composite(raw_data, core=None, **kwargs)`` — build and
        return a Composite whose Steps execute the full ParCa pipeline
    ``run_parca(raw_data, **kwargs)`` — convenience function that builds
        the Composite, then returns the fitted ``sim_data``
"""

from process_bigraph import Composite, allocate_core

from reconstruction.ecoli.simulation_data import SimulationDataEcoli
from reconstruction.ecoli.parca.parca_types import ParcaState, register_parca_types
from reconstruction.ecoli.parca.steps import ALL_STEP_CLASSES


def register_parca_steps(core):
    """Register the parca_state type and all Step classes with the core."""
    register_parca_types(core)
    core.register_links(ALL_STEP_CLASSES)
    return core


def build_parca_composite(raw_data, core=None, **kwargs):
    """Build a Composite that runs the full ParCa pipeline as Steps.

    Steps execute in dependency order during ``Composite.__init__()``.
    Each stage reads from ``state_<N-1>`` and writes to ``state_<N>``,
    forming a DAG that enforces sequential execution (1 → 2 → ... → 9).

    Pure stages (2, 8) are decomposed into Extract/Compute/Merge triplets
    so the Compute step has only explicit typed ports.

    Args:
        raw_data: A ``KnowledgeBaseEcoli`` instance.
        core: Optional pre-configured core.  If None, one is allocated
            and Step classes + types are registered automatically.
        **kwargs: Pipeline configuration forwarded to step configs:
            cpus (int), debug (bool), cache_dir (str),
            variable_elongation_transcription (bool),
            variable_elongation_translation (bool),
            disable_ribosome_capacity_fitting (bool),
            disable_rnapoly_capacity_fitting (bool).

    Returns:
        A ``Composite`` instance with the pipeline already executed.
        The final sim_data is at ``composite.state['state_9'].sim_data``.
    """
    if core is None:
        core = allocate_core(top=ALL_STEP_CLASSES)
        register_parca_types(core)

    # Initial ParcaState: empty sim_data + empty cell_specs
    state_0 = ParcaState(sim_data=SimulationDataEcoli(), cell_specs={})

    # Extract config values with defaults
    cpus = kwargs.get('cpus', 1)
    debug = kwargs.get('debug', False)
    cache_dir = kwargs.get('cache_dir', '')
    var_elong_tx = kwargs.get('variable_elongation_transcription', True)
    var_elong_tl = kwargs.get('variable_elongation_translation', False)
    disable_ribo = kwargs.get('disable_ribosome_capacity_fitting', False)
    disable_rnap = kwargs.get('disable_rnapoly_capacity_fitting', False)

    # Port names for stage 2 extract outputs / pure step inputs
    s02_in_fields = [
        'monomer_ids', 'translation_efficiencies',
        'translation_eff_adjustments', 'balanced_translation_groups',
        'rna_ids', 'cistron_ids', 'basal_rna_expression',
        'rna_expression_adjustments', 'cistron_id_to_rna_indexes',
        'rna_deg_rates', 'cistron_deg_rates', 'rna_deg_rate_adjustments',
        'protein_deg_rates', 'protein_deg_rate_adjustments',
        'tf_to_active_inactive_conditions',
    ]

    # Port names for stage 2 pure step outputs / merge inputs
    s02_out_fields = [
        'translation_efficiencies', 'basal_rna_expression',
        'rna_deg_rates', 'cistron_deg_rates', 'protein_deg_rates',
        'tf_to_active_inactive_conditions',
    ]

    # Port names for stage 8 extract outputs / pure step inputs
    s08_in_fields = [
        'conditions', 'is_mRNA', 'is_tRNA', 'is_rRNA',
        'includes_ribosomal_protein', 'includes_RNAP',
    ]

    # Port names for stage 8 pure step outputs / merge inputs
    s08_out_fields = [
        'rnaSynthProbFraction', 'rnapFractionActiveDict',
        'rnaSynthProbRProtein', 'rnaSynthProbRnaPolymerase',
        'rnaPolymeraseElongationRateDict', 'expectedDryMassIncreaseDict',
        'ribosomeElongationRateDict', 'ribosomeFractionActiveDict',
        'condition_outputs',
    ]

    spec = {
        'state': {
            # ---- Initial stores ----
            'state_0': state_0,
            'raw_data': raw_data,

            # ---- Stage 1: Initialize ----
            'initialize': {
                '_type': 'step',
                'address': 'local:InitializeStep',
                'config': {},
                'inputs': {
                    'state': ['state_0'],
                    'raw_data': ['raw_data'],
                },
                'outputs': {
                    'state': ['state_1'],
                },
            },

            # ---- Stage 2: Extract → Pure InputAdj → Merge ----
            'extract_02': {
                '_type': 'step',
                'address': 'local:ExtractForStage2Step',
                'config': {
                    'debug': debug,
                },
                'inputs': {
                    'state': ['state_1'],
                },
                'outputs': {f: ['s02_in', f] for f in s02_in_fields},
            },

            'input_adjustments': {
                '_type': 'step',
                'address': 'local:InputAdjustmentsStep',
                'config': {
                    'debug': debug,
                },
                'inputs': {f: ['s02_in', f] for f in s02_in_fields},
                'outputs': {f: ['s02_out', f] for f in s02_out_fields},
            },

            'merge_02': {
                '_type': 'step',
                'address': 'local:MergeAfterStage2Step',
                'config': {},
                'inputs': {
                    'state': ['state_1'],
                    **{f: ['s02_out', f] for f in s02_out_fields},
                },
                'outputs': {
                    'state': ['state_2'],
                },
            },

            # ---- Stage 3: Basal Specs ----
            'basal_specs': {
                '_type': 'step',
                'address': 'local:BasalSpecsStep',
                'config': {
                    'variable_elongation_transcription': var_elong_tx,
                    'variable_elongation_translation': var_elong_tl,
                    'disable_ribosome_capacity_fitting': disable_ribo,
                    'disable_rnapoly_capacity_fitting': disable_rnap,
                    'cache_dir': cache_dir,
                },
                'inputs': {
                    'state': ['state_2'],
                },
                'outputs': {
                    'state': ['state_3'],
                    'conc_dict': ['stage_03', 'conc_dict'],
                    'expression': ['stage_03', 'expression'],
                    'synth_prob': ['stage_03', 'synth_prob'],
                    'fit_cistron_expression': ['stage_03', 'fit_cistron_expression'],
                    'doubling_time': ['stage_03', 'doubling_time'],
                    'avg_cell_dry_mass_init': ['stage_03', 'avg_cell_dry_mass_init'],
                    'fit_avg_soluble_target_mol_mass': ['stage_03', 'fit_avg_soluble_target_mol_mass'],
                    'bulk_container': ['stage_03', 'bulk_container'],
                },
            },

            # ---- Stage 4: TF Condition Specs ----
            'tf_condition_specs': {
                '_type': 'step',
                'address': 'local:TfConditionSpecsStep',
                'config': {
                    'variable_elongation_transcription': var_elong_tx,
                    'variable_elongation_translation': var_elong_tl,
                    'disable_ribosome_capacity_fitting': disable_ribo,
                    'disable_rnapoly_capacity_fitting': disable_rnap,
                    'cpus': cpus,
                },
                'inputs': {
                    'state': ['state_3'],
                },
                'outputs': {
                    'state': ['state_4'],
                    'condition_outputs': ['stage_04', 'condition_outputs'],
                },
            },

            # ---- Stage 5: Fit Condition ----
            'fit_condition': {
                '_type': 'step',
                'address': 'local:FitConditionStep',
                'config': {
                    'cpus': cpus,
                },
                'inputs': {
                    'state': ['state_4'],
                },
                'outputs': {
                    'state': ['state_5'],
                    'condition_outputs': ['stage_05', 'condition_outputs'],
                    'translation_supply_rate': ['stage_05', 'translation_supply_rate'],
                },
            },

            # ---- Stage 6: Promoter Binding ----
            'promoter_binding': {
                '_type': 'step',
                'address': 'local:PromoterBindingStep',
                'config': {},
                'inputs': {
                    'state': ['state_5'],
                },
                'outputs': {
                    'state': ['state_6'],
                    'r_vector': ['stage_06', 'r_vector'],
                    'r_columns': ['stage_06', 'r_columns'],
                },
            },

            # ---- Stage 7: Adjust Promoters ----
            'adjust_promoters': {
                '_type': 'step',
                'address': 'local:AdjustPromotersStep',
                'config': {},
                'inputs': {
                    'state': ['state_6'],
                },
                'outputs': {
                    'state': ['state_7'],
                    'basal_prob': ['stage_07', 'basal_prob'],
                    'delta_prob': ['stage_07', 'delta_prob'],
                },
            },

            # ---- Stage 8: Extract → Pure SetConditions → Merge ----
            'extract_08': {
                '_type': 'step',
                'address': 'local:ExtractForStage8Step',
                'config': {},
                'inputs': {
                    'state': ['state_7'],
                },
                'outputs': {f: ['s08_in', f] for f in s08_in_fields},
            },

            'set_conditions': {
                '_type': 'step',
                'address': 'local:SetConditionsStep',
                'config': {},
                'inputs': {f: ['s08_in', f] for f in s08_in_fields},
                'outputs': {f: ['s08_out', f] for f in s08_out_fields},
            },

            'merge_08': {
                '_type': 'step',
                'address': 'local:MergeAfterStage8Step',
                'config': {},
                'inputs': {
                    'state': ['state_7'],
                    **{f: ['s08_out', f] for f in s08_out_fields},
                },
                'outputs': {
                    'state': ['state_8'],
                },
            },

            # ---- Stage 9: Final Adjustments ----
            'final_adjustments': {
                '_type': 'step',
                'address': 'local:FinalAdjustmentsStep',
                'config': {},
                'inputs': {
                    'state': ['state_8'],
                },
                'outputs': {
                    'state': ['state_9'],
                },
            },
        },
    }

    composite = Composite(spec, core=core)
    return composite


def run_parca(raw_data, **kwargs):
    """Run the full ParCa pipeline as process-bigraph Steps.

    This is the main entry point.  Steps execute during Composite
    construction via the DAG dependency chain.

    Args:
        raw_data: A ``KnowledgeBaseEcoli`` instance.
        **kwargs: Forwarded to ``build_parca_composite()``.

    Returns:
        The fitted ``SimulationDataEcoli`` object.
    """
    composite = build_parca_composite(raw_data, **kwargs)
    return composite.state['state_9'].sim_data
