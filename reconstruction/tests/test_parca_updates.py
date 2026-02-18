"""
Tests for the parca_updates module and pure function implementations.

These tests verify:
1. The update dataclasses work correctly
2. The apply functions properly update objects
3. Pure functions produce equivalent results to the original mutating functions
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from reconstruction.ecoli.parca_updates import (
    ArrayUpdate,
    SimDataUpdate,
    CellSpecsUpdate,
    StageResult,
    get_by_path,
    set_by_path,
    extract_path,
    apply_array_update,
    apply_sim_data_update,
    apply_cell_specs_update,
    apply_stage_result,
    make_attribute_update,
    make_array_multiply_update,
    make_array_set_update,
    make_dict_update,
    collect_updates,
)

# Silence Sphinx autodoc warning
unittest.TestCase.__module__ = "unittest"


class TestArrayUpdate(unittest.TestCase):
    """Tests for the ArrayUpdate dataclass."""

    def test_valid_ops(self):
        """Test that valid operations are accepted."""
        valid_ops = ['set', 'set_slice', 'multiply', 'add', 'divide', 'normalize']
        for op in valid_ops:
            update = ArrayUpdate(op=op, value=1.0)
            self.assertEqual(update.op, op)

    def test_invalid_op_raises(self):
        """Test that invalid operations raise ValueError."""
        with self.assertRaises(ValueError):
            ArrayUpdate(op='invalid_op', value=1.0)

    def test_with_indices(self):
        """Test ArrayUpdate with indices."""
        update = ArrayUpdate(op='set', value=5.0, indices=[0, 1, 2])
        self.assertEqual(update.indices, [0, 1, 2])

    def test_with_field(self):
        """Test ArrayUpdate with field for structured arrays."""
        update = ArrayUpdate(op='multiply', value=2.0, field='deg_rate')
        self.assertEqual(update.field, 'deg_rate')


class TestSimDataUpdate(unittest.TestCase):
    """Tests for the SimDataUpdate dataclass."""

    def test_empty_update(self):
        """Test creating an empty update."""
        update = SimDataUpdate()
        self.assertEqual(update.attributes, {})
        self.assertEqual(update.arrays, {})
        self.assertEqual(update.dicts, {})
        self.assertEqual(update.method_calls, [])

    def test_merge(self):
        """Test merging two SimDataUpdate objects."""
        update1 = SimDataUpdate(
            attributes={'a': 1, 'b': 2},
            arrays={'arr1': ArrayUpdate(op='set', value=1)},
            dicts={'d1': {'k1': 'v1'}},
        )
        update2 = SimDataUpdate(
            attributes={'b': 3, 'c': 4},
            arrays={'arr2': ArrayUpdate(op='multiply', value=2)},
            dicts={'d1': {'k2': 'v2'}, 'd2': {'k3': 'v3'}},
        )

        merged = update1.merge(update2)

        # Attributes: update2 takes precedence
        self.assertEqual(merged.attributes, {'a': 1, 'b': 3, 'c': 4})
        # Arrays: both present
        self.assertIn('arr1', merged.arrays)
        self.assertIn('arr2', merged.arrays)
        # Dicts: deeply merged
        self.assertEqual(merged.dicts['d1'], {'k1': 'v1', 'k2': 'v2'})
        self.assertEqual(merged.dicts['d2'], {'k3': 'v3'})


class TestCellSpecsUpdate(unittest.TestCase):
    """Tests for the CellSpecsUpdate dataclass."""

    def test_empty_update(self):
        """Test creating an empty update."""
        update = CellSpecsUpdate()
        self.assertEqual(update.conditions, {})

    def test_merge(self):
        """Test merging two CellSpecsUpdate objects."""
        update1 = CellSpecsUpdate(conditions={
            'basal': {'expression': [1, 2, 3]},
            'with_aa': {'expression': [4, 5, 6]},
        })
        update2 = CellSpecsUpdate(conditions={
            'basal': {'synthProb': [0.1, 0.2, 0.3]},
            'no_oxygen': {'expression': [7, 8, 9]},
        })

        merged = update1.merge(update2)

        self.assertIn('expression', merged.conditions['basal'])
        self.assertIn('synthProb', merged.conditions['basal'])
        self.assertIn('with_aa', merged.conditions)
        self.assertIn('no_oxygen', merged.conditions)


class TestStageResult(unittest.TestCase):
    """Tests for the StageResult dataclass."""

    def test_empty_result(self):
        """Test creating an empty result."""
        result = StageResult()
        self.assertIsInstance(result.sim_data_update, SimDataUpdate)
        self.assertIsInstance(result.cell_specs_update, CellSpecsUpdate)

    def test_merge(self):
        """Test merging two StageResult objects."""
        result1 = StageResult(
            sim_data_update=SimDataUpdate(attributes={'a': 1}),
            cell_specs_update=CellSpecsUpdate(conditions={'basal': {'x': 1}}),
        )
        result2 = StageResult(
            sim_data_update=SimDataUpdate(attributes={'b': 2}),
            cell_specs_update=CellSpecsUpdate(conditions={'basal': {'y': 2}}),
        )

        merged = result1.merge(result2)

        self.assertIn('a', merged.sim_data_update.attributes)
        self.assertIn('b', merged.sim_data_update.attributes)
        self.assertIn('x', merged.cell_specs_update.conditions['basal'])
        self.assertIn('y', merged.cell_specs_update.conditions['basal'])


class TestPathUtils(unittest.TestCase):
    """Tests for path utility functions."""

    def test_extract_path_no_suffix(self):
        """Test extract_path with no suffix."""
        self.assertEqual(extract_path('a.b.c'), 'a.b.c')

    def test_extract_path_with_suffix(self):
        """Test extract_path with unique identifier suffix."""
        self.assertEqual(extract_path('a.b.c:idx0'), 'a.b.c')
        self.assertEqual(extract_path('process.translation.data:protein123'), 'process.translation.data')

    def test_get_by_path(self):
        """Test get_by_path traverses nested attributes."""
        obj = MagicMock()
        obj.a.b.c = 'value'

        result = get_by_path(obj, 'a.b.c')
        self.assertEqual(result, 'value')

    def test_get_by_path_empty(self):
        """Test get_by_path with empty path returns object."""
        obj = 'test'
        self.assertEqual(get_by_path(obj, ''), 'test')

    def test_set_by_path_single_level(self):
        """Test set_by_path with single-level path."""
        obj = MagicMock()
        set_by_path(obj, 'attr', 'value')
        self.assertEqual(obj.attr, 'value')

    def test_set_by_path_nested(self):
        """Test set_by_path with nested path."""
        obj = MagicMock()
        set_by_path(obj, 'a.b.c', 'value')
        self.assertEqual(obj.a.b.c, 'value')


class TestApplyArrayUpdate(unittest.TestCase):
    """Tests for the apply_array_update function."""

    def test_set_full_array(self):
        """Test setting the entire array."""
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        update = ArrayUpdate(op='set', value=0)
        result = apply_array_update(arr, update)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])

    def test_set_with_indices(self):
        """Test setting specific indices."""
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        update = ArrayUpdate(op='set', value=10, indices=[0, 2, 4])
        result = apply_array_update(arr, update)
        np.testing.assert_array_equal(result, [10, 2, 10, 4, 10])

    def test_multiply_full_array(self):
        """Test multiplying the entire array."""
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        update = ArrayUpdate(op='multiply', value=2)
        result = apply_array_update(arr, update)
        np.testing.assert_array_equal(result, [2, 4, 6, 8, 10])

    def test_multiply_with_indices(self):
        """Test multiplying specific indices."""
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        update = ArrayUpdate(op='multiply', value=10, indices=[1, 3])
        result = apply_array_update(arr, update)
        np.testing.assert_array_equal(result, [1, 20, 3, 40, 5])

    def test_add_with_indices(self):
        """Test adding to specific indices."""
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        update = ArrayUpdate(op='add', value=100, indices=[0, 4])
        result = apply_array_update(arr, update)
        np.testing.assert_array_equal(result, [101, 2, 3, 4, 105])

    def test_divide_full_array(self):
        """Test dividing the entire array."""
        arr = np.array([10, 20, 30, 40, 50], dtype=float)
        update = ArrayUpdate(op='divide', value=10)
        result = apply_array_update(arr, update)
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])

    def test_normalize_full_array(self):
        """Test normalizing the entire array."""
        arr = np.array([1, 2, 3, 4], dtype=float)
        update = ArrayUpdate(op='normalize', value=None)
        result = apply_array_update(arr, update)
        np.testing.assert_almost_equal(result.sum(), 1.0)
        np.testing.assert_array_almost_equal(result, [0.1, 0.2, 0.3, 0.4])

    def test_structured_array_field(self):
        """Test updating a field in a structured array."""
        dt = np.dtype([('id', 'U10'), ('deg_rate', float)])
        arr = np.array([('a', 1.0), ('b', 2.0), ('c', 3.0)], dtype=dt)
        update = ArrayUpdate(op='multiply', value=10, field='deg_rate')
        result = apply_array_update(arr, update)
        np.testing.assert_array_equal(result['deg_rate'], [10, 20, 30])


class TestApplySimDataUpdate(unittest.TestCase):
    """Tests for the apply_sim_data_update function."""

    def test_apply_attributes(self):
        """Test applying attribute updates."""
        sim_data = MagicMock()
        sim_data.tf_to_active_inactive_conditions = {'old': 'value'}

        update = SimDataUpdate(attributes={
            'tf_to_active_inactive_conditions': {'new': 'value'}
        })

        apply_sim_data_update(sim_data, update)
        self.assertEqual(sim_data.tf_to_active_inactive_conditions, {'new': 'value'})

    def test_apply_nested_attributes(self):
        """Test applying nested attribute updates."""
        sim_data = MagicMock()
        sim_data.process.translation.efficiencies = 'old'

        update = SimDataUpdate(attributes={
            'process.translation.efficiencies': 'new'
        })

        apply_sim_data_update(sim_data, update)
        self.assertEqual(sim_data.process.translation.efficiencies, 'new')

    def test_apply_arrays(self):
        """Test applying array updates."""
        sim_data = MagicMock()
        sim_data.process.translation.efficiencies = np.array([1.0, 2.0, 3.0])

        update = SimDataUpdate(arrays={
            'process.translation.efficiencies': ArrayUpdate(
                op='multiply', value=2
            )
        })

        apply_sim_data_update(sim_data, update)
        np.testing.assert_array_equal(
            sim_data.process.translation.efficiencies, [2.0, 4.0, 6.0]
        )

    def test_apply_arrays_with_unique_keys(self):
        """Test applying array updates with unique key suffixes."""
        sim_data = MagicMock()
        sim_data.data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        update = SimDataUpdate(arrays={
            'data:idx0': ArrayUpdate(op='multiply', value=10, indices=0),
            'data:idx2': ArrayUpdate(op='multiply', value=100, indices=2),
        })

        apply_sim_data_update(sim_data, update)
        np.testing.assert_array_equal(
            sim_data.data, [10.0, 2.0, 300.0, 4.0, 5.0]
        )

    def test_apply_dicts(self):
        """Test applying dict updates."""
        sim_data = MagicMock()
        sim_data.process.transcription.rna_expression = {
            'basal': np.array([1, 2, 3]),
            'with_aa': np.array([4, 5, 6]),
        }

        new_basal = np.array([10, 20, 30])
        update = SimDataUpdate(dicts={
            'process.transcription.rna_expression': {'basal': new_basal}
        })

        apply_sim_data_update(sim_data, update)
        np.testing.assert_array_equal(
            sim_data.process.transcription.rna_expression['basal'], new_basal
        )


class TestApplyCellSpecsUpdate(unittest.TestCase):
    """Tests for the apply_cell_specs_update function."""

    def test_apply_new_condition(self):
        """Test adding a new condition."""
        cell_specs = {}
        update = CellSpecsUpdate(conditions={
            'basal': {'expression': [1, 2, 3]}
        })

        apply_cell_specs_update(cell_specs, update)
        self.assertEqual(cell_specs['basal']['expression'], [1, 2, 3])

    def test_update_existing_condition(self):
        """Test updating an existing condition."""
        cell_specs = {
            'basal': {'expression': [1, 2, 3]}
        }
        update = CellSpecsUpdate(conditions={
            'basal': {'synthProb': [0.1, 0.2, 0.3]}
        })

        apply_cell_specs_update(cell_specs, update)
        self.assertEqual(cell_specs['basal']['expression'], [1, 2, 3])
        self.assertEqual(cell_specs['basal']['synthProb'], [0.1, 0.2, 0.3])


class TestHelperFunctions(unittest.TestCase):
    """Tests for helper functions that build common update patterns."""

    def test_make_attribute_update(self):
        """Test make_attribute_update helper."""
        update = make_attribute_update('a.b.c', 'value')
        self.assertEqual(update.attributes, {'a.b.c': 'value'})

    def test_make_array_multiply_update(self):
        """Test make_array_multiply_update helper."""
        update = make_array_multiply_update('path', 2.0, indices=[0, 1])
        self.assertIn('path', update.arrays)
        self.assertEqual(update.arrays['path'].op, 'multiply')
        self.assertEqual(update.arrays['path'].value, 2.0)
        self.assertEqual(update.arrays['path'].indices, [0, 1])

    def test_make_array_set_update(self):
        """Test make_array_set_update helper."""
        update = make_array_set_update('path', 5.0, field='deg_rate')
        self.assertIn('path', update.arrays)
        self.assertEqual(update.arrays['path'].op, 'set')
        self.assertEqual(update.arrays['path'].field, 'deg_rate')

    def test_make_dict_update(self):
        """Test make_dict_update helper."""
        update = make_dict_update('path', {'key': 'value'})
        self.assertEqual(update.dicts, {'path': {'key': 'value'}})

    def test_collect_updates(self):
        """Test collect_updates merges multiple updates."""
        u1 = make_attribute_update('a', 1)
        u2 = make_attribute_update('b', 2)
        u3 = make_array_set_update('arr', 5.0)

        combined = collect_updates(u1, u2, u3)

        self.assertEqual(combined.attributes['a'], 1)
        self.assertEqual(combined.attributes['b'], 2)
        self.assertIn('arr', combined.arrays)


# ============================================================================
# Tests for Output Dataclasses and Mappers (parca_outputs.py)
# ============================================================================

from reconstruction.ecoli.parca_outputs import (
    CellSpecsData,
    InitializeOutput,
    InputAdjustmentsOutput,
    BasalSpecsOutput,
    TfConditionSpecsOutput,
    FitConditionOutput,
    PromoterBindingOutput,
    AdjustPromotersOutput,
    SetConditionsOutput,
    FinalAdjustmentsOutput,
    apply_initialize_output,
    apply_input_adjustments_output,
    apply_basal_specs_output,
    apply_tf_condition_specs_output,
    apply_fit_condition_output,
    apply_promoter_binding_output,
    apply_adjust_promoters_output,
    apply_set_conditions_output,
    apply_final_adjustments_output,
    cell_specs_data_to_dict,
    dict_to_cell_specs_data,
)


class TestCellSpecsData(unittest.TestCase):
    """Tests for the CellSpecsData dataclass."""

    def test_creation_with_required_fields(self):
        """Test creating CellSpecsData with required fields only."""
        data = CellSpecsData(
            concDict={'met1': 1.0},
            expression=np.array([0.1, 0.2]),
            doubling_time=60.0,
            synthProb=np.array([0.5, 0.5]),
            fit_cistron_expression=np.array([0.3, 0.3]),
            avgCellDryMassInit=1.0,
            fitAvgSolubleTargetMolMass=0.5,
            bulkContainer=np.array([100, 200]),
        )
        self.assertEqual(data.concDict, {'met1': 1.0})
        self.assertIsNone(data.cistron_expression)
        self.assertIsNone(data.r_vector)

    def test_creation_with_optional_fields(self):
        """Test creating CellSpecsData with optional fields."""
        data = CellSpecsData(
            concDict={'met1': 1.0},
            expression=np.array([0.1, 0.2]),
            doubling_time=60.0,
            synthProb=np.array([0.5, 0.5]),
            fit_cistron_expression=np.array([0.3, 0.3]),
            avgCellDryMassInit=1.0,
            fitAvgSolubleTargetMolMass=0.5,
            bulkContainer=np.array([100, 200]),
            r_vector=np.array([1, 2, 3]),
            r_columns=['a', 'b', 'c'],
        )
        np.testing.assert_array_equal(data.r_vector, [1, 2, 3])
        self.assertEqual(data.r_columns, ['a', 'b', 'c'])

    def test_to_dict_conversion(self):
        """Test converting CellSpecsData to dict."""
        data = CellSpecsData(
            concDict={'met1': 1.0},
            expression=np.array([0.1, 0.2]),
            doubling_time=60.0,
            synthProb=np.array([0.5, 0.5]),
            fit_cistron_expression=np.array([0.3, 0.3]),
            avgCellDryMassInit=1.0,
            fitAvgSolubleTargetMolMass=0.5,
            bulkContainer=np.array([100, 200]),
            r_vector=np.array([1, 2, 3]),
        )
        result = cell_specs_data_to_dict(data)
        self.assertEqual(result['concDict'], {'met1': 1.0})
        np.testing.assert_array_equal(result['expression'], [0.1, 0.2])
        np.testing.assert_array_equal(result['r_vector'], [1, 2, 3])
        self.assertNotIn('cistron_expression', result)  # None values excluded

    def test_from_dict_conversion(self):
        """Test converting dict to CellSpecsData."""
        d = {
            'concDict': {'met1': 1.0},
            'expression': np.array([0.1, 0.2]),
            'doubling_time': 60.0,
            'synthProb': np.array([0.5, 0.5]),
            'fit_cistron_expression': np.array([0.3, 0.3]),
            'avgCellDryMassInit': 1.0,
            'fitAvgSolubleTargetMolMass': 0.5,
            'bulkContainer': np.array([100, 200]),
        }
        data = dict_to_cell_specs_data(d)
        self.assertEqual(data.concDict, {'met1': 1.0})
        self.assertEqual(data.doubling_time, 60.0)
        self.assertIsNone(data.r_vector)


class TestOutputDataclasses(unittest.TestCase):
    """Tests for the stage output dataclasses."""

    def test_initialize_output(self):
        """Test InitializeOutput creation."""
        mock_sim_data = MagicMock()
        output = InitializeOutput(sim_data=mock_sim_data)
        self.assertEqual(output.sim_data, mock_sim_data)

    def test_input_adjustments_output_defaults(self):
        """Test InputAdjustmentsOutput with defaults."""
        output = InputAdjustmentsOutput()
        self.assertEqual(output.translation_efficiencies_multipliers, {})
        self.assertIsNone(output.tf_to_active_inactive_conditions)
        self.assertIsNone(output.rna_expression_basal)

    def test_input_adjustments_output_with_values(self):
        """Test InputAdjustmentsOutput with values."""
        output = InputAdjustmentsOutput(
            translation_efficiencies_multipliers={0: 1.5, 1: 2.0},
            tf_to_active_inactive_conditions={'TF1': {'active': 'cond1'}},
        )
        self.assertEqual(output.translation_efficiencies_multipliers[0], 1.5)
        self.assertIn('TF1', output.tf_to_active_inactive_conditions)

    def test_basal_specs_output(self):
        """Test BasalSpecsOutput creation."""
        basal_specs = CellSpecsData(
            concDict={},
            expression=np.array([0.1]),
            doubling_time=60.0,
            synthProb=np.array([0.5]),
            fit_cistron_expression=np.array([0.3]),
            avgCellDryMassInit=1.0,
            fitAvgSolubleTargetMolMass=0.5,
            bulkContainer=np.array([100]),
        )
        output = BasalSpecsOutput(
            avg_cell_dry_mass_init=1.0,
            avg_cell_dry_mass=2.0,
            avg_cell_water_mass_init=0.7,
            fitAvgSolubleTargetMolMass=0.5,
            rna_expression_basal=np.array([0.1, 0.2]),
            rna_synth_prob_basal=np.array([0.5, 0.5]),
            fit_cistron_expression_basal=np.array([0.3]),
            exp_ppgpp=np.array([0.01]),
            exp_free=np.array([0.99]),
            rna_data_Km_endoRNase=np.array([1.0, 2.0]),
            mature_rna_data_Km_endoRNase=np.array([1.5]),
            darkATP=100.0,
            basal_cell_specs=basal_specs,
        )
        self.assertEqual(output.avg_cell_dry_mass_init, 1.0)
        np.testing.assert_array_equal(output.rna_expression_basal, [0.1, 0.2])

    def test_tf_condition_specs_output_defaults(self):
        """Test TfConditionSpecsOutput with defaults."""
        output = TfConditionSpecsOutput()
        self.assertEqual(output.rna_expression, {})
        self.assertEqual(output.condition_specs, {})

    def test_fit_condition_output(self):
        """Test FitConditionOutput creation."""
        output = FitConditionOutput(
            translation_supply_rate={'minimal': np.array([1.0, 2.0])},
            condition_specs_updates={'basal': {'key': 'value'}},
        )
        self.assertIn('minimal', output.translation_supply_rate)
        self.assertEqual(output.condition_specs_updates['basal']['key'], 'value')


class TestMapperFunctions(unittest.TestCase):
    """Tests for the mapper functions."""

    def test_apply_initialize_output(self):
        """Test apply_initialize_output produces correct StageResult."""
        mock_sim_data = MagicMock()
        output = InitializeOutput(sim_data=mock_sim_data)
        result = apply_initialize_output(output)

        self.assertIsInstance(result, StageResult)
        self.assertEqual(
            result.sim_data_update.attributes['_replace_entire_object'],
            mock_sim_data
        )

    def test_apply_input_adjustments_output_empty(self):
        """Test apply_input_adjustments_output with empty output."""
        output = InputAdjustmentsOutput()
        result = apply_input_adjustments_output(output)

        self.assertIsInstance(result, StageResult)
        # Empty output should produce empty updates
        self.assertEqual(len(result.sim_data_update.arrays), 0)

    def test_apply_input_adjustments_output_with_smoke_mode(self):
        """Test apply_input_adjustments_output with smoke mode updates."""
        output = InputAdjustmentsOutput(
            tf_to_active_inactive_conditions={'TF1': {'active': 'cond1'}},
            condition_active_tfs={'basal': ['TF1']},
        )
        result = apply_input_adjustments_output(output)

        self.assertEqual(
            result.sim_data_update.attributes['tf_to_active_inactive_conditions'],
            {'TF1': {'active': 'cond1'}}
        )
        self.assertEqual(
            result.sim_data_update.attributes['condition_active_tfs'],
            {'basal': ['TF1']}
        )

    def test_apply_input_adjustments_output_with_multipliers(self):
        """Test apply_input_adjustments_output with efficiency multipliers."""
        output = InputAdjustmentsOutput(
            translation_efficiencies_multipliers={0: 1.5, 5: 2.0},
        )
        result = apply_input_adjustments_output(output)

        # Should have array updates for each multiplier
        self.assertIn(
            'process.translation.translation_efficiencies_by_monomer:idx0',
            result.sim_data_update.arrays
        )
        self.assertIn(
            'process.translation.translation_efficiencies_by_monomer:idx5',
            result.sim_data_update.arrays
        )
        # Check the update values
        update0 = result.sim_data_update.arrays[
            'process.translation.translation_efficiencies_by_monomer:idx0'
        ]
        self.assertEqual(update0.op, 'multiply')
        self.assertEqual(update0.value, 1.5)
        self.assertEqual(update0.indices, 0)

    def test_apply_basal_specs_output(self):
        """Test apply_basal_specs_output produces correct StageResult."""
        basal_specs = CellSpecsData(
            concDict={'met': 1.0},
            expression=np.array([0.1]),
            doubling_time=60.0,
            synthProb=np.array([0.5]),
            fit_cistron_expression=np.array([0.3]),
            avgCellDryMassInit=1.0,
            fitAvgSolubleTargetMolMass=0.5,
            bulkContainer=np.array([100]),
        )
        output = BasalSpecsOutput(
            avg_cell_dry_mass_init=1.0,
            avg_cell_dry_mass=2.0,
            avg_cell_water_mass_init=0.7,
            fitAvgSolubleTargetMolMass=0.5,
            rna_expression_basal=np.array([0.1]),
            rna_synth_prob_basal=np.array([0.5]),
            fit_cistron_expression_basal=np.array([0.3]),
            exp_ppgpp=np.array([0.01]),
            exp_free=np.array([0.99]),
            rna_data_Km_endoRNase=np.array([1.0]),
            mature_rna_data_Km_endoRNase=np.array([1.5]),
            darkATP=100.0,
            basal_cell_specs=basal_specs,
        )
        result = apply_basal_specs_output(output)

        self.assertIsInstance(result, StageResult)
        # Check mass attributes
        self.assertEqual(
            result.sim_data_update.attributes['mass.avg_cell_dry_mass_init'],
            1.0
        )
        # Check expression dicts
        self.assertIn('basal', result.sim_data_update.dicts['process.transcription.rna_expression'])
        # Check Km array updates
        self.assertIn(
            'process.transcription.rna_data:Km_endoRNase',
            result.sim_data_update.arrays
        )
        # Check cell_specs updates
        self.assertIn('basal', result.cell_specs_update.conditions)

    def test_apply_promoter_binding_output(self):
        """Test apply_promoter_binding_output produces correct StageResult."""
        output = PromoterBindingOutput(
            pPromoterBound={'TF1': 0.5},
            rna_synth_prob={'basal': np.array([0.1, 0.2])},
            basal_r_vector=np.array([1, 2, 3]),
            basal_r_columns=['a', 'b', 'c'],
        )
        result = apply_promoter_binding_output(output)

        self.assertIsInstance(result, StageResult)
        self.assertEqual(result.sim_data_update.attributes['pPromoterBound'], {'TF1': 0.5})
        self.assertEqual(
            result.cell_specs_update.conditions['basal']['r_columns'],
            ['a', 'b', 'c']
        )

    def test_apply_adjust_promoters_output(self):
        """Test apply_adjust_promoters_output produces correct StageResult."""
        output = AdjustPromotersOutput(
            free_to_inactive_total=0.1,
            rnap_to_bound_prob_from_TFRNAP={'TF1': 0.5},
            rnap_to_bound_prob_from_basal={'TF1': 0.3},
        )
        result = apply_adjust_promoters_output(output)

        self.assertIsInstance(result, StageResult)
        self.assertEqual(
            result.sim_data_update.attributes['process.equilibrium.free_to_inactive_total'],
            0.1
        )

    def test_apply_set_conditions_output(self):
        """Test apply_set_conditions_output produces correct StageResult."""
        output = SetConditionsOutput(
            rnaSynthProbFraction={'minimal': {'mRna': 0.5}},
            rnapFractionActiveDict={'minimal': 0.8},
            rnaSynthProbRProtein={'minimal': np.array([0.1])},
            rnaSynthProbRnaPolymerase={'minimal': np.array([0.05])},
            rnaPolymeraseElongationRateDict={'minimal': 50.0},
            ribosomeElongationRateDict={'minimal': 20.0},
            ribosomeFractionActiveDict={'minimal': 0.9},
            expectedDryMassIncreaseDict={'minimal': 1.0},
            condition_specs_updates={'basal': {'avgCellDryMassInit': 1.5}},
        )
        result = apply_set_conditions_output(output)

        self.assertIsInstance(result, StageResult)
        self.assertEqual(
            result.sim_data_update.attributes['process.transcription.rnaSynthProbFraction'],
            {'minimal': {'mRna': 0.5}}
        )
        self.assertEqual(
            result.cell_specs_update.conditions['basal']['avgCellDryMassInit'],
            1.5
        )

    def test_apply_final_adjustments_output(self):
        """Test apply_final_adjustments_output produces correct StageResult."""
        output = FinalAdjustmentsOutput(
            attenuation_basal_prob=0.1,
            ppgpp_expression=np.array([0.5]),
            exp_ppgpp=np.array([0.01]),
            synth_prob_ppgpp=0.05,
            ppgpp_km=1.0,
            ppgpp_ki_synthetase=0.5,
            ppgpp_ki_hydrolase=0.3,
            aa_supply_scaling=1.0,
            aa_supply=np.array([1, 2, 3]),
            aa_export_kcat=0.1,
            aa_import_kis=0.2,
        )
        result = apply_final_adjustments_output(output)

        self.assertIsInstance(result, StageResult)
        self.assertEqual(
            result.sim_data_update.attributes['process.transcription.attenuation_basal_prob'],
            0.1
        )
        self.assertEqual(
            result.sim_data_update.attributes['process.metabolism.aa_supply_scaling'],
            1.0
        )


# Skip slow integration tests by default
# Run with: pytest -k "slow" --run-slow
import os
SKIP_SLOW = os.environ.get('RUN_SLOW_TESTS', '0') != '1'


@unittest.skipIf(SKIP_SLOW, "Slow integration test - set RUN_SLOW_TESTS=1 to run")
class TestPureModeEquivalence(unittest.TestCase):
    """
    Integration tests verifying pure_mode produces equivalent results to legacy mode.

    These tests are slow (~20+ minutes) because they run the actual ParCa pipeline.
    Run with: RUN_SLOW_TESTS=1 pytest reconstruction/tests/test_parca_updates.py -k "equivalence" -v
    """

    @classmethod
    def setUpClass(cls):
        """Load raw data once for all tests in this class."""
        from reconstruction.ecoli.knowledge_base_raw import KnowledgeBaseEcoli
        print("\nLoading knowledge base (this takes a few minutes)...")
        cls.raw_data = KnowledgeBaseEcoli(
            operons_on=True,
            remove_rrna_operons=False,
            remove_rrff=False,
            stable_rrna=False,
        )
        print("Knowledge base loaded.")

    def test_input_adjustments_equivalence(self):
        """
        Verify input_adjustments produces same results in pure_mode vs legacy mode.

        This test runs only the initialize and input_adjustments stages to verify
        the pure function implementation is equivalent to the legacy mutating version.
        """
        import copy
        from reconstruction.ecoli.fit_sim_data_1 import (
            initialize, input_adjustments, compute_input_adjustments, SimulationDataEcoli
        )
        from reconstruction.ecoli.parca_updates import apply_stage_result

        # Create two independent sim_data objects
        sim_data_legacy = SimulationDataEcoli()
        sim_data_pure = SimulationDataEcoli()
        cell_specs_legacy = {}
        cell_specs_pure = {}

        # Initialize both
        sim_data_legacy, cell_specs_legacy = initialize(
            sim_data_legacy, cell_specs_legacy,
            raw_data=self.raw_data,
            save_intermediates=False
        )
        sim_data_pure, cell_specs_pure = initialize(
            sim_data_pure, cell_specs_pure,
            raw_data=self.raw_data,
            save_intermediates=False
        )

        # Run input_adjustments with legacy mode (in-place mutation)
        sim_data_legacy, cell_specs_legacy = input_adjustments(
            sim_data_legacy, cell_specs_legacy,
            smoke=True,
            save_intermediates=False
        )

        # Run input_adjustments with pure mode (compute + apply externally)
        result = compute_input_adjustments(sim_data_pure, cell_specs_pure, smoke=True)
        sim_data_pure, cell_specs_pure = apply_stage_result(
            sim_data_pure, cell_specs_pure, result
        )

        # Compare key attributes modified by input_adjustments
        self._compare_sim_data(sim_data_legacy, sim_data_pure)

    def _compare_sim_data(self, sim_data_legacy, sim_data_pure):
        """Compare sim_data objects for equivalence."""
        from wholecell.utils import units

        def strip_units(arr):
            """Strip units from array if present."""
            if hasattr(arr, 'asNumber'):
                return arr.asNumber()
            # Handle object arrays containing unit values
            if arr.dtype == object and len(arr) > 0:
                if hasattr(arr[0], 'asNumber'):
                    return np.array([v.asNumber() for v in arr])
            return arr

        # Compare tf_to_active_inactive_conditions (smoke mode filtering)
        self.assertEqual(
            set(sim_data_legacy.tf_to_active_inactive_conditions.keys()),
            set(sim_data_pure.tf_to_active_inactive_conditions.keys()),
            "tf_to_active_inactive_conditions keys differ"
        )

        # Compare condition_active_tfs (smoke mode filtering)
        self.assertEqual(
            set(sim_data_legacy.condition_active_tfs.keys()),
            set(sim_data_pure.condition_active_tfs.keys()),
            "condition_active_tfs keys differ"
        )

        # Compare translation efficiencies
        np.testing.assert_array_almost_equal(
            strip_units(sim_data_legacy.process.translation.translation_efficiencies_by_monomer),
            strip_units(sim_data_pure.process.translation.translation_efficiencies_by_monomer),
            decimal=10,
            err_msg="translation_efficiencies_by_monomer differ"
        )

        # Compare RNA expression (basal)
        np.testing.assert_array_almost_equal(
            strip_units(sim_data_legacy.process.transcription.rna_expression['basal']),
            strip_units(sim_data_pure.process.transcription.rna_expression['basal']),
            decimal=10,
            err_msg="rna_expression['basal'] differ"
        )

        # Compare RNA degradation rates
        np.testing.assert_array_almost_equal(
            strip_units(sim_data_legacy.process.transcription.rna_data['deg_rate']),
            strip_units(sim_data_pure.process.transcription.rna_data['deg_rate']),
            decimal=10,
            err_msg="rna_data['deg_rate'] differ"
        )

        # Compare protein degradation rates
        np.testing.assert_array_almost_equal(
            strip_units(sim_data_legacy.process.translation.monomer_data['deg_rate']),
            strip_units(sim_data_pure.process.translation.monomer_data['deg_rate']),
            decimal=10,
            err_msg="monomer_data['deg_rate'] differ"
        )

        # Compare cistron degradation rates
        np.testing.assert_array_almost_equal(
            strip_units(sim_data_legacy.process.transcription.cistron_data['deg_rate']),
            strip_units(sim_data_pure.process.transcription.cistron_data['deg_rate']),
            decimal=10,
            err_msg="cistron_data['deg_rate'] differ"
        )

        print("All comparisons passed - pure_mode produces equivalent results!")

    def test_basal_specs_equivalence(self):
        """
        Verify stages through basal_specs produce same results in pure_mode vs legacy mode.

        This test runs initialize -> input_adjustments -> basal_specs which takes
        about 1-2 minutes, without running the expensive fit_condition stage.
        """
        import tempfile
        from reconstruction.ecoli.fit_sim_data_1 import (
            initialize, input_adjustments, basal_specs,
            compute_input_adjustments, compute_basal_specs,
            SimulationDataEcoli
        )
        from reconstruction.ecoli.parca_updates import apply_stage_result

        # Create two independent sim_data objects
        sim_data_legacy = SimulationDataEcoli()
        sim_data_pure = SimulationDataEcoli()
        cell_specs_legacy = {}
        cell_specs_pure = {}

        # Create a temporary cache directory for Km caching
        with tempfile.TemporaryDirectory() as cache_dir:
            # Initialize both
            sim_data_legacy, cell_specs_legacy = initialize(
                sim_data_legacy, cell_specs_legacy,
                raw_data=self.raw_data,
                save_intermediates=False
            )
            sim_data_pure, cell_specs_pure = initialize(
                sim_data_pure, cell_specs_pure,
                raw_data=self.raw_data,
                save_intermediates=False
            )

            # Run input_adjustments - LEGACY (in-place)
            sim_data_legacy, cell_specs_legacy = input_adjustments(
                sim_data_legacy, cell_specs_legacy,
                smoke=True,
                save_intermediates=False
            )
            # Run input_adjustments - PURE (compute + apply externally)
            result = compute_input_adjustments(sim_data_pure, cell_specs_pure, smoke=True)
            sim_data_pure, cell_specs_pure = apply_stage_result(
                sim_data_pure, cell_specs_pure, result
            )

            # Run basal_specs - LEGACY (in-place)
            print("\n=== Running basal_specs in LEGACY mode ===")
            sim_data_legacy, cell_specs_legacy = basal_specs(
                sim_data_legacy, cell_specs_legacy,
                save_intermediates=False,
                cache_dir=cache_dir,
            )

            # Run basal_specs - PURE (compute + apply externally)
            print("\n=== Running basal_specs in PURE mode ===")
            result = compute_basal_specs(
                sim_data_pure, cell_specs_pure,
                cache_dir=cache_dir,
            )
            sim_data_pure, cell_specs_pure = apply_stage_result(
                sim_data_pure, cell_specs_pure, result
            )

        # Compare basal specs
        self._compare_basal_specs(sim_data_legacy, sim_data_pure,
                                   cell_specs_legacy, cell_specs_pure)

    def _compare_basal_specs(self, sim_data_legacy, sim_data_pure,
                              cell_specs_legacy, cell_specs_pure):
        """Compare results after basal_specs stage."""
        from wholecell.utils import units

        def strip_units(val):
            """Strip units from value if present."""
            if val is None:
                return None
            if hasattr(val, 'asNumber'):
                return val.asNumber()
            if hasattr(val, 'dtype') and val.dtype == object:
                if len(val) > 0 and hasattr(val[0], 'asNumber'):
                    return np.array([v.asNumber() for v in val])
            return val

        errors = []

        # Compare cell_specs structure
        try:
            self.assertEqual(set(cell_specs_legacy.keys()), set(cell_specs_pure.keys()),
                           "cell_specs keys differ")
            print("  cell_specs keys: OK")
        except AssertionError as e:
            errors.append(f"cell_specs keys: {e}")

        # Compare basal condition if present
        if 'basal' in cell_specs_legacy and 'basal' in cell_specs_pure:
            try:
                basal_legacy = cell_specs_legacy['basal']
                basal_pure = cell_specs_pure['basal']
                for key in basal_legacy.keys():
                    if key in basal_pure:
                        v1, v2 = basal_legacy[key], basal_pure[key]
                        if isinstance(v1, np.ndarray):
                            # Handle structured arrays with mixed types
                            if v1.dtype.names is not None:
                                # Compare each field of structured array separately
                                for field in v1.dtype.names:
                                    if np.issubdtype(v1[field].dtype, np.number):
                                        np.testing.assert_array_almost_equal(
                                            strip_units(v1[field]), strip_units(v2[field]),
                                            decimal=8,
                                            err_msg=f"cell_specs['basal'][{key}][{field}] differ"
                                        )
                                    else:
                                        # String comparison
                                        np.testing.assert_array_equal(
                                            v1[field], v2[field],
                                            err_msg=f"cell_specs['basal'][{key}][{field}] differ"
                                        )
                            else:
                                np.testing.assert_array_almost_equal(
                                    strip_units(v1), strip_units(v2), decimal=8,
                                    err_msg=f"cell_specs['basal'][{key}] differ"
                                )
                print("  cell_specs['basal']: OK")
            except AssertionError as e:
                errors.append(f"cell_specs['basal']: {e}")

        # Compare ppGpp-related expression
        try:
            np.testing.assert_array_almost_equal(
                strip_units(sim_data_legacy.process.transcription.exp_ppgpp),
                strip_units(sim_data_pure.process.transcription.exp_ppgpp),
                decimal=8
            )
            np.testing.assert_array_almost_equal(
                strip_units(sim_data_legacy.process.transcription.exp_free),
                strip_units(sim_data_pure.process.transcription.exp_free),
                decimal=8
            )
            print("  exp_ppgpp, exp_free: OK")
        except (AssertionError, AttributeError) as e:
            errors.append(f"ppgpp expression (exp_ppgpp/exp_free): {e}")

        if errors:
            print("\n=== ERRORS ===")
            for err in errors:
                print(f"  {err}")
            self.fail(f"Found {len(errors)} differences between pure and legacy mode in basal_specs")
        else:
            print("\n=== basal_specs: All comparisons passed! ===")

    def test_full_pipeline_equivalence(self):
        """
        Verify the full ParCa pipeline produces same results in pure_mode vs legacy mode.

        This test runs all stages with smoke=True to verify the complete pipeline.
        Takes ~20-30 minutes.
        """
        import tempfile
        from reconstruction.ecoli.fit_sim_data_1 import fitSimData_1

        # Create a temporary cache directory
        with tempfile.TemporaryDirectory() as cache_dir:
            print("\n=== Running full pipeline in LEGACY mode ===")
            sim_data_legacy = fitSimData_1(
                self.raw_data,
                smoke=True,
                pure_mode=False,
                save_intermediates=False,
                cache_dir=cache_dir,
            )

            print("\n=== Running full pipeline in PURE mode ===")
            sim_data_pure = fitSimData_1(
                self.raw_data,
                smoke=True,
                pure_mode=True,
                save_intermediates=False,
                cache_dir=cache_dir,
            )

        print("\n=== Comparing results ===")
        self._compare_full_sim_data(sim_data_legacy, sim_data_pure)

    def _compare_full_sim_data(self, sim_data_legacy, sim_data_pure):
        """Compare sim_data objects after full pipeline run."""
        from wholecell.utils import units

        def strip_units(arr):
            """Strip units from array if present."""
            if arr is None:
                return None
            if hasattr(arr, 'asNumber'):
                return arr.asNumber()
            if hasattr(arr, 'dtype') and arr.dtype == object and len(arr) > 0:
                if hasattr(arr[0], 'asNumber'):
                    return np.array([v.asNumber() for v in arr])
            return arr

        def compare_dicts(d1, d2, name):
            """Compare two dictionaries."""
            self.assertEqual(set(d1.keys()), set(d2.keys()), f"{name} keys differ")
            for key in d1.keys():
                v1, v2 = d1[key], d2[key]
                if isinstance(v1, np.ndarray):
                    np.testing.assert_array_almost_equal(
                        strip_units(v1), strip_units(v2), decimal=8,
                        err_msg=f"{name}[{key}] differ"
                    )
                elif isinstance(v1, dict):
                    compare_dicts(v1, v2, f"{name}[{key}]")
                else:
                    # For scalars with units
                    v1_val = strip_units(v1) if hasattr(v1, 'asNumber') else v1
                    v2_val = strip_units(v2) if hasattr(v2, 'asNumber') else v2
                    if isinstance(v1_val, (int, float)):
                        self.assertAlmostEqual(v1_val, v2_val, places=8,
                            msg=f"{name}[{key}] differ: {v1_val} vs {v2_val}")

        errors = []

        # Test translation efficiencies
        try:
            np.testing.assert_array_almost_equal(
                strip_units(sim_data_legacy.process.translation.translation_efficiencies_by_monomer),
                strip_units(sim_data_pure.process.translation.translation_efficiencies_by_monomer),
                decimal=8
            )
            print("  translation_efficiencies_by_monomer: OK")
        except AssertionError as e:
            errors.append(f"translation_efficiencies_by_monomer: {e}")

        # Test RNA expression
        try:
            for condition in sim_data_legacy.process.transcription.rna_expression.keys():
                np.testing.assert_array_almost_equal(
                    strip_units(sim_data_legacy.process.transcription.rna_expression[condition]),
                    strip_units(sim_data_pure.process.transcription.rna_expression[condition]),
                    decimal=8
                )
            print("  rna_expression: OK")
        except AssertionError as e:
            errors.append(f"rna_expression: {e}")

        # Test RNA synth prob
        try:
            for condition in sim_data_legacy.process.transcription.rna_synth_prob.keys():
                np.testing.assert_array_almost_equal(
                    strip_units(sim_data_legacy.process.transcription.rna_synth_prob[condition]),
                    strip_units(sim_data_pure.process.transcription.rna_synth_prob[condition]),
                    decimal=8
                )
            print("  rna_synth_prob: OK")
        except AssertionError as e:
            errors.append(f"rna_synth_prob: {e}")

        # Test translation supply rate
        try:
            compare_dicts(
                sim_data_legacy.translation_supply_rate,
                sim_data_pure.translation_supply_rate,
                "translation_supply_rate"
            )
            print("  translation_supply_rate: OK")
        except AssertionError as e:
            errors.append(f"translation_supply_rate: {e}")

        # Test expectedDryMassIncreaseDict
        try:
            compare_dicts(
                sim_data_legacy.expectedDryMassIncreaseDict,
                sim_data_pure.expectedDryMassIncreaseDict,
                "expectedDryMassIncreaseDict"
            )
            print("  expectedDryMassIncreaseDict: OK")
        except AssertionError as e:
            errors.append(f"expectedDryMassIncreaseDict: {e}")

        # Test rnaSynthProbFraction
        try:
            compare_dicts(
                sim_data_legacy.process.transcription.rnaSynthProbFraction,
                sim_data_pure.process.transcription.rnaSynthProbFraction,
                "rnaSynthProbFraction"
            )
            print("  rnaSynthProbFraction: OK")
        except AssertionError as e:
            errors.append(f"rnaSynthProbFraction: {e}")

        # Test mass values
        try:
            self.assertAlmostEqual(
                strip_units(sim_data_legacy.mass.avg_cell_dry_mass_init),
                strip_units(sim_data_pure.mass.avg_cell_dry_mass_init),
                places=8
            )
            print("  mass.avg_cell_dry_mass_init: OK")
        except AssertionError as e:
            errors.append(f"mass.avg_cell_dry_mass_init: {e}")

        if errors:
            print("\n=== ERRORS ===")
            for err in errors:
                print(f"  {err}")
            self.fail(f"Found {len(errors)} differences between pure and legacy mode")
        else:
            print("\n=== All comparisons passed! Pure mode produces equivalent results. ===")


if __name__ == '__main__':
    unittest.main()
