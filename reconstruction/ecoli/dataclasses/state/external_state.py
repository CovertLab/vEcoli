"""
Simulation data for external state

This base class includes all data associated with states external to the cells.
Initializes the environment using conditions and time series from raw_data.

        - saved_timelines: a dictionary of all timelines.
        - current_timeline_id: a string specifying the timelines
                used for the current simulation.
        - current_media: a dictionary of molecules (keys) and
                their concentrations (values).
        - saved_media: a dictionary of all media, each entry
                itself a dictionary molecules (keys) and their concentrations (values).

"""

import numpy as np

from wholecell.utils import units
from wholecell.utils.make_media import Media

from typing import Any

# threshold (units.mmol / units.L) separates concentrations that are import constrained with
# max flux = 0 from unconstrained molecules.
IMPORT_CONSTRAINT_THRESHOLD = 1e-5


OXYGEN_SCALING_MODE_DEFAULT = "binary"

# Needed for oxygen import variant:
# Half-saturation O2 concentration (mmol/L) and Hill coefficient for the
# uptake interpolation used when OXYGEN_SCALING_MODE == "continuous".
OXYGEN_UPTAKE_K_HALF_DEFAULT = 1e-2
OXYGEN_UPTAKE_HILL_N = 2.0

# Finite ceiling (mmol/g DCW/h) on oxygen's OWN import flux when O2 is
# abundant, used only when OXYGEN_SCALING_MODE == "continuous". Under
# "binary" mode oxygen import is literally unconstrained (infinite) whenever
# concentration is above IMPORT_CONSTRAINT_THRESHOLD; "continuous" mode
# replaces that all-or-nothing treatment with a smooth Michaelis-Menten/Hill
# curve (see _continuous_oxygen_uptake_bound) from 0 up to this ceiling.
#
# Measured directly from listeners__fba_results__estimated_exchange_dmdt__
# OXYGEN-MOLECULE in a real oxygen_depletion run (test_oxygen_depletion_
# 20260813-144130): actual aerobic O2 demand is only ~13-24 mmol/gDCW/h. Still not derived from
# literature O2-uptake-kinetics data -- same placeholder status as
# OXYGEN_UPTAKE_K_HALF_DEFAULT/OXYGEN_UPTAKE_HILL_N, just now grounded in
# this model's own measured demand rather than an arbitrary safety margin.
OXYGEN_UPTAKE_V_MAX_DEFAULT = 30.0


class ExternalState(object):
    """External State"""

    def __init__(self, raw_data, sim_data):
        # make media object
        self.make_media = Media(raw_data)

        self.carbon_sources = sim_data.molecule_groups.carbon_sources
        self._initialize_environment(raw_data)
        self.all_external_exchange_molecules = (
            self._get_all_external_exchange_molecules(raw_data)
        )
        self.secretion_exchange_molecules = self._get_secretion_exchange_molecules(
            raw_data
        )

    def _get_all_external_exchange_molecules(self, raw_data):
        """
        Returns:
                list[str]: all external exchange molecules
        """
        externalExchangeData = []
        # initiate all molecules with 0 concentrations
        for row in raw_data.condition.environment_molecules:
            externalExchangeData.append(
                row["molecule id"] + row["exchange molecule location"]
            )

        return externalExchangeData

    def _get_secretion_exchange_molecules(self, raw_data):
        """
        Returns:
                set[str]: all secretion exchange molecules
        """
        secretionExchangeMolecules = []
        for secretion in raw_data.secretions:
            if secretion["lower bound"] and secretion["upper bound"]:
                # "non-growth associated maintenance", not included in our metabolic model
                continue
            else:
                secretionExchangeMolecules.append(secretion["molecule id"])

        return set(secretionExchangeMolecules)

    def _initialize_environment(self, raw_data):
        self.import_constraint_threshold = IMPORT_CONSTRAINT_THRESHOLD
        # Both may be overwritten by the oxygen_depletion variant; see
        # OXYGEN_SCALING_MODE_DEFAULT/OXYGEN_UPTAKE_K_HALF_DEFAULT above.
        self.oxygen_scaling_mode = OXYGEN_SCALING_MODE_DEFAULT
        self.oxygen_uptake_k_half = OXYGEN_UPTAKE_K_HALF_DEFAULT
        self.oxygen_uptake_v_max = OXYGEN_UPTAKE_V_MAX_DEFAULT

        # create a dictionary with all saved timelines
        self.saved_timelines = {}
        for row in raw_data.condition.timelines_def:
            timeline_id = row["timeline"]
            timeline_str = row["events"]
            new_timeline = self.make_media.make_timeline(timeline_str)
            self.saved_timelines[timeline_id] = new_timeline

        # set default current_timeline_id to None, this can be overwritten by the timelines variant
        self.current_timeline_id = None

        # make a dictionary with all media conditions specified by media_recipes
        self.saved_media = self.make_media.make_saved_media()

        # make mapping from external molecule to exchange molecule
        self.env_to_exchange_map = {
            mol["molecule id"]: mol["molecule id"] + mol["exchange molecule location"]
            for mol_index, mol in enumerate(raw_data.condition.environment_molecules)
        }
        self.exchange_to_env_map = {v: k for k, v in self.env_to_exchange_map.items()}

        # make dict with exchange molecules for all saved environments, using env_to_exchange_map
        self.exchange_dict = {}
        for media, concentrations in self.saved_media.items():
            self.exchange_dict[media] = {
                self.env_to_exchange_map[mol]: conc
                for mol, conc in concentrations.items()
            }

    def exchange_data_from_concentrations(
        self, molecules: dict[str, float]
    ) -> dict[str, Any]:
        """
        Update importExchangeMolecules for FBA based on current nutrient concentrations.
        This provides a simple type of transport to accommodate changing nutrient
        concentrations in the environment. Transport is modeled as a binary switch:
        When there is a high concentrations of environment nutrients, transporters
        are unconstrained and nutrients are transported as needed by metabolism.
        When concentrations fall below the threshold, that nutrient's transport
        is constrained to max flux of 0.

        Args:
                molecules: external molecules (no location tag) with external concentration,
                        concentration can be inf

        Returns dict with the following keys:
                externalExchangeMolecules (set[str]): all exchange molecules (with
                        location tag), includes both import and secretion exchanged molecules
                importExchangeMolecules (set[str]): molecules (with location tag) that
                        can be imported from the environment into the cell
                importConstrainedExchangeMolecules (dict[str, float with mol/mass/time units]):
                        constrained molecules (with location tag) with upper bound flux constraints
                importUnconstrainedExchangeMolecules (set[str]): exchange molecules
                        (with location tag) that do not have an upper bound on their flux
                secretionExchangeMolecules (set[str]): molecules (with location tag)
                        that can be secreted by the cell into the environment
        """

        externalExchangeMolecules = set()
        importExchangeMolecules = set()
        secretionExchangeMolecules = self.secretion_exchange_molecules

        oxygen_id = "OXYGEN-MOLECULE[p]"

        exchange_molecules = {
            self.env_to_exchange_map[mol]: conc for mol, conc in molecules.items()
        }

        # Unconstrained uptake if greater than import threshold
        importUnconstrainedExchangeMolecules = {
            molecule_id
            for molecule_id, concentration in exchange_molecules.items()
            if concentration >= self.import_constraint_threshold
        }
        importExchangeMolecules.update(importUnconstrainedExchangeMolecules)
        externalExchangeMolecules.update(importUnconstrainedExchangeMolecules)

        # TODO: functionalize limits based on concentrations of transporters and environment
        # Limit carbon uptake if present depending on the presence of oxygen
        importConstrainedExchangeMolecules = {}
        # getattr fallback: sim_data pickled before this attribute existed
        # (unpickling doesn't re-run __init__) would otherwise crash here.
        oxygen_scaling_mode = getattr(
            self, "oxygen_scaling_mode", OXYGEN_SCALING_MODE_DEFAULT
        )
        if oxygen_scaling_mode == "continuous":
            o2_concentration = exchange_molecules.get(oxygen_id)
            carbon_bound = self._continuous_oxygen_carbon_bound(o2_concentration)
            for carbon_source_id in self.carbon_sources:
                if carbon_source_id in importUnconstrainedExchangeMolecules:
                    importConstrainedExchangeMolecules[carbon_source_id] = (
                        carbon_bound * (units.mmol / units.g / units.h)
                    )
                    importUnconstrainedExchangeMolecules.remove(carbon_source_id)
            # Oxygen's OWN import: replace the binary infinite/absent
            # treatment with a smooth, finite flux bound (see
            # _continuous_oxygen_uptake_bound) -- scoped to oxygen only,
            # everything else (carbon sources above, all other molecules)
            # keeps its existing unconstrained/constrained treatment.
            if oxygen_id in importUnconstrainedExchangeMolecules:
                oxygen_bound = self._continuous_oxygen_uptake_bound(o2_concentration)
                importConstrainedExchangeMolecules[oxygen_id] = oxygen_bound * (
                    units.mmol / units.g / units.h
                )
                importUnconstrainedExchangeMolecules.remove(oxygen_id)
        else:
            for carbon_source_id in self.carbon_sources:
                if carbon_source_id in importUnconstrainedExchangeMolecules:
                    if oxygen_id in importUnconstrainedExchangeMolecules:
                        importConstrainedExchangeMolecules[carbon_source_id] = 20.0 * (
                            units.mmol / units.g / units.h
                        )
                    else:
                        importConstrainedExchangeMolecules[carbon_source_id] = 100.0 * (
                            units.mmol / units.g / units.h
                        )
                    importUnconstrainedExchangeMolecules.remove(carbon_source_id)

        externalExchangeMolecules.update(secretionExchangeMolecules)

        return {
            "externalExchangeMolecules": externalExchangeMolecules,
            "importExchangeMolecules": importExchangeMolecules,
            "importConstrainedExchangeMolecules": importConstrainedExchangeMolecules,
            "importUnconstrainedExchangeMolecules": importUnconstrainedExchangeMolecules,
            "secretionExchangeMolecules": secretionExchangeMolecules,
        }

    @staticmethod
    def _o2_magnitude_mmol_per_l(o2_concentration):
        """
        Extracts a plain O2 concentration magnitude in mmol/L, for use by
        the continuous-scaling helpers below.

        Args:
                o2_concentration: external O2 concentration, as either a
                        plain float/None or a units-wrapped quantity --
                        environment concentrations may arrive as either
                        wholecell.utils.units (Unum, exposes .asNumber()) or
                        vivarium.library.units (pint, exposes .to()/
                        .magnitude, no .asNumber()) depending on the call
                        site, so handle both rather than assuming one.

        Returns:
                float or None (if ``o2_concentration`` is None); may be inf.
        """
        if o2_concentration is None:
            return None
        if hasattr(o2_concentration, "asNumber"):
            return o2_concentration.asNumber(units.mmol / units.L)
        if hasattr(o2_concentration, "to"):
            from vivarium.library.units import units as vivarium_units

            return o2_concentration.to(vivarium_units.mmol / vivarium_units.L).magnitude
        return o2_concentration

    def _oxygen_hill_fraction(self, o2_magnitude):
        """
        Shared Hill-curve fraction (0 -> 1 as O2 -> abundant) used by both
        continuous-scaling bounds below, so their K_half/Hill-N shape stays
        consistent with each other.
        """
        # getattr fallback: sim_data pickled before this attribute existed
        # (unpickling doesn't re-run __init__) would otherwise crash here.
        oxygen_uptake_k_half = getattr(
            self, "oxygen_uptake_k_half", OXYGEN_UPTAKE_K_HALF_DEFAULT
        )
        return o2_magnitude**OXYGEN_UPTAKE_HILL_N / (
            o2_magnitude**OXYGEN_UPTAKE_HILL_N
            + oxygen_uptake_k_half**OXYGEN_UPTAKE_HILL_N
        )

    def _continuous_oxygen_carbon_bound(self, o2_concentration):
        """
        Smoothly interpolates the carbon-uptake FBA bound (mmol/g/h) between
        the aerobic (20) and anaerobic (100) literal bounds as a Hill
        function of O2 concentration, for OXYGEN_SCALING_MODE == "continuous".

        Args:
                o2_concentration: external O2 concentration (mmol/L), as
                        either a plain float or a units-wrapped quantity;
                        may be missing (treated as 0) or infinite (treated
                        as fully aerobic).
        """
        o2_magnitude = self._o2_magnitude_mmol_per_l(o2_concentration)
        if o2_magnitude is None:
            return 100.0
        if np.isinf(o2_magnitude):
            return 20.0

        hill = self._oxygen_hill_fraction(o2_magnitude)
        # hill -> 1 as O2 is abundant (aerobic bound 20); hill -> 0 as O2 -> 0
        # (anaerobic bound 100).
        return 100.0 - hill * (100.0 - 20.0)

    def _continuous_oxygen_uptake_bound(self, o2_concentration):
        """
        Smooth Michaelis-Menten/Hill-shaped bound (mmol/g/h) on oxygen's OWN
        import flux, saturating at a finite ceiling (oxygen_uptake_v_max) as
        O2 becomes abundant and ->0 as O2->0 -- replacing "binary" mode's
        all-or-nothing infinite/absent treatment of oxygen's own
        availability, for OXYGEN_SCALING_MODE == "continuous". Distinct from
        _continuous_oxygen_carbon_bound above, which only ever adjusted the
        downstream carbon-source bound, not oxygen's own import.

        Args:
                o2_concentration: external O2 concentration (mmol/L), as
                        either a plain float or a units-wrapped quantity;
                        may be missing (treated as 0 -> 0 bound) or infinite
                        (treated as fully aerobic -> V_max).
        """
        # getattr fallback: sim_data pickled before this attribute existed
        # (unpickling doesn't re-run __init__) would otherwise crash here.
        oxygen_uptake_v_max = getattr(
            self, "oxygen_uptake_v_max", OXYGEN_UPTAKE_V_MAX_DEFAULT
        )
        o2_magnitude = self._o2_magnitude_mmol_per_l(o2_concentration)
        if o2_magnitude is None:
            return 0.0
        if np.isinf(o2_magnitude):
            return oxygen_uptake_v_max

        hill = self._oxygen_hill_fraction(o2_magnitude)
        # hill -> 1 as O2 is abundant (bound -> V_max); hill -> 0 as O2 -> 0
        # (bound -> 0).
        return hill * oxygen_uptake_v_max

    def exchange_data_from_media(self, media_label):
        """
        Returns:
                dict: exchange_data for a media_label saved in exchange_data_dict.
        """

        concentrations = self.saved_media[media_label]
        return self.exchange_data_from_concentrations(concentrations)

    def get_import_constraints(self, unconstrained, constrained, units):
        """
        Returns:
                unconstrained_molecules (list[bool]): the indices of all
                        importUnconstrainedExchangeMolecules in
                        self.all_external_exchange_molecules are true, the rest as false
                constrained_molecules (list[bool]): the indices of all
                        importConstrainedExchangeMolecules in
                        self.all_external_exchange_molecules are true, the rest as false
                constraints (list[float]): uptake constraints for each molecule
                        that is constrained, nan for no constraint
        """

        # molecules from all_external_exchange_molecules set to 'true' if they are current importExchangeMolecules.
        unconstrained_molecules = [
            molecule_id in unconstrained
            for molecule_id in self.all_external_exchange_molecules
        ]

        # molecules from all_external_exchange_molecules set to 'true' if they are current importConstrainedExchangeMolecules.
        constrained_molecules = [
            molecule_id in constrained
            for molecule_id in self.all_external_exchange_molecules
        ]

        constraints = [
            constrained.get(molecule_id, np.nan * units).asNumber(units)
            for molecule_id in self.all_external_exchange_molecules
        ]

        return unconstrained_molecules, constrained_molecules, constraints
