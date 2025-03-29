from vivarium.core.process import Step


class DivisionDetector(Step):
    """Calculates division threshold for inner simulation in EngineProcess.
    Upon reaching threshold, sets a flag through the `division_trigger` port
    that can be detected via a tunnel and used to initiate division.

    By default, we forgo the dry mass threshold in favor of a boolean
    threshold set by the MarkDPeriod Step in ecoli.processes.cell_division.
    Users can revert to the mass threshold by setting d_period to False in
    their config json.
    """

    name = "division-detector"

    defaults = {
        "division_threshold": None,
        "division_variable": None,
        "chromosome_path": None,
        "dry_mass_inc_dict": None,
        "division_mass_multiplier": 1,
    }

    def __init__(self, config):
        super().__init__(config)
        self.division_threshold = self.parameters["division_threshold"]
        self.dry_mass_inc_dict = self.parameters["dry_mass_inc_dict"]
        self.division_mass_multiplier = self.parameters["division_mass_multiplier"]

    def ports_schema(self):
        return {
            "division_variable": {},
            "division_trigger": {
                "_default": False,
                "_updater": "set",
                "_divider": {"divider": "set_value", "config": {"value": False}},
            },
            "full_chromosomes": {},
            "media_id": {},
            "division_threshold": {
                "_default": self.parameters["division_threshold"],
                "_updater": "set",
                "_divider": {
                    "divider": "set_value",
                    "config": {"value": self.parameters["division_threshold"]},
                },
            },
        }

    def next_update(self, timestep, states):
        update = {}
        division_threshold = states["division_threshold"]
        if division_threshold == "mass_distribution":
            mass_inc = self.dry_mass_inc_dict[states["media_id"]]
            division_threshold = (
                states["division_variable"]
                + mass_inc.asNumber() * self.division_mass_multiplier
            )
            update["division_threshold"] = division_threshold
        if (states["division_variable"] >= division_threshold) and (
            states["full_chromosomes"]["_entryState"].sum() >= 2
        ):
            update["division_trigger"] = True
        return update
