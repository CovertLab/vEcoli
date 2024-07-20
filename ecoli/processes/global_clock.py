from vivarium.core.process import Process


class GlobalClock(Process):
    """
    Track global time for Steps that do not rely on vivarium-core's built-in
    time stepping (see :ref:`timesteps`).
    """

    name = "global_clock"

    def ports_schema(self):
        return {
            "global_time": {"_default": 0.0, "_updater": "accumulate"},
            "next_update_time": {"*": {}},
        }

    def calculate_timestep(self, states):
        """
        Subtract global time from next update times for each manually time-stepped
        processes to calculate time until a process updates. Use that time as the
        time step for this process so vivarium-core's internal simulation clock
        advances by the same amount of time and processes that do not rely on
        this manual time stepping stay in sync with the ones that do.
        """
        return min(
            next_update_time - states["global_time"]
            for next_update_time in states["next_update_time"].values()
        )

    def next_update(self, timestep, states):
        """
        The timestep that we increment global_time by is the same minimum time step
        that we calculated in calculate_timestep. This guarantees that we never
        accidentally skip over a process update time.
        """
        return {"global_time": timestep}
