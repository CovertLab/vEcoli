from vivarium.core.engine import Engine, pf


class Scan:
    def __init__(self, parameter_sets, simulator_class, total_time, metrics=(), global_time_precision=None):

        self.parameter_sets = parameter_sets
        self.simulator_class = simulator_class
        self.total_time = total_time
        self.metrics = metrics
        self.global_time_precision = global_time_precision

    def run_simulation(self, parameters):
        simulator = self.simulator_class(parameters['parameters']).generate()
        engine = Engine(
            processes=simulator["processes"],
            topology=simulator["topology"],
            initial_state=parameters["states"],
        )
        engine.update(self.total_time, global_time_precision=self.global_time_precision)
        return engine.emitter.get_data()

    def run_scan(self):
        results = {}
        for id, parameter_set in self.parameter_sets.items():
            data = self.run_simulation(parameter_set)
            metrics = {name: metric(data, parameter_set) for name, metric in self.metrics.items()}
            results[id] = {"data": data, "metrics": metrics}
        return results