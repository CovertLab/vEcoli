from vivarium.core.engine import Engine, pf


class Scan:
    def __init__(self, parameter_sets, simulator_class, total_time, metrics=()):

        self.parameter_sets = parameter_sets
        self.simulator_class = simulator_class
        self.total_time = total_time
        self.metrics = metrics

    def run_simulation(self, parameters):
        simulator = self.simulator_class(parameters).generate(path=('agents', '1'))
        engine = Engine(
            processes=simulator["processes"],
            topology=simulator["topology"],
            initial_state=parameters["states"],
        )
        engine.update(self.total_time)
        return engine.emitter.get_data()

    def run_scan(self):
        results = {}
        for id, parameter_set in self.parameter_sets.items():
            data = self.run_simulation(parameter_set)
            metrics = {name: metric(data) for name, metric in self.metrics.items()}
            results[id] = {"data": data, "metrics": metrics}
        return results