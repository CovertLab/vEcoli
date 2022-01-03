from matplotlib import pyplot as plt
import numpy as np

from vivarium.core.process import Process


class CellWall(Process):

    name = "cell_wall"
    defaults = {
        'cracked': False
    }

    def ports_schema(self):
        schema = {
        }

        return schema

    def initial_state(self, config=None):
        config = config or {}
        parameters = self.parameters
        parameters.update(config)

        initial_state = {
        }
        initial_state.update(parameters['initial_state'])
        return initial_state

    def next_update(self, timestep, states):
        update = {
        }
        return update


def main():
    cellwall = CellWall()


if __name__ == '__main__':
    main()
