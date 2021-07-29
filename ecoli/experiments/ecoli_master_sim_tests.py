from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


def testDefault():
    sim = EcoliSim.from_file()
    sim.total_time = 2
    sim.run()


def testAddProcess():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'test_configs/test_add_process.json')
    data = sim.run()


def testExcludeProcess():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'test_configs/test_exclude_process.json')
    sim.run()
    assert "ecoli-polypeptide-elongation" not in sim.ecoli.processes.keys()
    assert "ecoli-polypeptide-elongation" not in sim.ecoli.topology.keys()
    assert "ecoli-two-component-system" not in sim.ecoli.processes.keys()
    assert "ecoli-two-component-system" not in sim.ecoli.topology.keys()


def testSwapProcess():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'test_configs/test_swap_process.json')
    data = sim.run()


def main():
    testDefault()
    #testAddProcess()  # TODO: Known not to work (config issue)
    testExcludeProcess()
    # testSwapProcess()  # TODO: Known not to work (mass listener process not on this branch)


if __name__=="__main__":
    main()