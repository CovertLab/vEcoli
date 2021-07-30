from ecoli.experiments.ecoli_master_sim import EcoliSim, CONFIG_DIR_PATH


def testDefault():
    sim = EcoliSim.from_file()
    sim.total_time = 2
    sim.run()


def testAddProcess():
    sim = EcoliSim.from_file(CONFIG_DIR_PATH + 'test_configs/test_add_process.json')
    data = sim.run()

    assert 'clock' in sim.ecoli.processes.keys()
    assert 'global_time' in data.keys()


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
    
    assert "ecoli-mass-listener" in sim.ecoli.processes.keys()
    assert "ecoli-mass-listener" in sim.ecoli.topology.keys()
    assert "dnaMass" in data['listeners']['mass'].keys()
    assert "ecoli-mass" not in sim.ecoli.processes.keys()
    assert "ecoli-mass" not in sim.ecoli.topology.keys()


def main():
    testDefault()
    testAddProcess()
    testExcludeProcess()
    testSwapProcess()


if __name__=="__main__":
    main()