import os

from crowddynamics.simulation.logic import SaveSimulationData
from tqdm import tqdm, trange

from finlandia_talo import FinlandiaTalo2ndFloor


def run(simulation, max_iterations, base_directory):
    save_condition = lambda simu: (simu.data['iterations'] + 1) % 1000 == 0

    node = SaveSimulationData(
        simulation,
        base_directory=os.path.join('.', base_directory),
        save_condition=save_condition)
    node.add_to_simulation_logic()

    num_agents = simulation.agents.array.size

    simulation.update()

    iterations_bar = tqdm(total=max_iterations, desc='Iterations')
    inactive_bar = tqdm(total=num_agents, desc='Inactive')

    prev = 0
    for i in range(max_iterations):
        simulation.update()
        iterations_bar.update(1)
        inactive = simulation.data.get('inactive', 0)
        inactive_bar.update(inactive - prev)
        if inactive >= num_agents and (i + 1) % 1000 == 0:
            break
        prev = inactive
    iterations_bar.close()
    inactive_bar.close()


def run_simulations(klass, num_simulations, max_iterations, base_directory):
    for _ in trange(num_simulations, desc='Simulation'):
        simulation = klass()
        run(simulation, max_iterations, base_directory)


if __name__ == '__main__':
    run_simulations(
        FinlandiaTalo2ndFloor,
        num_simulations=1,
        max_iterations=12 * 1000,
        base_directory='data/finlandia_talo')
