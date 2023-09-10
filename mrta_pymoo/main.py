import matplotlib.pyplot as plt
import numpy as np
import os
import time
import tsplib95

from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.mcdm.pseudo_weights import PseudoWeights
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from mrta.utils import (
    convert_chromosome_to_allocation,
    plot_travel_path,
    plot_pareto_front,
)
from mrta.multiple_robot_task_allocation_model import (
    MultipleRobotMultipleRobotTaskAllocationCrossover,
    MultipleRobotMultipleRobotTaskAllocationDuplicateElimination,
    MultipleRobotMultipleRobotTaskAllocationMutation,
    MultipleRobotMultipleRobotTaskAllocationSampling,
    MultipleRobotTaskAllocationProblem,
)

POPULATION_SIZE = 10
GENERATION_NUM = 3000
EXPERIMENT_TIMES = 1
ONLINE_OPTIMIZATION = True
PLOT_TRAVEL_PATH = True
PLOT_PARETO_FRONT = True


try:
    tsp_dataset_name = input(
        "Please input the name of the dataset (e.g. rat99, eil51, eil76): "
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tsp_dataset_path = os.path.join(
        script_dir, "..", "dataset", f"{tsp_dataset_name}.tsp"
    )
    tsp_dataset = tsplib95.load(tsp_dataset_path)
except:
    raise SystemExit("Dataset not found. Please put the dataset in the dataset folder.")

robot_num = int(input("Please input the number of robots: "))


# pre-calculated distance metric
G = tsp_dataset.get_graph()
node_num = G.number_of_nodes()
C = np.zeros((node_num, node_num))
for i in range(0, node_num):
    for j in range(0, node_num):
        C[i, j] = np.linalg.norm(
            [
                G.nodes[i + 1]["coord"][0] - G.nodes[j + 1]["coord"][0],
                G.nodes[i + 1]["coord"][1] - G.nodes[j + 1]["coord"][1],
            ]
        )

# In the implementation, the first city/node/POI/mission is the depot, and therefore the number of missions is node_num - 1
mission_num = node_num - 1
mission_table = C[1:, 1:]
robot_to_mission_start_table = np.tile(C[0, 1:], (robot_num, 1))
robot_to_mission_end_table = np.tile(C[1:, 0].T, (robot_num, 1))


problem = MultipleRobotTaskAllocationProblem(
    robot_num=robot_num,
    mission_num=mission_num,
    robot_to_mission_start_distance_lut=robot_to_mission_start_table,
    robot_to_mission_end_distance_lut=robot_to_mission_end_table,
    mission_end_to_mission_start_distance_lut=mission_table,
)

total_distance_list = []
distance_each_agent_list = []
max_makespan_list = []
process_time_list = []
selected_solution_list = []
pareto_front_list = []
previous_best_allocation = None

for _ in range(EXPERIMENT_TIMES):
    algorithm = NSGA2(
        pop_size=POPULATION_SIZE,
        sampling=MultipleRobotMultipleRobotTaskAllocationSampling(
            previous_best_allocation_results=previous_best_allocation
            if ONLINE_OPTIMIZATION
            else None,
        ),
        crossover=MultipleRobotMultipleRobotTaskAllocationCrossover(),
        mutation=MultipleRobotMultipleRobotTaskAllocationMutation(),
        save_history=False,  # set True to save history for visualization, default should be False
        eliminate_duplicates=MultipleRobotMultipleRobotTaskAllocationDuplicateElimination(),
    )
    start_time = time.time()
    res_nsga2 = minimize(
        problem,
        algorithm,
        get_termination("n_gen", GENERATION_NUM),
        seed=int(time.time()),
        verbose=False,
    )
    process_time_list.append(time.time() - start_time)

    # Multi-Criteria Decision Making
    results_nsga2 = res_nsga2.X[np.argsort(res_nsga2.F[:, 0])]
    F_nsga2 = res_nsga2.F

    weights = np.array(
        [0.5, 0.5]
    )  # The weights for [fitness_total_sum, fitness_max_time_span]
    pseudo_weight = PseudoWeights(weights).do(F_nsga2)
    previous_best_allocation = results_nsga2[pseudo_weight]
    result = convert_chromosome_to_allocation(
        robot_num, mission_num, res_nsga2.X[pseudo_weight]
    )
    selected_solution_list.append(result)
    traveled_distance_per_robot = problem.objective_func(result)
    distance_each_agent_list.append(traveled_distance_per_robot)
    max_makespan_list.append(max(traveled_distance_per_robot))
    total_distance_list.append(sum(traveled_distance_per_robot))
    pareto_front_list.append(F_nsga2)
    print(
        "Total traveled Distance: {}".format(problem.objective_func_total_sum(result))
    )
    print("Traveled Distance each robot: {}".format(traveled_distance_per_robot))

print("----------------------------------------")
print("Average Process Time: ", sum(process_time_list) / len(process_time_list))
print("Min Total Travel Distance: ", min(total_distance_list))
print(
    "Average Total Travel Distance: ",
    sum(total_distance_list) / len(total_distance_list),
)
print("Total Travel Distance std dev: ", np.std(total_distance_list))
print("Min Max Makespan: ", min(max_makespan_list))
print("Average Max Makespan: ", sum(max_makespan_list) / len(max_makespan_list))
print("Max Makespan std dev: ", np.std(max_makespan_list))
print("----------------------------------------")


if PLOT_TRAVEL_PATH:
    # draw
    min_index = total_distance_list.index(min(total_distance_list))
    solution = selected_solution_list[min_index]
    plot_travel_path(solution, tsp_dataset)


if PLOT_PARETO_FRONT:
    plot_pareto_front(pareto_front_list)
