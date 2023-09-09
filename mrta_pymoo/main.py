import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
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
    generate_random_chromosome,
    plot_pseudo_weight_choice,
    plot_F_history,
    plot_hypervolume_history,
    plot_running_metrics,
    constraint_satisfactory,
)
from mrta.multiple_robot_task_allocation_model import (
    MultipleRobotTaskAllocationProblem,
    MultipleRobotMultipleRobotTaskAllocationSampling,
    MultipleRobotMultipleRobotTaskAllocationCrossover,
    MultipleRobotMultipleRobotTaskAllocationMutation,
    MultipleRobotMultipleRobotTaskAllocationDuplicateElimination,
)

POPULATION_SIZE = 10
GENERATION_NUM = 4000
EXPERIMENT_TIMES = 5

tsp_dataset = tsplib95.load_problem("../tsp/rat99.tsp")
G = tsp_dataset.get_graph()
# robot_num = int(input("Please input the number of robots: "))
robot_num = int(input("Please input the number of robots: "))
node_num = G.number_of_nodes()

#
# print(G.nodes[1]["coord"]) # start from 1

# pre-calculated distance metric
C = np.zeros((node_num, node_num))
for i in range(0, node_num):
    for j in range(0, node_num):
        C[i, j] = np.linalg.norm(
            [
                G.nodes[i + 1]["coord"][0] - G.nodes[j + 1]["coord"][0],
                G.nodes[i + 1]["coord"][1] - G.nodes[j + 1]["coord"][1],
            ]
        )

mission_table = C[1:, 1:]
robot_to_mission_start_table = np.tile(C[0, 1:], (robot_num, 1))
robot_to_mission_end_table = np.tile(C[1:, 0].T, (robot_num, 1))

mission_num = node_num - 1


# ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)
algorithm = NSGA2(
    pop_size=POPULATION_SIZE,
    sampling=MultipleRobotMultipleRobotTaskAllocationSampling(
        previous_best_allocation_results=None,
    ),
    crossover=MultipleRobotMultipleRobotTaskAllocationCrossover(),
    mutation=MultipleRobotMultipleRobotTaskAllocationMutation(),
    save_history=False,  # set True to save history for visualization, default should be False
    eliminate_duplicates=MultipleRobotMultipleRobotTaskAllocationDuplicateElimination(),
    # ref_dirs=ref_dirs
)

problem = MultipleRobotTaskAllocationProblem(
    robot_num=robot_num,
    mission_num=mission_num,
    # robot_max_velocity=robot_max_velocities,
    robot_to_mission_start_distance_lut=robot_to_mission_start_table,
    robot_to_mission_end_distance_lut=robot_to_mission_end_table,
    mission_end_to_mission_start_distance_lut=mission_table,
    # mission_priorities=mission_priorities,
    # robot_to_mission_skill_fulfillment_lut=robot_to_mission_skill_fulfillment_lut,
)
termination = get_termination("n_gen", GENERATION_NUM)

total_distance_list = []
distance_each_agent_list = []
max_makespan_list = []
process_time_list = []
standard_deviation_list = []
ans_list = []
pareto_front_list = []

for _ in range(EXPERIMENT_TIMES):
    start_time = time.time()
    res_nsga2 = minimize(
        problem, algorithm, termination, seed=int(time.time()), verbose=False
    )
    end_time = time.time()
    process_time = end_time - start_time
    process_time_list.append(process_time)


    # ## Multi-Criteria Decision Making
    results_nsga2 = res_nsga2.X[np.argsort(res_nsga2.F[:, 0])]
    F_nsga2 = res_nsga2.F
    weights = np.array(
        [0.5, 0.5]
    )  # [fitness_total_sum, fitness_max_time_span, fitness_mission_priority]
    pseudo_weight = PseudoWeights(weights).do(F_nsga2)
    # plot_pseudo_weight_choice(F_nsga2, pseudo_weight)
    previous_best_allocation = results_nsga2[pseudo_weight]
    result = convert_chromosome_to_allocation(
        robot_num, mission_num, res_nsga2.X[pseudo_weight] # NSGA2
        # robot_num, mission_num, res_nsga2.X # GA
    )
    ans_list.append(result)
    traveled_distance_per_robot = problem.objective_func(result)
    distance_each_agent_list.append(traveled_distance_per_robot)
    max_makespan_list.append(max(traveled_distance_per_robot))
    standard_deviation_list.append(np.std(traveled_distance_per_robot))
    total_distance_list.append(sum(traveled_distance_per_robot))
    pareto_front_list.append(F_nsga2)
    # for i in range(robot_num):
    #     print("Robot {} : {}, travel distance: {}".format(i, result[i], traveled_distance_per_robot[i]))
    print("Total traveled Distance: {}".format(problem.objective_func_total_sum(result)))
    print("Traveled Distance each robot: {}".format(traveled_distance_per_robot))

    # plot_F_history([("NSGA-II", res_nsga2.history)])

    # plt.figure()

    # colors = ['black', 'blue', 'green', 'red', 'pink', 'orange', 'purple', 'brown', 'gray', 'olive', 'cyan']
    #     # sort the pareto front with x
    pareto_front = F_nsga2[np.argsort(F_nsga2[:, 0])]
    print("pareto_front: ", pareto_front)
    # # plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c=colors[i], label="pareto_front_"+str(i), marker='o-')
    # plt.plot(pareto_front[:, 0], pareto_front[:, 1], c=colors[0], label="pareto_front_itr_"+str(0), marker='o')
    # # link the points
    # # for i in range(len(pareto_front) - 1):
    # #     plt.plot([pareto_front[i][0], pareto_front[i + 1][0]], [pareto_front[i][1], pareto_front[i + 1][1]], c=colors[pareto_front_list.index(pareto_front)])

    # plt.title("Pareto front Solutions in Objective Space")
    # plt.xlabel("Total Traveled Distance [m]")
    # plt.ylabel("Max Makespan [m]")
    # plt.legend()
    # plt.show()

print("total_distance_list: ", total_distance_list)
print("min_distance: ", min(total_distance_list))
print("max_distance: ", max(total_distance_list))
print("average_distance: ", sum(total_distance_list) / len(total_distance_list))
print("average_process_time: ", sum(process_time_list) / len(process_time_list))
print("distance std dev: ", np.std(total_distance_list))

# choose the min total distance and print the distance of each agent
# min_index = total_distance_list.index(min(total_distance_list))
# print("min_distance_each_agent: ", distance_each_agent_list[min_index].tolist())
print("max_makespan: ", max_makespan_list)
print("average_makespan: ", sum(max_makespan_list) / len(max_makespan_list))
print("makespan std dev: ", np.std(max_makespan_list))
print("standard_deviation: ", standard_deviation_list[min_index])
print("process_time: ", process_time_list[min_index])


def list_to_path(l):
    path = []
    path.append((1, l[0]))
    for i in range(len(l) - 1):
        path.append((l[i], l[i + 1]))
    path.append((l[-1], 1))
    return path

# draw
# ans = ans_list[min_index]
# colors = ['black', 'blue', 'green', 'red', 'pink', 'orange', 'purple', 'brown', 'gray', 'olive', 'cyan']
# plt.figure(dpi=600)
# _, ax = plt.subplots()
# pos = tsp_dataset.node_coords
# nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=(0.4157, 0.3529, 0.3490))
# nx.draw_networkx_labels(G, pos=pos, labels={i: str(i) for i in range(1, len(G.nodes) + 1)}, font_size=8, font_color='white')
# for i in range(len(ans)):
#     solution = ans[i] + 2
#     path = list_to_path(solution)
#     nx.draw_networkx_edges(G, pos=pos, edgelist=path, arrows=True, edge_color=colors[i], label="robot_"+str(i))
#     # nx.draw_networkx_edges(G, pos=pos, edgelist=path, arrows=True, edge_color=[random.random() for i in range(3)])

# # If this doesn't exsit, x_axis and y_axis's numbers are not there.
# ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

# ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
# plt.subplots_adjust(right=0.8)  # Increase the right margin to make space for the legend
# plt.show()
