import matplotlib.pyplot as plt
import numpy as np

from pymoo.indicators.hv import Hypervolume
from pymoo.util.running_metric import RunningMetricAnimation

MISSION_INDEX_DTYPE = np.uint8


def is_beyond_dtype_range(num, dtype):
    return num > np.iinfo(dtype).max or num < np.iinfo(dtype).min


def generate_random_order(mission_num):
    """Create a queue of random numbers from 0 to num-1 as mission indices."""
    if is_beyond_dtype_range(mission_num, MISSION_INDEX_DTYPE):
        raise ValueError("mission_num is beyond dtype range")
    num_queue = np.arange(
        mission_num,
        dtype=MISSION_INDEX_DTYPE,
    )
    np.random.shuffle(num_queue)
    return num_queue


def generate_separation_index(robot_num, mission_num):
    """Create a queue of random numbers sum up to total mission number as mission number for each robot."""
    if is_beyond_dtype_range(mission_num, MISSION_INDEX_DTYPE):
        raise ValueError("mission_num is beyond dtype range")
    if robot_num < 1:
        raise ValueError("robot_num must be greater than 0")
    sum = 0
    values = np.array([], dtype=MISSION_INDEX_DTYPE)
    for _ in range(robot_num - 1):
        value = np.random.randint(0, mission_num + 1 - sum)
        sum += value
        values = np.append(values, value)
    values = np.append(values, mission_num - sum)
    return values


def convert_allocation_to_chromosome(allocation):
    """
    @param allocation: a list of list of mission indices. allocation[i] is a list of mission indices for robot i
    @return: Chromosome concatenates mission indices and number of missions for each robot
    """
    num_robot = len(allocation)
    num_mission = sum([len(subarray) for subarray in allocation])
    chromosome = np.array([None] * (num_robot + num_mission), dtype=object)
    current_index = 0
    for i in range(num_robot):
        for j in range(len(allocation[i])):
            chromosome[current_index] = allocation[i][j]
            current_index += 1
        chromosome[num_mission + i] = len(allocation[i])
    return chromosome


def convert_chromosome_to_allocation(num_robot, num_mission, chromosome):
    """
    @param num_robot: number of robots
    @param num_mission: number of missions
    @param chromosome: Chromosome concatenates mission indices and number of missions for each robot
    @return: a list of list of mission indices. allocation[i] is a list of mission indices for robot i
    """
    allocation = np.array([None] * num_robot, dtype=object)
    current_index = 0
    for i in range(num_robot):
        allocation[i] = chromosome[
            current_index : current_index + chromosome[num_mission + i]
        ]
        current_index += chromosome[num_mission + i]
    return allocation


def generate_random_chromosome(num_robot, num_mission):
    """
    @param num_robot: number of robots
    @param num_mission: number of missions
    @return: Chromosome concatenates mission indices and number of missions for each robot
    """
    separation_index = generate_separation_index(num_robot, num_mission)
    mission_order = generate_random_order(num_mission)
    return np.concatenate((mission_order, separation_index), dtype=object)


def generate_random_allocation(num_robot, num_mission):
    """
    @param num_robot: number of robots
    @param num_mission: number of missions
    @return: a list of list of mission indices. allocation[i] is a list of mission indices for robot i
    """
    return convert_chromosome_to_allocation(
        num_robot, num_mission, generate_random_chromosome(num_robot, num_mission)
    )


def plot_hypervolume_history(hypervolume_input, ref_point=None):
    """
    Plot the hypervolume history.

    @param hypervolume_input: a list of tuples. Each tuple contains the name of the algorithm and the result history
    @example: hypervolume_input = [("NSGA-II", result_history_nsga2), ("NSGA-III", result_history_nsga3)]
    """
    plt.figure(figsize=(10, 10))
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for name, result_history in hypervolume_input:
        n_evals = []  # corresponding number of function evaluations\
        hist_F = []  # the objective space values in each generation
        hist_F_avg = []  # average objective space values in each generation
        hist_X = []  # constraint violation in each generation
        for algo in result_history:
            # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)
            # retrieve the optimum from the algorithm
            opt = algo.opt
            # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
            hist_F_avg.append(opt.get("F").mean(axis=0))
            hist_X.append(opt.get("X"))

        if ref_point is None:
            ref_point = np.array(
                [2000.0, 2000, 100]
            )  # update this for each objective function
        approx_ideal = hist_F[-1].min(axis=0)
        approx_nadir = hist_F[-1].max(axis=0)

        metric = Hypervolume(
            ref_point=ref_point,
            norm_ref_point=False,
            zero_to_one=True,
            nadir=approx_nadir,
            ideal=approx_ideal,
        )

        hv = [metric.do(_F) for _F in hist_F]
        hv_avg = [metric.do(_F) for _F in hist_F_avg]

        color = colors.pop(0)
        plt.plot(n_evals, hv, color=color, lw=0.7, label="Objective value")
        plt.scatter(n_evals, hv, facecolor="none", edgecolor=color, marker="p")
        color = colors.pop(0)
        plt.plot(n_evals, hv_avg, color=color, lw=0.7, label="Avg. Objective value")
        plt.scatter(n_evals, hv_avg, facecolor="none", edgecolor=color, marker="p")
    plt.title("Hypervolume Convergence")
    plt.xlabel("Function Evaluations")
    plt.ylabel("Hypervolume")
    plt.legend()
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")


def plot_F_history(result_histories):
    """
    Plot the objective value history.

    @param result_histories: a list of tuples. Each tuple contains the name of the algorithm and the result history
    @example: result_histories = [("NSGA-II", result_history_nsga2), ("NSGA-III", result_history_nsga3)]
    """
    plt.figure()
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    for name, result_history in result_histories:
        ret = [np.min(e.pop.get("F")) for e in result_history]
        plt.plot(np.arange(len(ret)), ret[:], label=name, color=colors.pop(0))
    plt.title("objective value Convergence")
    plt.xlabel("Generation")
    plt.ylabel("Objective value")
    plt.legend()
    plt.draw()
    plt.pause(0.001)
    input("Press [enter] to continue.")


def plot_running_metrics(result_history):
    """
    Plot the running metrics showing difference in each generation.

    @param result_history: the result history of the algorithm
    """
    running = RunningMetricAnimation(
        delta_gen=50,
        n_plots=5,
        key_press=False,
        do_show=True,
    )
    for algorithm in result_history:
        running.update(algorithm=algorithm)


def plot_pseudo_weight_choice(objective_values, pseudo_weight):
    """
    Plot the pseudo weight choice.

    @param objective_values: normalized objective values of the solutions
    @param pseudo_weight: the index of the pseudo weight choice
    """
    plt.figure(figsize=(7, 5))
    plt.scatter(
        objective_values[:, 0],
        objective_values[:, 1],
        s=30,
        facecolors="none",
        edgecolors="blue",
    )
    plt.title("Pareto front Solutions in Objective Space")
    plt.xlabel("Total Traveled Distance [m]")
    plt.ylabel("Max Time Span [s]")
    plt.scatter(
        objective_values[pseudo_weight, 0],
        objective_values[pseudo_weight, 1],
        marker="x",
        color="red",
        s=200,
    )
    plt.show()


def constraint_satisfactory(result_history):
    """
    Plot the constraint violation history.

    @param result_history: the result history of the algorithm
    """
    n_evals = []  # corresponding number of function evaluations\
    hist_F = []  # the objective space values in each generation
    hist_cv = []  # constraint violation in each generation
    hist_cv_avg = []  # average constraint violation in the whole population

    for algo in result_history:
        # store the number of function evaluations
        n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
        opt = algo.opt

        # store the least contraint violation and the average in each population
        hist_cv.append(opt.get("CV").min())
        hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
        feas = np.where(opt.get("feasible"))[0]
        hist_F.append(opt.get("F")[feas])

    print("hist_cv: ", hist_cv)
    if not np.any(np.array(hist_cv) <= 0.0):
        print("No feasible solution found.")
    else:
        k = np.where(np.array(hist_cv) <= 0.0)[0].min()
        print(
            f"At least one feasible solution in Generation {k} after {n_evals[k]} evaluations."
        )
        # replace this line by `hist_cv` if you like to analyze the least feasible optimal solution and not the population
        vals = hist_cv_avg

        k = np.where(np.array(vals) <= 0.0)[0].min()
        print(
            f"Whole population feasible in Generation {k} after {n_evals[k]} evaluations."
        )

        plt.figure(figsize=(7, 5))
        plt.plot(n_evals, vals, color="black", lw=0.7, label="Avg. CV of Pop")
        plt.scatter(n_evals, vals, facecolor="none", edgecolor="black", marker="p")
        plt.axvline(n_evals[k], color="red", label="All Feasible", linestyle="--")
        plt.title("Convergence")
        plt.xlabel("Function Evaluations")
        plt.legend()
        plt.show()
