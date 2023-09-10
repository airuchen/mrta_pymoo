import numpy as np

from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling

from mrta.utils import (
    generate_random_order,
    generate_separation_index,
    generate_random_chromosome,
    convert_chromosome_to_allocation,
    convert_allocation_to_chromosome,
)


class MultipleRobotTaskAllocationProblem(ElementwiseProblem):
    def __init__(
        self,
        robot_num,
        mission_num,
        robot_to_mission_start_distance_lut,
        robot_to_mission_end_distance_lut,
        mission_end_to_mission_start_distance_lut,
        **kwargs
    ):
        super().__init__(
            n_var=robot_num + mission_num,
            n_obj=2,
            n_ieq_constr=1,  # ["G"]
            **kwargs
        )
        self.robot_num = robot_num
        self.mission_num = mission_num
        self.robot_to_mission_start_distance_lut = robot_to_mission_start_distance_lut
        self.robot_to_mission_end_distance_lut = robot_to_mission_end_distance_lut
        self.mission_end_to_mission_start_distance_lut = (
            mission_end_to_mission_start_distance_lut
        )

    def _evaluate(self, x, out, *args, **kwargs):
        allocation = convert_chromosome_to_allocation(
            self.robot_num, self.mission_num, x
        )
        fitness_total_sum = self.objective_func_total_sum(allocation)
        fitness_max_time_span = self.objective_func_max_time_span(allocation)
        constraint_matched_skills = self.objective_func(allocation)

        out["F"] = np.array(
            [fitness_total_sum, fitness_max_time_span],
            dtype=float,
        )
        out["G"] = np.isnan(
            constraint_matched_skills
        ).any()  # all robot should have at least one mission

    def objective_func_total_sum(self, mission_allocation):
        """
        Objective Function: Total Sum of Travel Distance.

        @param mission_allocation: mission queue for each robot
        @return: total traveled distance
        """
        return np.nansum(self.objective_func(mission_allocation))

    def objective_func_max_time_span(self, mission_allocation):
        """
        Objective Function: Max Time Span of Travel Distance.

        @param mission_allocation: mission queue for each robot
        @return: max time span of mission allocation
        """
        travel_distance_lut = self.objective_func(mission_allocation)
        return np.nanmax(travel_distance_lut)

    def objective_func(self, mission_allocation):
        """
        Objective Function: Travel Distance for each robot.

        @param mission_allocation: mission queue for each robot
        @return: travel distance for each robot
        """
        distance_each_robot = np.full(self.robot_num, np.nan)
        for robot_index, mission_queue in enumerate(mission_allocation):
            if np.size(mission_queue) == 0 or np.array_equal(mission_queue[0], None):
                distance_each_robot[robot_index] = np.nan
                continue
            distance_per_robot = 0
            distance_per_robot += self.robot_to_mission_start_distance_lut[robot_index][
                mission_queue[0]
            ]
            for current_mission, next_mission in zip(
                mission_queue[:-1], mission_queue[1:]
            ):
                if (
                    self.robot_to_mission_start_distance_lut[robot_index][next_mission]
                    == np.inf
                ):
                    distance_per_robot = np.inf
                distance_per_robot += self.mission_end_to_mission_start_distance_lut[
                    current_mission
                ][current_mission]
                distance_per_robot += self.mission_end_to_mission_start_distance_lut[
                    current_mission
                ][next_mission]
            distance_per_robot += self.mission_end_to_mission_start_distance_lut[
                mission_queue[-1]
            ][mission_queue[-1]]
            distance_per_robot += self.robot_to_mission_end_distance_lut[robot_index][
                mission_queue[-1]
            ]  # TSP
            distance_each_robot[robot_index] = distance_per_robot
        return distance_each_robot


class MultipleRobotMultipleRobotTaskAllocationSampling(Sampling):
    def __init__(
        self,
        previous_best_allocation_results=None,
        robot_to_mission_skill_fulfillment_lut=None,
        mission_priorities=None,
    ):
        super().__init__()
        self.previous_best_allocation_results = previous_best_allocation_results
        self.robot_to_mission_skill_fulfillment_lut = (
            robot_to_mission_skill_fulfillment_lut
        )
        self.mission_priorities = mission_priorities

    def _do(self, problem, n_samples, **kwargs):
        if n_samples < 2:
            X = np.array(
                [
                    generate_random_chromosome(problem.robot_num, problem.mission_num)
                    for _ in range(n_samples)
                ]
            )
            return X
        X = np.array(
            [
                generate_random_chromosome(problem.robot_num, problem.mission_num)
                for _ in range(n_samples)
            ]
        )

        if self.previous_best_allocation_results is None:
            return X

        X = np.vstack(
            (
                X[:-1],
                self.previous_best_allocation_results.reshape(1, -1),
            )
        )
        return X


class MultipleRobotMultipleRobotTaskAllocationCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):
        """
        Crossover operations.

        - (70%) Two chromosomes crossover
        - (5%) Mission-order switch mutation
        - (10%) Single chromosome crossover
        - (15%) Replication

        @param X: Selected parents for crossover operation. Shape: [n_parents=2, n_matings=(population number)/2, n_var]
        @return: Y: Offsprings after crossover of parents. Shape: [n_parents=2, n_matings=(population number)/2, n_var]

        @note: The shape of X and Y is identical.
        """
        # best 0.3, 0.1, 0.0
        TWO_CHROMOSOMES_CROSSOVER_RATE = 0.7
        MISSION_ORDER_SWITCH_MUTATION_RATE = 0.05
        SINGLE_CHROMOSOME_CROSSOVER_RATE = 0.1
        _, n_matings, _ = X.shape
        Y = np.full_like(X, None, dtype=object)
        for k in range(n_matings):
            r = np.random.random()
            p1, p2 = X[0, k, :], X[1, k, :]
            if r < TWO_CHROMOSOMES_CROSSOVER_RATE:
                # Two chromosomes crossover
                r = np.random.random()
                p1[0 : problem.mission_num], p2[0 : problem.mission_num] = (
                    p2[0 : problem.mission_num],
                    p1[0 : problem.mission_num].copy(),
                )
            elif (
                r
                < (TWO_CHROMOSOMES_CROSSOVER_RATE + MISSION_ORDER_SWITCH_MUTATION_RATE)
                and problem.mission_num > 1
            ):
                # Mission-order switch mutation
                switch_pair = np.random.randint(problem.mission_num, size=2)
                (
                    p1[switch_pair[0]],
                    p1[switch_pair[1]],
                ) = (
                    p1[switch_pair[1]],
                    p1[switch_pair[0]],
                )
                (
                    p2[switch_pair[0]],
                    p2[switch_pair[1]],
                ) = (
                    p2[switch_pair[1]],
                    p2[switch_pair[0]],
                )
            elif (
                r
                < (
                    TWO_CHROMOSOMES_CROSSOVER_RATE
                    + MISSION_ORDER_SWITCH_MUTATION_RATE
                    + SINGLE_CHROMOSOME_CROSSOVER_RATE
                )
                and problem.mission_num > 0
            ):
                # Single chromosome crossover​
                self_crossover_point = np.random.randint(problem.mission_num)
                p1 = np.concatenate(
                    (
                        p1[self_crossover_point : problem.mission_num],
                        p1[0:self_crossover_point],
                        p1[problem.mission_num :],
                    )
                )
                p2 = np.concatenate(
                    (
                        p2[self_crossover_point : problem.mission_num],
                        p2[0:self_crossover_point],
                        p2[problem.mission_num :],
                    )
                )
            else:
                # Replication
                p1, p2 = np.copy(p1), np.copy(p2)
            Y[0, k, :], Y[1, k, :] = p1, p2
        return Y


class MultipleRobotMultipleRobotTaskAllocationMutation(Mutation):
    def __init__(self):
        super().__init__(prob_var=0.1)

    def _do(self, problem, X, **kwargs):
        """
        Mutation operations.

        - (5%) Mission-number switch mutation
        - (40%) Mission-order shuffle mutation
        - (5%) Mission-number random mutation
        - (5%) Mission-order random mutation
        - (10%) Mission-order shuffle-per-robot mutation
        - (35%) Replication

        @param X: All individuals in the population. Shape: [population size, n_var]
        @return: X: Offsprings after mutation of parents. Shape: [population size, n_var]
        """
        MISSION_NUMBER_SWITCH_MUTATION_RATE = 0.05
        MISSION_ORDER_SHUFFLE_MUTATION_RATE = 0.4
        MISSION_NUMBER_RANDOM_MUTATION_RATE = 0.05
        MISSION_ORDER_RANDOM_MUTATION_RATE = 0.05
        MISSION_ORDER_SHUFFLE_PER_ROBOT_MUTATION_RATE = 0.1
        for i in range(len(X)):
            r = np.random.random()
            if r < MISSION_NUMBER_SWITCH_MUTATION_RATE:
                # Mission-number switch mutation
                switch_pair = np.random.randint(problem.robot_num, size=2)
                (
                    X[i][problem.mission_num + switch_pair[0]],
                    X[i][problem.mission_num + switch_pair[1]],
                ) = (
                    X[i][problem.mission_num + switch_pair[1]],
                    X[i][problem.mission_num + switch_pair[0]],
                )
            elif r < (
                MISSION_NUMBER_SWITCH_MUTATION_RATE
                + MISSION_ORDER_SHUFFLE_MUTATION_RATE
            ):
                # Mission-order shuffle mutation
                switch_pair = np.random.randint(problem.mission_num, size=2)
                (
                    X[i][switch_pair[0]],
                    X[i][switch_pair[1]],
                ) = (
                    X[i][switch_pair[1]],
                    X[i][switch_pair[0]],
                )
            elif r < (
                MISSION_NUMBER_SWITCH_MUTATION_RATE
                + MISSION_ORDER_SHUFFLE_MUTATION_RATE
                + MISSION_NUMBER_RANDOM_MUTATION_RATE
            ):
                # Mission-number random mutation
                X[i][problem.mission_num :] = generate_separation_index(
                    problem.robot_num, problem.mission_num
                )
            elif r < (
                MISSION_NUMBER_SWITCH_MUTATION_RATE
                + MISSION_ORDER_SHUFFLE_MUTATION_RATE
                + MISSION_NUMBER_RANDOM_MUTATION_RATE
                + MISSION_ORDER_RANDOM_MUTATION_RATE
            ):
                # Mission-order random mutation
                X[i][0 : problem.mission_num] = generate_random_order(
                    problem.mission_num
                )
            elif r < (
                MISSION_NUMBER_SWITCH_MUTATION_RATE
                + MISSION_ORDER_SHUFFLE_MUTATION_RATE
                + MISSION_NUMBER_RANDOM_MUTATION_RATE
                + MISSION_ORDER_RANDOM_MUTATION_RATE
                + MISSION_ORDER_SHUFFLE_PER_ROBOT_MUTATION_RATE
            ):
                # Mission-order shuffle-per-robot mutation
                for j in range(problem.robot_num):
                    np.random.shuffle(
                        X[i][
                            np.sum(
                                X[i][problem.mission_num : problem.mission_num + j]
                            ) : np.sum(
                                X[i][problem.mission_num : problem.mission_num + j + 1]
                            )
                        ]
                    )
            else:
                # Replication
                X[i] = np.copy(X[i])
        return X


class MultipleRobotMultipleRobotTaskAllocationDuplicateElimination(
    ElementwiseDuplicateElimination
):
    def is_equal(self, a, b):
        return (a.X == b.X).all()
