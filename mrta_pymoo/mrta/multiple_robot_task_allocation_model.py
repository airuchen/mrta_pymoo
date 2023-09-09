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
        # robot_max_velocity,
        robot_to_mission_start_distance_lut,
        robot_to_mission_end_distance_lut,
        mission_end_to_mission_start_distance_lut,
        # robot_to_mission_skill_fulfillment_lut,
        # mission_priorities,
        **kwargs
    ):
        super().__init__(
            n_var=robot_num + mission_num,
            n_obj=2,
            # n_eq_constr=2,  # ["H"]
            n_ieq_constr=1,  # ["G"]
            **kwargs
        )
        self.robot_num = robot_num
        self.mission_num = mission_num
        # self.robot_max_velocity = robot_max_velocity
        self.robot_to_mission_start_distance_lut = robot_to_mission_start_distance_lut
        self.robot_to_mission_end_distance_lut = robot_to_mission_end_distance_lut
        self.mission_end_to_mission_start_distance_lut = (
            mission_end_to_mission_start_distance_lut
        )
        # self.mission_priorities = mission_priorities
        # self.robot_to_mission_skill_fulfillment_lut = (
        #     robot_to_mission_skill_fulfillment_lut
        # )

    def _evaluate(self, x, out, *args, **kwargs):
        allocation = convert_chromosome_to_allocation(
            self.robot_num, self.mission_num, x
        )
        fitness_total_sum = self.objective_func_total_sum(allocation)
        fitness_max_time_span = self.objective_func_max_time_span(allocation)
        constraint_matched_skills = self.objective_func(allocation)
        # constraint_priority = self.constraint_func_priority(allocation)
        # fitness_mission_priority = self.objective_func_priority_diff(allocation)
        # fitness_mission_priority = self.objective_func_priority(allocation) # Test priority objective function

        out["F"] = np.array(
            # [fitness_total_sum, fitness_max_time_span, fitness_mission_priority],
            [fitness_total_sum, fitness_max_time_span],
            # [fitness_max_time_span],
            dtype=float,
        )
        # out["H"] = np.array(
        #     [
        #         np.isinf(constraint_matched_skills).any(),
        #         np.isnan(constraint_matched_skills).all(),
        #     ],
        #     dtype=float,
        # )
        # out["G"] = -constraint_priority
        out["G"] = np.isnan(
            constraint_matched_skills
        ).any()  # all robot should have at least one mission

    def constraint_func_priority(self, mission_allocation):
        """
        Constraint Function: The first mission for each robot should be the highest priority one in its queue.

        @param mission_allocation: mission queue for each robot
        @return: constraint_violation
        """
        priority_violation_check = 0
        for mission_queue in mission_allocation:
            if np.size(mission_queue) < 2 or np.array_equal(mission_queue[0], None):
                continue
            priority_in_mission_queue = np.array(
                [self.mission_priorities[i] for i in mission_queue], dtype=int
            )
            priority_violation_check += min(
                (priority_in_mission_queue[0] - np.max(priority_in_mission_queue)),
                0,
            )
        return priority_violation_check

    def objective_func_priority_diff(self, mission_allocation):
        """
        Objective Function: Sum up the conflicting priority (back - front).

        @example: [1, 5, 2] -> [4, x] -> 4*2 = 8
        @example: [1, 2, 4, 3] -> [1, 2, x] -> 1*3 + 2*2 = 7
        """
        priority_diff_cost = 0
        for mission_queue in mission_allocation:
            if np.size(mission_queue) < 2 or np.array_equal(mission_queue[0], None):
                continue
            priority_diff = np.clip(
                np.diff(
                    np.array(
                        [self.mission_priorities[i] for i in mission_queue], dtype=int
                    )
                ),
                a_max=None,
                a_min=0,
            )
            weight = np.arange(1, len(priority_diff) + 1)[::-1]
            priority_diff_cost += np.sum(priority_diff * weight)
        return priority_diff_cost

    # def objective_func_priority(self, mission_allocation):  # not used
    #     """
    #     Objective Function: Mission Priority.

    #     @param mission_allocation: mission queue for each robot
    #     @return: mission priority fitness value
    #     """
    #     mission_priority_gain_list = []
    #     for robot_index, mission_queue in enumerate(mission_allocation):
    #         mission_priority = [self.mission_priorities[i] for i in mission_queue]
    #         mission_priority_gain = 0
    #         for mission_index, mission in enumerate(mission_queue):
    #             if (
    #                 mission is None
    #                 or not self.robot_to_mission_skill_fulfillment_lut[robot_index][
    #                     mission_index
    #                 ]
    #             ):
    #                 continue
    #             weight = mission_index - len(mission_queue)
    #             mission_priority_gain += (
    #                 weight * weight * self.mission_priorities[mission]
    #             )
    #         mission_priority_gain_list.append(mission_priority_gain)
    #     mission_priority = np.nansum(mission_priority_gain)
    #     return mission_priority

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

        - (30%) Two chromosomes crossover
        - (30%) Mission-order switch mutation
        - (30%) Single chromosome crossover
        - (10%) Replication

        @param X: Selected parents for crossover operation. Shape: [n_parents=2, n_matings=(population number)/2, n_var]
        @return: Y: Offsprings after crossover of parents. Shape: [n_parents=2, n_matings=(population number)/2, n_var]

        @note: The shape of X and Y is identical.
        """
        # best 0.3, 0.1, 0.0
        TWO_CHROMOSOMES_CROSSOVER_RATE = 0.7 # 0.7
        MISSION_ORDER_SWITCH_MUTATION_RATE = 0.1 # 0.05
        SINGLE_CHROMOSOME_CROSSOVER_RATE = 0.1
        REPLICATION_RATE = 0.1
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

        - (25%) Mission-number switch mutation
        - (25%) Mission-order shuffle mutation
        - (10%) Mission-number random mutation
        - (10%) Mission-order random mutation
        - (20%) Mission-order shuffle-per-robot mutation
        - (10%) Replication

        @param X: All individuals in the population. Shape: [population size, n_var]
        @return: X: Offsprings after mutation of parents. Shape: [population size, n_var]
        """
        MISSION_NUMBER_SWITCH_MUTATION_RATE = 0.05
        MISSION_ORDER_SHUFFLE_MUTATION_RATE = 0.4 # effective 0.6
        MISSION_NUMBER_RANDOM_MUTATION_RATE = 0.05
        MISSION_ORDER_RANDOM_MUTATION_RATE = 0.05
        MISSION_ORDER_SHUFFLE_PER_ROBOT_MUTATION_RATE = 0.1
        # REPLICATION_RATE = 1 - (
        #     MISSION_NUMBER_SWITCH_MUTATION_RATE
        #     + MISSION_ORDER_SHUFFLE_MUTATION_RATE
        #     + MISSION_ORDER_RANDOM_MUTATION_RATE
        #     + MISSION_ORDER_SHUFFLE_PER_ROBOT_MUTATION_RATE
        # )
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
