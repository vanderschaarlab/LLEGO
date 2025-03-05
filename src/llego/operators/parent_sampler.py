import random
from typing import List

import numpy as np

from llego.operators.individual import Individual


class ParentSampler:
    def __init__(
        self,
        sampling_strategy: str,
        num_parents: int,
        lower_is_better: bool,
        seed: int,
        **kwargs,
    ):

        self.SAMPLING_STRATEGY = {
            "global_random": self.sample_global_random,
            "global_fitness_weighted": self.sample_global_fitness_weighted,
        }

        self.num_parents = num_parents
        self.lower_is_better = lower_is_better
        self.seed = seed
        self.sampling_key = kwargs.get("sampling_key", None)
        assert isinstance(
            lower_is_better, bool
        ), "lower_is_better should be of type boolean"

        if sampling_strategy == "global_fitness_weighted":
            assert (
                self.sampling_key is not None
            ), "sampling_key should be provided for global_fitness_weighted strategy"

        assert (
            sampling_strategy in self.SAMPLING_STRATEGY.keys()
        ), f"Sampling strategy {sampling_strategy} not found in {self.SAMPLING_STRATEGY.keys()}"
        self.sampling_f = self.SAMPLING_STRATEGY[sampling_strategy]

    def sample_parents(
        self, population: List[Individual], num_operations: int
    ) -> List[List[Individual]]:
        list_of_parents = self.sampling_f(population, num_operations)

        # do global assertion checks
        assert (
            len(list_of_parents) == num_operations
        ), f"Expected {num_operations} operations but got {len(list_of_parents)}"
        assert all(
            [len(parents) == self.num_parents for parents in list_of_parents]
        ), f"Expected {self.num_parents} parents"

        return list_of_parents

    def sample_global_random(
        self, population: List[Individual], num_operations: int
    ) -> List[List[Individual]]:
        """
        Randomly sample parents from population.
        """
        random.seed(self.seed)

        population_size = len(population)

        if population_size < self.num_parents:
            raise ValueError(
                f"Population size {population_size} is less than number of parents {self.num_parents}"
            )

        list_of_parents = []
        for _ in range(num_operations):
            selected_parents = random.sample(population, self.num_parents)
            list_of_parents.append(selected_parents.copy())

        return list_of_parents

    def sample_global_fitness_weighted(
        self, population: List[Individual], num_operations: int
    ) -> List[List[Individual]]:
        """
        Sample parents from population based on fitness.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)

        population_size = len(population)

        if population_size < self.num_parents:
            raise ValueError(
                f"Population size {population_size} is less than number of parents {self.num_parents}"
            )

        # sort population by fitness
        population = sorted(
            population,
            key=lambda x: x.fitness[self.sampling_key],  # type:ignore
            reverse=True,
        )

        fitness_values = []
        # get fitness values
        if self.lower_is_better:
            for individual in population:
                assert individual.fitness is not None, "Fitness is None"
                fitness_values.append(1 / individual.fitness[self.sampling_key])

        else:
            for individual in population:
                assert individual.fitness is not None, "Fitness is None"
                fitness_values.append(individual.fitness[self.sampling_key])

        fitness_values = [max(0, fitness) for fitness in fitness_values]

        # calculate probability distribution
        sum_fitness = sum(fitness_values)
        probability_distribution = [fitness / sum_fitness for fitness in fitness_values]

        list_of_parents = []
        for _ in range(num_operations):
            selected_parents = np.random.choice(
                population,
                p=probability_distribution,
                size=self.num_parents,
                replace=False,
            )

            list_of_parents.append(selected_parents.tolist().copy())

        return list_of_parents
