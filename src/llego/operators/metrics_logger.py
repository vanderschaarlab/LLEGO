import copy
from typing import Dict, List

import numpy as np
import wandb

from llego.operators.filter_operator import Filter
from llego.operators.individual import Individual


class MetricsLogger:
    def __init__(self, log_wandb: bool, filter: Filter) -> None:
        self.log_wandb = log_wandb
        self.filter_operator = filter
        self.PREFIX = ["population", "crossover", "mutation"]
        self.population_across_iterations: Dict = {}

    def get_population_across_iterations(self, num_iterations: int) -> Dict:
        assert all(
            [
                iter in self.population_across_iterations
                for iter in range(num_iterations + 1)
            ]
        )
        return self.population_across_iterations

    def compute_fitness_statistics(self, population: List) -> Dict:

        dict_logging = {}
        for feature_name in population[0].fitness.keys():
            fitness_values = [ind.fitness[feature_name] for ind in population]

            mean_val = np.mean(fitness_values)
            median_val = np.median(fitness_values)
            min_val = np.min(fitness_values)
            max_val = np.max(fitness_values)

            fitness_statistics = {
                f"{feature_name}/mean": mean_val,
                f"{feature_name}/median": median_val,
                f"{feature_name}/min": min_val,
                f"{feature_name}/max": max_val,
            }
            dict_logging.update(fitness_statistics)

        return dict_logging

    def compute_diversity(self, population: List[Individual]) -> Dict:
        # compute the median and mean L1 distance for the functional signatures
        functional_signatures = []
        for ind in population:
            assert ind.functional_signature is not None, "Functional signature is None"
            functional_signatures.append(ind.functional_signature)

        n = len(functional_signatures)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.sum(
                    np.abs(functional_signatures[i] - functional_signatures[j])
                )
        mean_distance = np.mean(distances)
        median_distance = np.median(distances)
        return {
            "mean_l1_distance": mean_distance,
            "median_l1_distance": median_distance,
        }

    def compute_uniqueness(self, population: List[Individual]) -> Dict:

        filtered_population = self.filter_operator.filter_population(population)
        uniqueness = len(filtered_population) / len(population)

        return {"uniqueness": uniqueness}

    def log_population(
        self, population: List[Individual], step: int, prefix: str
    ) -> Dict:
        assert prefix in self.PREFIX, f"prefix should be one of {self.PREFIX}"

        # log population
        if step not in self.population_across_iterations:
            self.population_across_iterations[step] = {}
        self.population_across_iterations[step][prefix] = copy.deepcopy(population)

        # fitness based logging
        dict_logging = self.compute_fitness_statistics(population)

        # diversity based logging
        dict_diversity = self.compute_diversity(population)
        dict_logging.update(dict_diversity)

        # uniqueness based logging
        dict_uniqueness = self.compute_uniqueness(population)
        dict_logging.update(dict_uniqueness)

        # update the keys of the dictionary with the prefix
        dict_logging = {
            prefix + "/" + key: value for key, value in dict_logging.items()
        }

        if self.log_wandb:
            wandb.log(dict_logging, step=step)

        self.population_across_iterations[step].update(dict_logging)

        return dict_logging
