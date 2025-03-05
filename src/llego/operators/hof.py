import copy
import logging
from typing import List

from llego.operators.individual import Individual

logger = logging.getLogger(__name__)


class HallOfFame:
    def __init__(self, hof_size: int, hof_metric: str, lower_is_better: bool) -> None:
        self.hof_size = hof_size
        self.hof_metric = hof_metric
        self.lower_is_better = lower_is_better
        self.hof: List[Individual] = []
        assert isinstance(lower_is_better, bool), "lower_is_better should be a boolean"

    def update_hof(self, population: List[Individual]) -> None:
        """
        Update hall of fame with the best individuals
        """
        current_hof = self.get_hof()
        if len(current_hof) > 0:
            assert current_hof[0].fitness is not None, "Fitness is None"
            current_best_fitness = current_hof[0].fitness[self.hof_metric]
        else:
            if self.lower_is_better:
                current_best_fitness = float("inf")
            else:
                current_best_fitness = -float("inf")

        hof_individual_fitness = []
        if len(current_hof) > 0:
            for ind in current_hof:
                assert ind.fitness is not None
                hof_individual_fitness.append(ind.fitness[self.hof_metric])
        else:
            hof_individual_fitness = []

        for individual in population:
            assert individual.fitness is not None, "Fitness is None"
            fitness_value = individual.fitness[self.hof_metric]
            if fitness_value not in hof_individual_fitness:
                self.hof.append(copy.deepcopy(individual))
                hof_individual_fitness.append(fitness_value)

        if self.lower_is_better:
            self.hof = sorted(
                self.hof, key=lambda x: x.fitness[self.hof_metric], reverse=False
            )
        else:
            self.hof = sorted(
                self.hof, key=lambda x: x.fitness[self.hof_metric], reverse=True
            )

        if self.hof_size is not None:
            if len(self.hof) > self.hof_size:
                self.hof = self.hof[: self.hof_size]

        num_individuals_added = 0
        num_individuals_improved_fitness = 0
        for ind in self.hof:
            if ind not in current_hof:
                num_individuals_added += 1

            assert ind.fitness is not None, "Fitness is None"
            if self.lower_is_better:

                if ind.fitness[self.hof_metric] < current_best_fitness:
                    num_individuals_improved_fitness += 1
            else:
                if ind.fitness[self.hof_metric] > current_best_fitness:
                    num_individuals_improved_fitness += 1

        logger.info(
            f"[LLEGO Hall Of Fame] Added {num_individuals_added} new individuals to HOF, {num_individuals_improved_fitness} individuals improved fitness; HOF size: {len(self.hof)}."
        )
        logger.info(
            f"[LLEGO Hall Of Fame] Current best individual: {[(key, f'{value:.3f}') for key, value in self.hof[0].fitness.items()]}."
        )

    def get_hof(self) -> List[Individual]:
        """
        Return the hall of fame
        """
        return copy.deepcopy(self.hof)
