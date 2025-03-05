import logging
import random
import time
from typing import Callable

import numpy as np

from llego.custom.fitness_evaluation import FitnessEvaluation
from llego.custom.population_initialization import PopulationInitialization
from llego.operators.crossover_operator import CrossoverOperator
from llego.operators.hof import HallOfFame
from llego.operators.metrics_logger import MetricsLogger
from llego.operators.mutation_operator import MutationOperator
from llego.operators.selection_operator import SelectionOperator

logger = logging.getLogger(__name__)


class Algorithm:
    def __init__(
        self,
        n_iterations: int,
        pop_size: int,
        n_offspring_mut: int,
        n_offspring_xo: int,
        use_crossover: bool,
        use_mutation: bool,
        pop_initializer: PopulationInitialization,
        pop_selector: SelectionOperator,
        crossover_operator: CrossoverOperator,
        mutation_operator: MutationOperator,
        fitness_evaluator: FitnessEvaluation,
        metrics_logger: MetricsLogger,
        hall_of_fame: HallOfFame,
    ) -> None:

        self.n_iterations = n_iterations
        self.pop_size = pop_size
        self.n_offspring_mut = n_offspring_mut
        self.n_offspring_xo = n_offspring_xo

        # Initialize the flags
        self.use_crossover = use_crossover
        self.use_mutation = use_mutation
        assert (
            self.use_crossover or self.use_mutation
        ), "At least one of the operators should be used"

        # Initiliaze the operators
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.pop_initializer = pop_initializer
        self.fitness_evaluator = fitness_evaluator
        self.pop_selector = pop_selector
        self.metrics_logger = metrics_logger
        self.hall_of_fame = hall_of_fame

    def run(self, seed: int = 42):
        """
        Run the algorithm for a certain number of iterations
        Inputs:
            fitness_f: fitness function that takes in a member of evolved population, returns dict of fitness scores
        """
        # Set all the seeds
        np.random.seed(seed)
        random.seed(seed)

        # Initialize the population
        population = self.pop_initializer.generate_population(
            init_pop_size=self.pop_size
        )

        # Evaluate the fitness for the initial population
        self.fitness_evaluator.evaluate_fitness(population, verbose=True)

        # Log the initial population
        _ = self.metrics_logger.log_population(population, step=0, prefix="population")
        # Update the hall of fame
        self.hall_of_fame.update_hof(population)

        # Run the algorithm for a certain number of iterations
        for iteration in range(1, self.n_iterations + 1):
            logger.info(
                f"[LLEGO Algorithm] Iteration {iteration}/{self.n_iterations}..."
            )

            # Select parents
            parents = population

            # Evaluate the fitness
            self.fitness_evaluator.evaluate_fitness(parents, verbose=True)

            # Crossover
            if self.use_crossover:
                offspring_crossover, _ = self.crossover_operator.generate_offspring(
                    parents, self.n_offspring_xo
                )
                # Evaluate the fitness
                self.fitness_evaluator.evaluate_fitness(offspring_crossover)

                _ = self.metrics_logger.log_population(
                    offspring_crossover, step=iteration, prefix="crossover"
                )
            else:
                offspring_crossover = []

            # Mutate
            if self.use_mutation:
                offspring_mutation, _ = self.mutation_operator.generate_offspring(
                    parents, self.n_offspring_mut
                )

                # Evaluate the fitness
                self.fitness_evaluator.evaluate_fitness(offspring_mutation)

                _ = self.metrics_logger.log_population(
                    offspring_mutation, step=iteration, prefix="mutation"
                )
            else:
                offspring_mutation = []

            # Select the population for the next iteration
            candidate_population = parents + offspring_crossover + offspring_mutation

            population = self.pop_selector.select(
                candidate_population, pop_size=self.pop_size
            )

            _ = self.metrics_logger.log_population(
                population, step=iteration, prefix="population"
            )

            self.hall_of_fame.update_hof(population)

            time.sleep(10)

        # Select the best model by validation fitness
        self.fitness_evaluator.evaluate_fitness(population)

        population_across_iterations = (
            self.metrics_logger.get_population_across_iterations(
                num_iterations=self.n_iterations
            )
        )

        hof = self.hall_of_fame.get_hof()

        logger.info("[LLEGO Algorithm] Done!")

        return population_across_iterations, hof
