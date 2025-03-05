import logging
import random
from typing import Callable

import numpy as np
from gatree.ga.crossover import Crossover
from gatree.ga.mutation import Mutation

from baselines.utils.gatree_utils import tournament_selection
from llego.custom.fitness_evaluation import FitnessEvaluation
from llego.custom.population_initialization import PopulationInitialization
from llego.operators.hof import HallOfFame
from llego.operators.metrics_logger import MetricsLogger
from llego.utils.tree import convert_gatree_to_individual, convert_individual_to_gatree

logger = logging.getLogger(__name__)


class GATreeAlgorithm:
    def __init__(
        self,
        n_iterations: int,
        pop_size: int,
        mutation_probability: float,
        pop_initializer: PopulationInitialization,
        fitness_evaluator: FitnessEvaluation,
        metrics_logger: MetricsLogger,
        hall_of_fame: HallOfFame,
        elite_size: int,
        fitness_metric: str,
        lower_is_better: bool,
        selection_tournament_size: int,
        max_depth: int = 3,
    ) -> None:

        self.n_iterations = n_iterations
        self.pop_size = pop_size
        self.elite_size = elite_size
        self.mutation_probability = mutation_probability
        self.fitness_metric = fitness_metric
        self.lower_is_better = lower_is_better
        self.selection_tournament_size = selection_tournament_size
        self.max_depth = max_depth

        # Initiliaze the operators
        self.pop_initializer = pop_initializer
        self.fitness_evaluator = fitness_evaluator
        self.metrics_logger = metrics_logger
        self.hall_of_fame = hall_of_fame
        self.random = np.random

    def run(
        self,
        att_indexes: list,
        att_values: dict,
        class_count: int,
        feature_list: list,
        task: str,
        seed: int = 42,
    ):
        """
        Run the algorithm for a certain number of iterations
        """
        # Set all the seeds
        np.random.seed(seed)
        random.seed(seed)

        # Initialize the population
        ind_population = self.pop_initializer.generate_population(
            init_pop_size=self.pop_size
        )

        # Evaluate the fitness for the initial population
        self.fitness_evaluator.evaluate_fitness(ind_population, verbose=True)

        # Log the initial population
        _ = self.metrics_logger.log_population(
            ind_population, step=0, prefix="population"
        )

        # Update the hall of fame
        self.hall_of_fame.update_hof(ind_population)

        # Run the algorithm for a certain number of iterations
        for iteration in range(1, self.n_iterations + 1):

            logger.info(
                f"[GATree Algorithm] Iteration {iteration}/{self.n_iterations}..."
            )

            # sort the population on fitness
            ind_population.sort(
                key=lambda x: x.fitness[self.fitness_metric],  # type: ignore
                reverse=not self.lower_is_better,
            )

            # Elites
            ind_elites = ind_population[: self.elite_size]

            # Offspring generation
            gatree_descendant = []
            for _ in range(0, len(ind_population), 2):
                # Tournament selection (this function works with our individuals class)
                ind_tree1, ind_tree2 = tournament_selection(
                    population=ind_population,
                    selection_tournament_size=self.selection_tournament_size,
                    fitness_key=self.fitness_metric,
                    reverse=not self.lower_is_better,
                    random=self.random,  # type: ignore
                )

                gatree_tree1 = convert_individual_to_gatree(
                    ind_tree1, task=task, feature_list=feature_list
                )
                gatree_tree2 = convert_individual_to_gatree(
                    ind_tree2, task=task, feature_list=feature_list
                )

                too_deep = True
                while too_deep:
                    # Crossover between selected trees
                    crossover1 = Crossover.crossover(
                        tree1=gatree_tree1, tree2=gatree_tree2, random=self.random  # type: ignore
                    )
                    if crossover1.depth() <= self.max_depth:
                        too_deep = False

                too_deep = True
                while too_deep:
                    crossover2 = Crossover.crossover(
                        tree1=gatree_tree1, tree2=gatree_tree2, random=self.random  # type: ignore
                    )
                    if crossover2.depth() <= self.max_depth:
                        too_deep = False

                # Mutation of new trees
                mutation1 = crossover1
                mutation2 = crossover2
                max_iter = int(1e5)

                if self.random.random() < self.mutation_probability:
                    iter_count = 0
                    too_deep = True

                    while too_deep and iter_count < max_iter:
                        mutation1 = Mutation.mutation(
                            root=crossover1,
                            att_indexes=att_indexes,
                            att_values=att_values,
                            class_count=class_count,
                            random=self.random,
                        )

                        if mutation1.depth() <= self.max_depth:
                            too_deep = False

                        iter_count += 1

                    if iter_count >= max_iter:
                        logger.warning(
                            f"[GATree Algorithm] Mutation 1 took too long to find a valid tree. Falling back to crossover 1."
                        )
                        mutation1 = crossover1

                if self.random.random() < self.mutation_probability:
                    iter_count = 0
                    too_deep = True

                    while too_deep and iter_count < max_iter:
                        mutation2 = Mutation.mutation(
                            root=crossover2,
                            att_indexes=att_indexes,
                            att_values=att_values,
                            class_count=class_count,
                            random=self.random,
                        )

                        if mutation2.depth() <= self.max_depth:
                            too_deep = False

                        iter_count += 1

                    if iter_count >= max_iter:
                        logger.warning(
                            f"[GATree Algorithm] Mutation 2 took too long to find a valid tree. Falling back to crossover 2."
                        )
                        mutation2 = crossover2  # Fallback to original crossover result if no valid mutation found

                # Add new trees to descendant population
                gatree_descendant.extend([mutation1, mutation2])

            # Convert the descendants to our individuals
            ind_descendant = [
                convert_gatree_to_individual(ga_tree, feature_list)
                for ga_tree in gatree_descendant
            ]

            self.fitness_evaluator.evaluate_fitness(ind_descendant, verbose=True)

            # Elites + descendants
            ind_descendant.sort(
                key=lambda x: x.fitness[self.fitness_metric],  # type: ignore
                reverse=not self.lower_is_better,
            )

            ind_descendant = (
                ind_elites + ind_descendant[: self.pop_size - self.elite_size]
            )
            assert len(ind_descendant) == self.pop_size

            # Replace old population with new population
            ind_population = ind_descendant

            # Log the population
            _ = self.metrics_logger.log_population(
                ind_population, step=iteration, prefix="population"
            )

            self.hall_of_fame.update_hof(ind_population)

        # Select the best model by validation fitness
        self.fitness_evaluator.evaluate_fitness(ind_population)

        population_across_iterations = (
            self.metrics_logger.get_population_across_iterations(
                num_iterations=self.n_iterations
            )
        )

        hof = self.hall_of_fame.get_hof()

        logger.info("[GATree Algorithm] Finished.")

        return population_across_iterations, hof
