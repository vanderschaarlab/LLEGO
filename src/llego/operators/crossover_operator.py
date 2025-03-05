import asyncio

# Define logger
import logging
from typing import Callable, Dict, List, Optional, Tuple

from langchain.prompts import FewShotPromptTemplate, PromptTemplate

from llego.custom.parsing_to_string import parse_dict_to_string
from llego.custom.tree_validation import validate_individual
from llego.operators.evolutionary_operator import EvolutionaryOperator
from llego.operators.individual import Individual
from llego.utils.llm_api import LLM_API

logger = logging.getLogger(__name__)


class CrossoverOperator(EvolutionaryOperator):
    def __init__(
        self,
        llm_api: LLM_API,
        llm_output_parser: Callable,
        prompt_prefix: str,
        num_offspring: int,
        num_parents: int,
        ordering: str,
        parent_sampling_strategy: str,
        parent_sampling_kwargs: Dict,
        fitness_key: str,
        alpha: float,
        lower_is_better: bool,
        seed: int,
    ) -> None:
        super().__init__(
            llm_api=llm_api,
            prompt_prefix=prompt_prefix,
            num_offspring=num_offspring,
            num_parents=num_parents,
            ordering=ordering,
            parent_sampling_strategy=parent_sampling_strategy,
            parent_sampling_kwargs=parent_sampling_kwargs,
            llm_output_parser=llm_output_parser,
            lower_is_better=lower_is_better,
            seed=seed,
        )
        self.metric_name = "fitness"
        self.metric_key = fitness_key
        self.alpha = alpha
        self.PATIENCE = 3
        self.FLOAT_PRECISION = 4
        self.lower_is_better = lower_is_better
        assert isinstance(
            lower_is_better, bool
        ), "lower_is_better should be of type boolean"

    def _create_example_template(self, is_example=True) -> str:
        example_template = """
"""
        example_template += f"{self.metric_name}" + ": {" + f"{self.metric_key}" + "}, "

        example_template += "Expression: "
        if is_example:
            example_template += "{Q}"
        return example_template

    def _create_prompt_for_one_operation(
        self, serialized_examples: List[Dict]
    ) -> FewShotPromptTemplate:

        example_template = self._create_example_template(is_example=True)
        input_variables = ["Q"] + [self.metric_key]

        example_prompt = PromptTemplate(
            input_variables=input_variables,
            template=example_template,
        )

        suffix = self._create_example_template(is_example=False)

        few_shot_prompt = FewShotPromptTemplate(
            examples=serialized_examples,
            example_prompt=example_prompt,
            prefix=self.prompt_prefix,
            suffix=suffix,
            input_variables=[self.metric_key],
            example_separator="",
        )

        return few_shot_prompt

    def _compute_target_fitness(self, parents: List[Individual]) -> Dict:
        """
        Compute target fitness for the offspring
        Args:
            parents: list of parent individuals
        Return:
            target_fitness: Dict
        """

        fitnesses = [parent.fitness[self.metric_key] for parent in parents]
        if self.lower_is_better:
            target_fitness_value = min(fitnesses) - self.alpha * (
                max(fitnesses) - min(fitnesses)
            )
        else:
            target_fitness_value = max(fitnesses) + self.alpha * (
                max(fitnesses) - min(fitnesses)
            )
        target_fitness = {
            self.metric_key: f"{target_fitness_value:.{self.FLOAT_PRECISION}f}"
        }

        return target_fitness

    def generate_offspring(
        self, population: List[Individual], total_num_offspring: int
    ) -> Tuple[List[Individual], int]:

        self._check_population(population)

        num_operations = total_num_offspring // self.num_offspring

        logger.info(
            f"[LLEGO Crossover] Generating {total_num_offspring} offspring using {num_operations} operations. "
            f"Each operation sees {self.num_parents} parents, generates {self.num_offspring} offspring."
        )

        # Get a list of parents for each prompt
        list_of_parents = self.parent_sampler.sample_parents(population, num_operations)

        # Create the list of prompts
        list_of_prompts = []
        for parents in list_of_parents:
            serialized_parents = self._serialize_parents(
                parents,
                with_fitness=True,
                float_precision=self.FLOAT_PRECISION,
                sorting_key=self.metric_key,
            )
            prompt = self._create_prompt_for_one_operation(serialized_parents)
            target_fitness = self._compute_target_fitness(parents)
            list_of_prompts.append(prompt.format(**target_fitness))

        assert (
            len(list_of_prompts) == num_operations
        ), f"Expected {num_operations} prompts but got {len(list_of_prompts)} prompts."

        num_generated_offspring = 0  # tracks the number of offspring generated so far
        generated_offspring_population = []  # tracks the offspring generated so far
        tot_tokens_used = 0  # tracks the total number of tokens used so far
        patience = 0
        attempt_counter = 1

        while num_generated_offspring < total_num_offspring:

            llm_responses = asyncio.run(
                self.llm_api._async_generate_concurrently(
                    list_of_prompts, self.num_offspring + 2
                )
            )

            list_offspring, num_valid_offspring, token_used = (
                self.llm_output_parser.parse_llm_responses(llm_responses)
            )

            tot_tokens_used += token_used

            if num_valid_offspring == 0:
                # this indicates some sort of parsing error
                patience += 1
                logger.info(
                    f"[LLEGO Crossover] No valid offspring generated. Patience: {patience}/{self.PATIENCE}"
                )
            if patience == self.PATIENCE:
                logger.info(f"[LLEGO Crossover] Patience limit reached. Exiting...")
                break

            if num_valid_offspring > 0:
                generated_offspring_population.extend(list_offspring)
                num_generated_offspring += len(list_offspring)

            logger.info(
                f"[LLEGO Crossover] Attempt {attempt_counter}: Number of valid offspring selected {num_generated_offspring}/{total_num_offspring}."
            )
            attempt_counter += 1

        # convert offspring to list of individuals
        offspring_population = []
        for offspring in generated_offspring_population:
            try:
                # transform offspring to individual, validate and add to offspring population
                individual = Individual(
                    machine_readable_format=offspring,
                    llm_readable_format=parse_dict_to_string(offspring),
                )
                validate_individual(individual)
                offspring_population.append(individual)
            except Exception as e:
                logger.debug(f"Error validating offspring: {e}")
                continue

        num_valid_individuals = len(offspring_population)

        logger.info(
            f"[LLEGO Crossover] Number of valid offspring transformed to individuals {num_valid_individuals}/{num_generated_offspring}."
        )

        return offspring_population, tot_tokens_used
