import asyncio
import logging
from typing import Callable, Dict, List, Tuple

from langchain.prompts import FewShotPromptTemplate, PromptTemplate

from llego.custom.parsing_to_string import parse_dict_to_string
from llego.custom.tree_validation import validate_individual
from llego.operators.evolutionary_operator import EvolutionaryOperator
from llego.operators.individual import Individual
from llego.operators.offspring_selector import OffspringSelector
from llego.utils.llm_api import LLM_API

logger = logging.getLogger(__name__)


class MutationOperator(EvolutionaryOperator):
    def __init__(
        self,
        llm_api: LLM_API,
        llm_output_parser: Callable,
        prompt_prefix: str,
        num_offspring: int,
        num_candidate_offspring: int,
        num_parents: int,
        ordering: str,
        parent_sampling_strategy: str,
        parent_sampling_kwargs: Dict,
        offspring_selection_strategy: str,
        offspring_selection_kwargs: Dict,
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
            seed=seed,
        )
        assert (
            num_candidate_offspring >= num_offspring
        ), f"num_candidate_offspring should be greater than or equal to num_offspring. Got {num_candidate_offspring} and {num_offspring} respectively."
        self.num_candidate_offspring = num_candidate_offspring
        self.PATIENCE = 3

        self.offspring_selector = OffspringSelector(
            offspring_selection_strategy=offspring_selection_strategy,
            num_offspring_to_select=num_offspring,
            seed=seed,
            **offspring_selection_kwargs,
        )

    def _create_prompt_for_one_operation(
        self, serialized_examples: List[Dict]
    ) -> FewShotPromptTemplate:

        example_template = """
Expression: {Q}"""

        example_prompt = PromptTemplate(
            input_variables=["Q"], template=example_template
        )

        suffix = """
Expression: ## """

        few_shot_prompt = FewShotPromptTemplate(
            examples=serialized_examples,
            example_prompt=example_prompt,
            prefix=self.prompt_prefix,
            suffix=suffix,
            example_separator="",
            input_variables=[],
        )

        return few_shot_prompt

    def generate_offspring(
        self, population: List[Individual], total_num_offspring: int
    ) -> Tuple[List[Individual], int]:
        """
        Generate offspring by mutating the population
        Args:
            population: list of individuals
            total_offspring: number of offspring to generate
        Returns:
            list of offspring individuals
        """
        self._check_population(population)

        num_operations = total_num_offspring // self.num_offspring

        logger.info(
            f"[LLEGO Mutation] Generating {total_num_offspring} offspring using {num_operations} operations. "
            f"Each operation sees {self.num_parents} parents, generates {self.num_candidate_offspring} candidate offspring, "
            f"selects {self.num_offspring} offspring."
        )

        list_of_parents = self.parent_sampler.sample_parents(population, num_operations)

        list_of_prompts = []
        for parents in list_of_parents:
            parents = self._serialize_parents(parents, with_fitness=False)
            prompt = self._create_prompt_for_one_operation(parents)
            list_of_prompts.append(prompt.format())

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
                    list_of_prompts, self.num_candidate_offspring + 2
                )
            )

            if self.llm_api.model.startswith("gpt4o"):
                gpt4o = True
            else:
                gpt4o = False
            list_offspring, list_logprobs, num_valid_offspring, token_used = (
                self.llm_output_parser.parse_llm_responses(llm_responses, gpt4o=gpt4o)
            )

            tot_tokens_used += token_used

            if num_valid_offspring == 0:
                # this indicates some sort of parsing error
                patience += 1
                logger.info(
                    f"[LLEGO Mutation] No valid offspring generated. Patience: {patience}/{self.PATIENCE}"
                )
            if patience == self.PATIENCE:
                logger.info(f"[LLEGO Mutation] Patience limit reached. Exiting...")
                break

            if num_valid_offspring > 0:
                list_offspring, num_selected_offspring = (
                    self.offspring_selector.select_offspring(
                        list_offspring, list_logprobs
                    )
                )
                generated_offspring_population.extend(list_offspring)
                num_generated_offspring += num_selected_offspring

            logger.info(
                f"[LLEGO Mutation] Attempt {attempt_counter}: Number of valid offspring selected {num_generated_offspring}/{total_num_offspring}."
            )
            attempt_counter += 1

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
            f"[LLEGO Mutation] Number of valid offspring transformed to individuals {num_valid_individuals}/{num_generated_offspring}."
        )

        return offspring_population, tot_tokens_used
