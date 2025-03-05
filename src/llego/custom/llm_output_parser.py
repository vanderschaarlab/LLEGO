"""
Parses the output of a LLM API call to a dictionary representation of the tree
"""

import logging
from typing import Callable, Dict, List, Tuple, Union

from llego.custom.parsing_to_dict import parse_string_to_dict
from llego.custom.tree_validation import (
    are_attributes_valid,
    get_dt_depth,
    validate_tree_dict,
)

logger = logging.getLogger(__name__)


class LLMOutputParser:
    def __init__(
        self, max_depth: int, tree_metadata: Dict, with_logprobs: bool = False
    ) -> None:
        self.with_logprobs = with_logprobs
        self.max_depth = max_depth
        self.tree_metadata = tree_metadata

        if self.with_logprobs:
            self.parse_f = self.parse_llm_responses_with_logprobs
        else:
            self.parse_f = self.parse_llm_responses_without_logprobs  # type: ignore

    def parse_llm_responses(
        self, llm_responses: List, **kwargs
    ) -> Union[Tuple[List, List, int], Tuple[List, List, int, int]]:
        """
        Parse the output of LLM API call to a list of dictionaries and number of valid offsprings
        """
        return self.parse_f(llm_responses, **kwargs)  # type: ignore

    def _track_token_usage(self, llm_responses: List) -> int:
        """
        Track the token usage in the responses
        """
        tokens_used = sum(
            [
                response["usage"]["total_tokens"]
                for response in llm_responses
                if response is not None
            ]
        )
        assert isinstance(tokens_used, int)
        return tokens_used

    def _parse_to_dict(self, serialized_offspring: str) -> Union[Dict, None]:
        """
        Parse the serialized offspring to a dictionary
        """
        # step 1. strip leading or trailing "## " and white spaces
        serialized_offspring = serialized_offspring.strip()
        serialized_offspring = serialized_offspring.lstrip("#")
        serialized_offspring = serialized_offspring.rstrip("#")
        serialized_offspring = serialized_offspring.strip()

        valid_attribute_names = self.tree_metadata["attribute_names"]

        # step 2. parse the serialized offspring to a dictionary
        try:
            offspring_dict = parse_string_to_dict(serialized_offspring)
        except Exception as e:
            return None
        # step 3. validate the offspring dictionary
        try:
            validate_tree_dict(offspring_dict)
        except Exception as e:
            return None
        # step 4. check if the depth of the offspring dictionary is within the limit
        try:
            assert get_dt_depth(offspring_dict) <= self.max_depth
        except Exception as e:
            return None
        # step 5. check that attribute names in the offspring dictionary are valid
        try:
            assert are_attributes_valid(offspring_dict, valid_attribute_names)
        except Exception as e:
            return None

        assert isinstance(offspring_dict, dict)

        return offspring_dict

    def parse_llm_responses_with_logprobs(
        self, llm_responses: List, gpt4o: bool = False
    ) -> Tuple[List[List], List[List], int, int]:
        """
        Parse the output of LLM API call to a list of dictionaries, list of logprobs and number of valid offsprings
        """
        num_valid_offspring = 0
        num_generated_offspring = 0
        list_offsprings: list = [[] for _ in range(len(llm_responses))]
        list_logprobs: list = [[] for _ in range(len(llm_responses))]

        for response_i, response in enumerate(llm_responses):
            if response is None:
                continue
            n_generations_per_prompt = len(response["choices"])
            for generation_j in range(n_generations_per_prompt):
                num_generated_offspring += 1
                if gpt4o:
                    serialized_offspring = response["choices"][generation_j]["message"][
                        "content"
                    ]
                else:
                    serialized_offspring = response["choices"][generation_j]["text"]

                offspring_dict = self._parse_to_dict(serialized_offspring)

                if offspring_dict is None:
                    offspring_dict = self._parse_to_dict(
                        f"{serialized_offspring}" + "}"
                    )

                if offspring_dict is not None:

                    list_offsprings[response_i].append(offspring_dict)

                    # add the logprob to the list of logprobs
                    if gpt4o:
                        logprob = sum(
                            [
                                x.logprob
                                for x in response["choices"][generation_j]["logprobs"][
                                    "content"
                                ]
                            ]
                        )
                    else:
                        logprob = sum(
                            response["choices"][generation_j]["logprobs"][
                                "token_logprobs"
                            ]
                        )
                    list_logprobs[response_i].append(logprob)

                    num_valid_offspring += 1

        assert len(list_offsprings) == len(list_logprobs) == len(llm_responses)
        assert all(
            len(list_offsprings[i]) == len(list_logprobs[i])
            for i in range(len(llm_responses))
        )

        tokens_used = self._track_token_usage(llm_responses)

        logger.info(
            f"[LLEGO Mutation Offspring Parser] Number of valid offspring generated: {num_valid_offspring}/{num_generated_offspring}."
        )

        return list_offsprings, list_logprobs, num_valid_offspring, tokens_used

    def parse_llm_responses_without_logprobs(
        self, llm_responses: List
    ) -> Tuple[List[dict], int, int]:
        """
        Parse the output of LLM API call to a list of dictionaries and number of valid offsprings
        """
        num_valid_offspring = 0
        num_generated_offspring = 0
        list_offsprings = []

        for response_i, response in enumerate(llm_responses):
            if response is None:
                continue
            n_generations_per_prompt = len(response["choices"])
            for generation_j in range(n_generations_per_prompt):
                num_generated_offspring += 1
                serialized_offspring = response["choices"][generation_j]["message"][
                    "content"
                ]

                offspring_dict = self._parse_to_dict(serialized_offspring)
                if offspring_dict is None:
                    offspring_dict = self._parse_to_dict(
                        f"{serialized_offspring}" + "}"
                    )

                if offspring_dict is not None:
                    list_offsprings.append(offspring_dict)
                    num_valid_offspring += 1

        tokens_used = self._track_token_usage(llm_responses)
        logger.info(
            f"[LLEGO Crossover Offspring Parser] Number of valid offspring generated: {num_valid_offspring}/{num_generated_offspring}."
        )

        return list_offsprings, num_valid_offspring, tokens_used
