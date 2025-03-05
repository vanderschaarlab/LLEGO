"""
Select offspring from the list of offsprings based on the selection strategy
"""

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class OffspringSelector:
    def __init__(
        self,
        offspring_selection_strategy: str,
        num_offspring_to_select: int,
        seed: int,
        sampling_temperature: float = 1.0,
    ) -> None:

        self.OFFSPRING_SELECTION_STRATEGY: Dict[str, Callable] = {
            "random": self.select_random,
            "logprob_weighted": self.select_logprob_weighted,
        }
        self.num_offspring_to_select = num_offspring_to_select
        self.seed = seed
        self.sampling_temperature = sampling_temperature
        assert sampling_temperature > 0, "Sampling temperature should be greater than 0"

        assert (
            offspring_selection_strategy in self.OFFSPRING_SELECTION_STRATEGY.keys()
        ), f"Offspring selection strategy {offspring_selection_strategy} not found in {self.OFFSPRING_SELECTION_STRATEGY.keys()}"

        self.selection_f = self.OFFSPRING_SELECTION_STRATEGY[
            offspring_selection_strategy
        ]

    def select_offspring(
        self, offsprings: List[List], offspring_logprob: Optional[List[List]] = None
    ) -> Tuple[List, int]:
        # global assertion checks
        if offspring_logprob is not None:
            assert len(offsprings) == len(
                offspring_logprob
            ), f"Expected {len(offsprings)} offsprings but got {len(offspring_logprob)} logprobs"
            assert all(
                [
                    len(offsprings[i]) == len(offspring_logprob[i])
                    for i in range(len(offsprings))
                ]
            ), f"Expected same number of offsprings and logprobs"

        selected_offsprings, num_selected_offsprings = self.selection_f(
            offsprings, offspring_logprob=offspring_logprob
        )

        assert (
            num_selected_offsprings > 0
        ), f"Expected at least 1 offspring but got {num_selected_offsprings}"

        return selected_offsprings, num_selected_offsprings

    def select_random(self, offsprings: List[List], **kwargs) -> Tuple[List, int]:
        """
        Randomly select offspring from the list of offsprings.
        """
        random.seed(self.seed)

        selected_offsprings = []
        for offspring in offsprings:
            if len(offspring) <= self.num_offspring_to_select:
                selected_offsprings.extend(offspring)
            else:
                selected_offsprings.extend(
                    random.sample(offspring, self.num_offspring_to_select)
                )

        num_selected_offsprings = len(selected_offsprings)

        return selected_offsprings, num_selected_offsprings

    def select_logprob_weighted(
        self, offsprings: List[List], offspring_logprob: List[List]
    ) -> Tuple[List, int]:
        """
        Weighted sampling of offspring from the list of offsprings by using logprobs.
        Lower logprobs (less likely) will be sampled more often.
        """
        random.seed(self.seed)

        selected_offsprings = []
        for offspring_idx, offspring in enumerate(offsprings):
            if len(offspring) <= self.num_offspring_to_select:
                selected_offsprings.extend(offspring)
            else:
                logprobs = offspring_logprob[offspring_idx]
                # take negative of logprobs because we want to sample lower logprobs more often
                # temperature scaling of logprobs
                adjusted_logprobs = [-lp / self.sampling_temperature for lp in logprobs]
                max_logprob = max(adjusted_logprobs)
                adjusted_logprobs = [
                    lp - max_logprob for lp in adjusted_logprobs
                ]  # for numerical stability
                weights = [np.exp(lp) for lp in adjusted_logprobs]
                weights = [w / sum(weights) for w in weights]

                # set the size to be the minimum between self.num_offspring_to_select and the number of non-zero logprobs

                n_selected = min(
                    self.num_offspring_to_select,
                    len([lp for lp in weights if lp != 0]),
                )

                sampled_offspring = np.random.choice(
                    offspring,
                    p=weights,
                    size=n_selected,
                    replace=False,
                )
                selected_offsprings.extend(sampled_offspring.tolist().copy())

        num_selected_offsprings = len(selected_offsprings)

        return selected_offsprings, num_selected_offsprings
