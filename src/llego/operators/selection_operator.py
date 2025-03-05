import logging
from typing import List, Optional

from llego.operators.filter_operator import Filter
from llego.operators.individual import Individual

logger = logging.getLogger(__name__)


class SelectionOperator:
    def __init__(
        self, filter: Optional[Filter], sorting_key: str, lower_is_better: bool
    ):
        self.filter = filter
        self.sorting_key = sorting_key
        self.lower_is_better = lower_is_better
        assert isinstance(
            lower_is_better, bool
        ), "lower_is_better should be of type boolean"

    def select(self, population: List[Individual], pop_size: int) -> List[Individual]:

        # Filtering step using the functional signatures
        if self.filter is not None:
            filtered_population = self.filter.filter_population(population)
        else:
            filtered_population = population

        filtered_population_size = len(filtered_population)
        original_population_size = len(population)

        logger.info(
            f"[LLEGO Selection] Filtered population size: {filtered_population_size}/{original_population_size}."
        )
        if len(filtered_population) > pop_size:

            if self.lower_is_better:
                sorted_population = sorted(
                    filtered_population,
                    key=lambda x: x.fitness[self.sorting_key],  # type: ignore
                    reverse=False,
                )
            else:
                sorted_population = sorted(
                    filtered_population,
                    key=lambda x: x.fitness[self.sorting_key],  # type: ignore
                    reverse=True,
                )
            selected_population = sorted_population[:pop_size]

        else:
            selected_population = (
                (pop_size // len(filtered_population)) + 1
            ) * filtered_population
            selected_population = selected_population[:pop_size]

        assert len(selected_population) == pop_size
        selected_population_size = len(selected_population)

        logger.info(
            f"[LLEGO Selection] Selected population size: {selected_population_size}/{pop_size}."
        )

        return selected_population
