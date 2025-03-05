from typing import List

from llego.operators.individual import Individual


class Filter:
    def __init__(self, filter_type: str):
        self.filter_type = filter_type
        self.FILTER_FUNCTIONS = {
            "functional_signature": self.filter_functional_signature,
        }

    def filter_population(self, population: List[Individual]) -> List[Individual]:
        """
        Filter the population based on the filter type
        """
        assert (
            self.filter_type in self.FILTER_FUNCTIONS.keys()
        ), f"Filter type {self.filter_type} not found in {self.FILTER_FUNCTIONS.keys()}"
        return self.FILTER_FUNCTIONS[self.filter_type](population)

    def filter_functional_signature(
        self, population: List[Individual]
    ) -> List[Individual]:
        """
        Filter the population based on the functional signature
        """
        signatures = [ind.functional_signature for ind in population]
        unique_signatures = set()
        unique_indices = []

        for i, sig in enumerate(signatures):
            assert sig is not None, "Functional signature is None"
            sig_tuple = tuple(sig.tolist())  # Convert ndarray to tuple
            if sig_tuple not in unique_signatures:
                unique_signatures.add(sig_tuple)
                unique_indices.append(i)

        filtered_population = [population[i] for i in unique_indices]
        return filtered_population
