from typing import Dict, Optional

import numpy as np


class Individual:
    __slots__ = [
        "_machine_readable_format",
        "_llm_readable_format",
        "_fitness",
        "_functional_signature",
    ]  # allowed attributes

    def __init__(
        self,
        machine_readable_format: Optional[Dict] = None,
        llm_readable_format: Optional[str] = None,
        fitness: Optional[Dict] = None,
        functional_signature: Optional[np.ndarray] = None,
    ):

        self._machine_readable_format = None
        self._llm_readable_format = None
        self._fitness = None
        self._functional_signature = None
        # machine readable format of the individual
        if machine_readable_format:
            self.machine_readable_format = machine_readable_format
        # llm readable format of the individual (string), this is passed to the LLM
        if llm_readable_format:
            self.llm_readable_format = llm_readable_format
        # fitness of the individual
        if fitness:
            self.fitness = fitness
        # functional signature of the individual
        if functional_signature:
            self.functional_signature = functional_signature

    @property
    def machine_readable_format(self):
        return self._machine_readable_format

    @machine_readable_format.setter
    def machine_readable_format(self, value: Dict):
        if not isinstance(value, Dict):
            raise ValueError(
                f"Invalid machine readable format, expected dict but got {type(value)}"
            )
        self._machine_readable_format = value

    @property
    def llm_readable_format(self):
        return self._llm_readable_format

    @llm_readable_format.setter
    def llm_readable_format(self, value: str):
        if not isinstance(value, str):
            raise ValueError(
                f"Invalid llm readable format, expected string but got {type(value)}"
            )
        self._llm_readable_format = value

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value):
        if not isinstance(value, Dict):
            raise ValueError(f"Invalid fitness, expected dict but got {type(value)}")
        self._fitness = value

    @property
    def functional_signature(self):
        return self._functional_signature

    @functional_signature.setter
    def functional_signature(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"Invalid functional signature, expected numpy array but got {type(value)}"
            )
        self._functional_signature = value

    def __eq__(self, other) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented

        # compare machine_readable_format (dict)
        machine_eq = self.machine_readable_format == other.machine_readable_format

        # compare llm_readable_format (string)
        llm_eq = self.llm_readable_format == other.llm_readable_format

        # compare fitness (dict)
        fitness_eq = self.fitness == other.fitness

        # compare functional_signature (numpy array)
        signature_eq = np.array_equal(
            self.functional_signature, other.functional_signature
        )

        return machine_eq and llm_eq and fitness_eq and signature_eq
