import logging
from typing import Callable, Dict, List

from sklearn.ensemble import (  # type: ignore
    RandomForestClassifier,
    RandomForestRegressor,
)

from llego.custom.parsing_to_dict import parse_cart_to_dict
from llego.custom.parsing_to_string import parse_dict_to_string
from llego.custom.tree_validation import validate_individual
from llego.operators.individual import Individual

logger = logging.getLogger(__name__)


class PopulationInitialization:
    def __init__(
        self, data: dict, meta_data: dict, pop_init_f: str, max_depth: int, seed: int
    ):

        self.INIT_FUNCTIONS: Dict[str, Callable[[int], List[Individual]]] = {
            "cart": self.initialize_with_cart,
        }

        self.pop_init_f = pop_init_f
        self.seed = seed
        self.data = data
        self.meta_data = meta_data
        self.max_depth = max_depth

        assert (
            pop_init_f in self.INIT_FUNCTIONS.keys()
        ), f"Initialization function {pop_init_f} not found in {self.INIT_FUNCTIONS.keys()}"

        self.init_f = self.INIT_FUNCTIONS[pop_init_f]

    def generate_population(self, init_pop_size: int) -> List[Individual]:
        """
        Generate initial population as list of individuals,
        with llm_readable_format and machine_readable_format attributes populated
        """
        population = self.init_f(init_pop_size)

        # do global assertion checks
        for individual in population:
            assert isinstance(
                individual, Individual
            ), f"Expected Individual object but got {type(individual)}"
            assert (
                individual.machine_readable_format is not None
            ), "Machine readable format not initialized"
            assert (
                individual.llm_readable_format is not None
            ), "LLM readable format not initialized"

        num_valid_trees = len(population)
        logger.info(
            f"[LLEGO Population Initialization] Generated {num_valid_trees} valid trees out of {init_pop_size} trees"
        )

        assert isinstance(population, list), f"Expected list but got {type(population)}"

        return population

    def initialize_with_cart(self, init_pop_size: int) -> List[Individual]:

        population = []

        task_type = self.meta_data["task_type"]
        X_train, y_train = self.data["X_train"], self.data["y_train"]

        if task_type == "classification":
            rf = RandomForestClassifier(
                n_estimators=init_pop_size,
                max_depth=self.max_depth,
                random_state=self.seed,
                max_samples=0.5,
            )
            rf.fit(X_train, y_train)
        else:
            rf = RandomForestRegressor(
                n_estimators=init_pop_size,
                max_depth=self.max_depth,
                random_state=self.seed,
                max_samples=0.5,
            )
            rf.fit(X_train, y_train)

        assert (
            len(rf.estimators_) == init_pop_size
        ), f"Random forest not initialized with {init_pop_size} trees"

        feature_names = self.meta_data["attribute_names"]
        task_type = self.meta_data["task_type"]
        for cart_model in rf.estimators_:
            parsed_tree_dict = parse_cart_to_dict(
                cart_model, feature_names, task_type=task_type, precision=4
            )[0]
            assert isinstance(
                parsed_tree_dict, dict
            ), f"Expected tree in dict form but got: {type(parsed_tree_dict)}"
            try:
                # validate_tree_dict(parsed_tree_dict)
                individual = Individual(
                    machine_readable_format=parsed_tree_dict,
                    llm_readable_format=parse_dict_to_string(parsed_tree_dict),
                )
                validate_individual(individual, max_depth=self.max_depth)
                population.append(individual)
            except Exception as e:
                logger.info(f"Error validating tree: {e}")

        return population

    def initialize_with_random(self):
        """
        Initialize the population with random trees
        """
        pass
