import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error

from llego.custom.generic_tree import GenericTree
from llego.custom.parsing_to_string import parse_dict_to_string
from llego.operators.individual import Individual

logger = logging.getLogger(__name__)


class FitnessEvaluation:
    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        task_type: str,
        fitness_metric: str,
        complexity_metric: str,
        lower_is_better: bool,
        **kwargs,
    ):

        self.FITNESS_METRICS = {
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
            "mse": mean_squared_error,
        }

        self.COMPLEXITY_METRICS = {
            "depth": self._calculate_depth,
        }

        self.task_type = task_type

        assert (
            type(data["X_train"])
            == type(data["X_val"])
            == type(data["X_test"])
            == pd.DataFrame
        ), f"Expected pandas DataFrame but got {type(data['X_train'])}, {type(data['X_val'])}, {type(data['X_test'])}"
        assert (
            type(data["y_train"])
            == type(data["y_val"])
            == type(data["y_test"])
            == np.ndarray
        ), f"Expected numpy array but got {type(data['y_train'])}, {type(data['y_val'])}, {type(data['y_test'])}"

        self.data = data
        self.lower_is_better = lower_is_better
        assert (
            type(lower_is_better) == bool
        ), f"Expected boolean but got {type(lower_is_better)}"

        assert (
            fitness_metric in self.FITNESS_METRICS.keys()
        ), f"Metric {fitness_metric} not found in {self.FITNESS_METRICS.keys()}"

        assert (
            complexity_metric in self.COMPLEXITY_METRICS.keys()
        ), f"Metric {complexity_metric} not found in {self.COMPLEXITY_METRICS.keys()}"

        self.fitness_name = fitness_metric
        self.fitness_metric = self.FITNESS_METRICS[fitness_metric]

        self.complexity_name = complexity_metric
        self.complexity_metric = self.COMPLEXITY_METRICS[complexity_metric]

    def _calculate_depth(self, tree: GenericTree) -> int:
        assert isinstance(tree.depth, int)
        return tree.depth

    def evaluate_fitness(
        self,
        population: List[Individual],
        fit_tree: bool = False,
        verbose: bool = False,
    ) -> None:
        assert not fit_tree, "This method does not support fitting the tree."
        """
        Evaluate fitness of population in place
        Args:
            population: list of trees in dict format
        """

        if self.lower_is_better:
            best_fitness = float("inf")
        else:
            best_fitness = 0
        for individual in population:
            generic_tree = GenericTree(task=self.task_type)
            if fit_tree:
                generic_tree.create_from_dict(
                    dict_tree=individual.machine_readable_format,
                    X_train=self.data["X_train"],
                    y_train=self.data["y_train"],
                )
            else:
                generic_tree.construct_tree(
                    dict_tree=individual.machine_readable_format
                )

            complexity = self.complexity_metric(generic_tree)

            fitness_dict = {}
            fitness_dict[self.complexity_name] = complexity

            for split in ["train", "val", "test"]:
                y_pred = generic_tree.predict(self.data[f"X_{split}"])
                y = self.data[f"y_{split}"]
                fitness = self.fitness_metric(y, y_pred)
                fitness_dict[f"{self.fitness_name}_{split}"] = fitness

                if split == "train":
                    individual.functional_signature = y_pred
                    if self.lower_is_better:
                        if fitness < best_fitness:
                            best_fitness = fitness
                    else:
                        if fitness > best_fitness:
                            best_fitness = fitness

            individual.fitness = fitness_dict

            new_machine_format = generic_tree.convert_to_dict(return_list=False)
            new_llm_format = parse_dict_to_string(new_machine_format)

            individual.machine_readable_format = new_machine_format
            individual.llm_readable_format = new_llm_format

        if verbose:
            logger.info(
                f"[LLEGO Fitness Evaluator] Best population fitness: {best_fitness:.4f}."
            )
