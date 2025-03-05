import ast
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

OPERATORS = {
    "<": lambda x, y: x < y,
    "<=": lambda x, y: x <= y,
    ">": lambda x, y: x > y,
    ">=": lambda x, y: x >= y,
}

COMPLEMENTARY_OPERATORS = {
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
}


def parse_split(text: str) -> Tuple[Callable, Union[int, str, float, list], str, str]:
    operator_name, value_str = text.split(" ")

    # Get the operator
    operator = parse_operator(operator_name)

    # Parse the value
    value = parse_value(value_str)

    return operator, value, operator_name, value_str


def parse_operator(text: str) -> Callable:
    assert text in OPERATORS.keys(), f"Operator {text} not recognized"
    return OPERATORS[text]


def parse_value(text: str) -> Union[int, str, float, list]:
    # Check if the value is a list
    try:
        parsed_value = ast.literal_eval(text)
        if isinstance(parsed_value, list):
            # Check if all elements in the list are numbers or strings
            if all(isinstance(x, (int, float, str)) for x in parsed_value):
                return parsed_value
    except (ValueError, SyntaxError):
        pass

    # Check if the value is a float
    try:
        parsed_value = float(text)
        assert isinstance(parsed_value, float)
        return parsed_value
    except ValueError:
        pass

    # Check if the value is an integer
    try:
        parsed_value = int(text)
        assert isinstance(parsed_value, int)
        return parsed_value
    except ValueError:
        pass

    # If none of the above, treat it as a string (name of a feature)
    return text


class GenericTree:
    def __init__(self, task: str = "classification") -> None:
        self.child1: Optional[GenericTree] = None
        self.child2: Optional[GenericTree] = None
        self.feature: Optional[str] = None
        self.depth: int = 0

        self.task = task

        self.operator: Optional[Callable] = None
        self.operator_str: Optional[str] = None
        self.split_value: Optional[Union[int, str, float, list]] = None
        self.split_value_str: Optional[str] = None

        self.value: Optional[Union[str, float, int, bool]] = (
            None  # Value of the node, which is not None if and only if the node is a leaf
        )
        self.value_list: Optional[list] = None
        assert self.value is None or (
            self.child1 is None and self.child2 is None
        ), "A node cannot have children if it is a leaf"

    def predict_single(self, x: pd.DataFrame, return_proba: bool = False):
        if self.task == "regression" and return_proba:
            raise ValueError(
                "Cannot return probabilities for regression task. Please set return_proba=False"
            )

        # Check if the node is a leaf, with the terminal condition
        if self.value is not None:
            if return_proba:
                assert self.value_list is not None, "The value list should be defined"
                # then return the probability
                mean = np.mean(self.value_list)
                return mean

            return float(self.value)
        # Else, recursively predict on the children
        else:
            assert self.operator is not None, "The operator should be defined"
            if self.operator(x[self.feature], self.split_value):
                assert self.child1 is not None, "The child1 should be defined"
                return self.child1.predict_single(x, return_proba=return_proba)
            else:
                assert self.child2 is not None, "The child2 should be defined"
                return self.child2.predict_single(x, return_proba=return_proba)

    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        pred = np.array(
            [
                self.predict_single(x, return_proba=return_proba)
                for index, x in X.iterrows()
            ]
        )
        assert len(pred) == len(
            X
        ), f"Expected {len(X)} predictions but got {len(pred)} predictions"
        assert pred.shape == (len(X),), f"Expected 1D array but got {pred.shape}"
        return pred

    def convert_to_json(self):
        pass

    def construct_tree(self, dict_tree: dict) -> None:
        # Keep the tree in the form of a dict for later printing
        self.dict_tree = dict_tree
        # Base condition
        if "value" in dict_tree.keys():
            self.value_list = []
            self.value = dict_tree[
                "value"
            ]  # default value in case the leaf is empty after fitting
            return

        keys = list(dict_tree.keys())

        # Obtain the feature value
        assert (
            len(keys) == 1
        ), "The dictionary should have only one key, i.e. the root is just one feature on which to split"

        feature = keys[0]

        conditions = list(dict_tree[feature].keys())
        assert (
            len(conditions) <= 2
        ), "The tree should be at most binary, i.e. two children per node"

        split1 = conditions[0]

        operator, split_value, operator_str, split_value_str = parse_split(split1)

        self.feature = feature

        self.operator = operator
        self.operator_str = operator_str
        self.split_value_str = split_value_str
        self.split_value = split_value

        self.child1 = GenericTree(task=self.task)

        self.child1.construct_tree(dict_tree[feature][split1])

        if len(conditions) > 1:
            self.child2 = GenericTree(task=self.task)
            split2 = conditions[1]
            self.child2.construct_tree(dict_tree[feature][split2])

        self.depth = (
            max(self.child1.depth, self.child2.depth) + 1
            if self.child2
            else self.child1.depth + 1
        )

    def populate_leaves(self, x: pd.Series, y: Union[int, float]) -> None:
        # x and y are just one observation

        # add the observation to the leaves
        if self.value_list is not None:
            # If the node is a leaf, add the observations to the leaf
            self.value_list.append(y)
        else:
            assert (
                self.child1 is not None and self.child2 is not None
            ), "The tree is not complete"
            # Else, recursively add the observation to the children
            assert self.operator is not None
            if self.operator(x[self.feature], self.split_value):
                self.child1.populate_leaves(x, y)
            else:
                self.child2.populate_leaves(x, y)

    def compute_leaves_values(self, task: str):
        if self.value_list is not None:
            if len(self.value_list) == 0:
                return self.value
            # Compute the value of the leaf
            if task == "classification":
                # take majority class
                self.value = max(set(self.value_list), key=self.value_list.count)

            elif task == "regression":
                self.value = np.mean(self.value_list)
            else:
                raise ValueError(f"Task {task} not recognized")

        # else we just do it recursively
        else:
            assert self.child1 is not None and self.child2 is not None

            self.child1.compute_leaves_values(task)
            self.child2.compute_leaves_values(task)

    def convert_to_dict(self, return_list: bool = False, precision: int = 4) -> Dict:
        # Convert the tree to a dict
        if self.value_list is not None:
            if return_list:
                return {"value": self.value_list}
            else:

                assert self.value is not None
                leaf_node_value = float(self.value)
                if leaf_node_value == int(leaf_node_value):
                    return {"value": int(leaf_node_value)}
                else:
                    rounded_value = round(leaf_node_value, precision)
                    return {"value": rounded_value}
        else:
            assert self.child1 is not None and self.child2 is not None
            assert self.operator_str is not None
            return {
                self.feature: {
                    f"{self.operator_str} {self.split_value_str}": self.child1.convert_to_dict(
                        return_list=return_list
                    ),
                    f"{COMPLEMENTARY_OPERATORS[self.operator_str]} {self.split_value_str}": self.child2.convert_to_dict(
                        return_list=return_list
                    ),
                }
            }

    def create_from_dict(
        self,
        dict_tree: Dict,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[np.ndarray] = None,
    ) -> None:
        # Construct the tree
        self.construct_tree(dict_tree)

        if X_train is None or y_train is None:
            return
        # Now change the values of the leaves
        for i in range(len(X_train)):
            x = X_train.iloc[i]
            y = y_train[i]
            self.populate_leaves(x, y)

        # compute the values of the leaves
        self.compute_leaves_values(self.task)

    def __str__(self) -> str:
        # Pretty print the tree with yaml
        return yaml.dump(self.dict_tree)
