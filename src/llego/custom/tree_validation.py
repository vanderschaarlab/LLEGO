import ast
from typing import Dict, Optional, Union

from llego.operators.individual import Individual


def validate_tree_dict(node: Dict) -> bool:
    """
    1. Check that splits are binary
    2. Check that split conditions are complementary
    3. Check that leaf nodes have {value: v}
    """

    # helper to checks if two conditions are complementary
    def are_complementary(cond1, cond2):
        if cond1.startswith("<=") and cond2.startswith(">"):
            return cond1[2:].strip() == cond2[1:].strip()
        elif cond2.startswith("<=") and cond1.startswith(">"):
            return cond2[2:].strip() == cond1[1:].strip()
        elif cond1.startswith(">=") and cond2.startswith("<"):
            return cond1[2:].strip() == cond2[1:].strip()
        elif cond2.startswith(">=") and cond1.startswith("<"):
            return cond2[2:].strip() == cond1[1:].strip()
        return False

    # check if this is a leaf node
    if isinstance(node, dict) and "value" in node:
        return True

    if isinstance(node, dict) and len(node) != 1:
        raise ValueError(
            f"Each internal node must exactly split into two branches {node.keys()}"
        )

    # get the feature and the branches
    feature, branches = next(iter(node.items()))
    conditions = list(branches.keys())

    # check if there are exactly two branches
    if len(conditions) != 2:
        raise ValueError(
            f"Each internal node must have exactly two complementary branches, {conditions}"
        )

    # check if the conditions are complementary
    if not are_complementary(conditions[0], conditions[1]):
        raise ValueError(
            f"The conditions {conditions[0]} and {conditions[1]} are not complementary"
        )

    # recursively validate each branch
    for condition in conditions:
        validate_tree_dict(branches[condition])

    return True


def get_dict_depth(d: Dict) -> int:
    """
    Get the depth of a dict
    """
    if isinstance(d, dict):
        return 1 + (max(get_dict_depth(v) for v in d.values()) if d.values() else 0)
    return 0


def get_dt_depth(tree_dict: Dict) -> int:
    """
    Get the depth of a decision tree represented as a dict
    """
    assert isinstance(tree_dict, dict)
    dict_depth = get_dict_depth(tree_dict)
    tree_depth = (dict_depth - 1) / 2
    assert tree_depth == int(tree_depth)

    return int(tree_depth)


def check_format_consistency(individual: Individual) -> bool:
    """
    Check if the machine and llm readable formats of an individual are consistent
    """
    assert isinstance(individual, Individual)
    assert isinstance(individual.machine_readable_format, dict)
    assert isinstance(individual.llm_readable_format, str)

    # check #1: assert llm readable format escaped by double brackets
    parsed_string = individual.llm_readable_format.replace("{{", "{").replace("}}", "}")
    assert (
        parsed_string != individual.llm_readable_format
    ), "LLM readable format not escaped by double brackets"

    # check #2: assert llm readable format can be parsed back to a dict
    parsed_dict = ast.literal_eval(parsed_string)
    assert (
        parsed_dict == individual.machine_readable_format
    ), "LLM readable format does not match machine readable format"

    return True


def validate_individual(
    individual: Individual, max_depth: Optional[int] = None
) -> bool:
    """
    Check if the individual is valid
    """
    assert isinstance(individual, Individual)
    assert isinstance(individual.machine_readable_format, dict)
    assert isinstance(individual.llm_readable_format, str)

    # check if the machine readable format is a valid tree dict
    assert validate_tree_dict(individual.machine_readable_format)

    # check if the machine and llm readable formats are consistent
    assert check_format_consistency(individual)

    if max_depth is not None:
        assert (
            get_dt_depth(individual.machine_readable_format) <= max_depth
        ), f"Tree depth exceeds {max_depth}"

    return True


def are_attributes_valid(nested_dict: dict, valid_attributes_list: list) -> bool:
    """
    Check if all the attributes in a nested dict are valid
    """
    for key in nested_dict:
        # skips the keys that are condition (contain "<=", ">", ">=", "<") or value
        if (
            "<=" not in key
            and "<" not in key
            and ">=" not in key
            and ">" not in key
            and key != "value"
        ):
            # check if the attribute key is not in the list
            if key not in valid_attributes_list:
                print(f"{key} not a valid attribute")
                return False

        # recurse
        if isinstance(nested_dict[key], dict):
            if not are_attributes_valid(nested_dict[key], valid_attributes_list):
                return False

    return True
