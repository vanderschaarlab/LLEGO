from typing import Dict


def parse_dict_to_string(tree_dict: Dict) -> str:
    """
    Parse a dictionary to a string, which is ingested by LLM
    Args:
        tree_dict: dictionary representation of a tree
    Return:
        serialized_tree: string representation of the tree
    """

    serialized_tree = str(tree_dict)

    ## This is to escape the curly braces
    serialized_tree = serialized_tree.replace("{", "{{").replace("}", "}}")

    return serialized_tree
