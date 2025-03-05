from gatree.tree.node import Node

from llego.custom.generic_tree import GenericTree
from llego.custom.parsing_to_string import parse_dict_to_string
from llego.operators.individual import Individual


def convert_gatree_to_individual(gatree: Node, features_list: list) -> Individual:
    """
    Convert a GATree to an Individual object.
    """
    machine_readable_format = convert_gatree_to_dict(gatree, features_list)
    llm_readable_format = parse_dict_to_string(machine_readable_format)

    individual = Individual(
        machine_readable_format=machine_readable_format,
        llm_readable_format=llm_readable_format,
    )

    return individual


def convert_gatree_to_dict(gatree: Node, features_list: list) -> dict:
    if gatree.att_index == -1:
        return {"value": gatree.att_value}
    else:
        feature_name = features_list[gatree.att_index]
        split_value = gatree.att_value

        left = (
            convert_gatree_to_dict(gatree.left, features_list)
            if gatree.left is not None
            else None
        )
        right = (
            convert_gatree_to_dict(gatree.right, features_list)
            if gatree.right is not None
            else None
        )
        dictionary: dict = {feature_name: {}}
        if left is not None:
            dictionary[feature_name][f"> {split_value}"] = left
        if right is not None:
            dictionary[feature_name][f"<= {split_value}"] = right
        return dictionary


def convert_generic_to_gatree(generic_tree: GenericTree, features_list: list) -> Node:
    if generic_tree.child1 is None and generic_tree.child2 is None:
        # then we have reached a leaf, create the gatree Tree
        value = generic_tree.value
        return Node(att_index=-1, att_value=value)

    name_feature = generic_tree.feature
    operator = generic_tree.operator
    split_value = generic_tree.split_value
    # get the index of the feature
    att_index = features_list.index(name_feature)
    if operator == ">" or operator == ">=":
        node = Node(att_index=att_index, att_value=split_value)
        if generic_tree.child1 is not None:
            node_left = convert_generic_to_gatree(generic_tree.child1, features_list)
            node.left = node_left
        if generic_tree.child2 is not None:
            node_right = convert_generic_to_gatree(generic_tree.child2, features_list)
            node.right = node_right
    else:
        node = Node(att_index=att_index, att_value=split_value)

        if generic_tree.child1 is not None:
            node_right = convert_generic_to_gatree(generic_tree.child1, features_list)
            node.right = node_right
        if generic_tree.child2 is not None:
            node_left = convert_generic_to_gatree(generic_tree.child2, features_list)
            node.left = node_left
    return node


def convert_individual_to_gatree(
    individual: Individual, task: str, feature_list: list
) -> Node:
    """
    Convert an Individual to a GATree object.
    """
    machine_readable_format = individual.machine_readable_format
    assert machine_readable_format is not None, "Machine readable format is None."
    generic_tree = GenericTree(task=task)
    generic_tree.create_from_dict(dict_tree=machine_readable_format)
    gatree = convert_generic_to_gatree(generic_tree, feature_list)
    return gatree
