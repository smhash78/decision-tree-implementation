from math import log2
from typing import Tuple, Union

import pandas as pd

import CONSTANTS
from data import Data
from node import LeafNode, Node


def get_entropy(
        data: Data,
) -> float:
    result = 0.0
    for label in data.y.unique():
        p = data.y.value_counts()[label] / len(data.y)
        result -= p * log2(p)

    return result


def get_conditional_entropy(
        data: Data,
        xj: str,
        threshold: Union[int, float, None] = None,
) -> float:
    result = 0.0
    for dv in data.get_dv_portions(xj, threshold):
        result += len(dv) / len(data) * get_entropy(dv)

    return result


def get_information_gain(
        data: Data,
        xj: str,
        threshold: Union[int, float, None] = None,
) -> float:
    return get_entropy(data) - get_conditional_entropy(data, xj, threshold)


def get_best_information_gain(
        data: Data,
        xj: str,
        threshold: Union[int, float]
) -> Tuple[float, Union[int, float, None]]:
    # nominal/categorical
    if data.X.feature_types[xj] == CONSTANTS.NOMINAL:
        return get_information_gain(data, xj), None

    # numeric
    else:
        pass


def choose_best_feature(
        data: Data,
        method: str = 'IG',
) -> Tuple[str, Union[int, float, None]]:
    if method == 'IG':
        best_information_gain = -1
        best_feature = None
        threshold = None

        for xj in data.X.get_feature_names():
            ig, t = get_best_information_gain(data, xj)

            if ig > best_information_gain:
                best_information_gain = ig
                best_feature = xj
                threshold = t

        return best_feature, threshold


def create_node(
        data: Data,
        method: str = 'IG',
) -> Union[Node, LeafNode]:
    if len(data.y.unique()) == 1:
        return LeafNode(data.y[0])

    best_feature, threshold = choose_best_feature(data, method)

    # nominal
    if threshold is None:
        return Node(
            selected_feature=best_feature,
            feature_type=data.feature_types[best_feature],
            feature_values=data.X[best_feature].unique(),
        )

    # numeric
    else:
        return Node(
            selected_feature=best_feature,
            feature_type=data.feature_types[best_feature],
            threshold=threshold
        )


def construct_tree(
        data: Data,
        method: str = 'IG',
) -> Union[Node, LeafNode]:
    new_node = create_node(data)

    if isinstance(new_node, LeafNode):
        return new_node

    split_data = new_node.run_for_data(data)

    for key, value in split_data.items():
        new_node.children[key] = construct_tree(value)

    return new_node


class DecisionTreeID3:
    def __init__(self):
        self.tree = None

    def train(
            self,
            data: Data,
            method: str = 'IG',
    ):
        self.tree = construct_tree(data, method)

    def predict(self, data_point: pd.Series):
        current_node = self.tree

        while not isinstance(current_node, LeafNode):
            current_node = current_node.run_for_point(data_point)

        return current_node.label
