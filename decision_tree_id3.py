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
    for _, dv in data.get_dv_portions(xj, threshold).items():
        result += len(dv) / len(data) * get_entropy(dv)

    return result


def get_information_gain(
        data: Data,
        xj: str,
        threshold: Union[int, float, None] = None,
) -> float:
    return get_entropy(data) - get_conditional_entropy(data, xj, threshold)


def find_best_threshold_ig(
        data: Data,
        xj: str,
) -> Tuple[float, float]:
    unique_values = data.X[xj].unique()
    if len(unique_values) == 1:
        thresholds = unique_values
    else:
        thresholds = [
            (unique_values[i] + unique_values[i + 1]) / 2
            for i in range(len(unique_values) - 1)
        ]

    best_ig = -1
    best_threshold = None

    for threshold in thresholds:
        ig = get_information_gain(data, xj, threshold)
        if ig > best_ig:
            best_ig = ig
            best_threshold = threshold

    return best_threshold, best_ig


def get_best_information_gain(
        data: Data,
        xj: str,
) -> Tuple[float, Union[int, float, None]]:
    # nominal/categorical
    if data.feature_types[xj] == CONSTANTS.NOMINAL:
        return get_information_gain(data, xj), None

    # TODO numeric [done]
    else:
        threshold, ig = find_best_threshold_ig(data, xj)

        return ig, threshold


def get_best_gain(
        data: Data,
        xj: str,
        method: str = CONSTANTS.IG,
) -> Tuple[float, Union[int, float, None]]:
    # information gain
    if method == CONSTANTS.IG:
        return get_best_information_gain(data, xj)

    # TODO gain ration
    elif method == CONSTANTS.GR:
        pass


def choose_best_feature(
        data: Data,
        method: str = CONSTANTS.IG,
) -> Tuple[Union[str, None], Union[int, float, None]]:
    if method == CONSTANTS.IG:
        best_gain = -1
        best_feature = None
        threshold = None

        for xj in data.get_feature_names():
            ig, t = get_best_information_gain(data, xj)

            if ig > best_gain:
                best_gain = ig
                best_feature = xj
                threshold = t

        # when going deeper doesn't help
        if best_gain == 0:
            return None, None

        return best_feature, threshold


def construct_tree(
        data: Data,
        method: str = 'IG',
) -> Union[Node, LeafNode]:
    # all data sorted correctly
    if len(data.y.unique()) == 1 or len(data.y) == 1:
        return LeafNode(data.y.iloc[0])

    # no feature left
    elif data.X.shape[1] == 0:
        return LeafNode(data.y.mode()[0])

    best_feature, threshold = choose_best_feature(data, method)

    # none of the features is useful
    if best_feature is None:
        return LeafNode(data.y.mode()[0])

    # # nominal
    # if threshold is None:

    feature_values = data.feature_types[best_feature] if threshold is None else None

    node = Node(
        selected_feature=best_feature,
        feature_type=data.feature_types[best_feature],
        feature_values=feature_values,
        threshold=threshold,
    )
    split_data = data.get_dv_portions(best_feature, threshold)

    for key, value in split_data.items():
        # nominal/categorical
        if threshold is None:
            value.remove_feature(best_feature)
        node.children[key] = construct_tree(value, method)

    return node

    # # TODO [check] numeric [done]
    # else:
    #     return Node(
    #         selected_feature=best_feature,
    #         feature_type=data.feature_types[best_feature],
    #         threshold=threshold
    #     )


class DecisionTreeID3:
    def __init__(self):
        self.tree = None

    def train(
            self,
            data: Data,
            method: str = CONSTANTS.IG,
    ):
        self.tree = construct_tree(data, method)

    def predict(self, data_point: pd.Series):
        current_node = self.tree

        while not isinstance(current_node, LeafNode):
            current_node = current_node.run_for_point(data_point)

        return current_node.label

    def print_tree(self):
        self.tree.print_node(0)
