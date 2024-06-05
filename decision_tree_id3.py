from math import log2
from typing import Tuple, Union, List

import pandas as pd

import CONSTANTS
from data import Data
from node import LeafNode, Node


def calculate_entropy(
        data: Data,
) -> float:
    result = 0.0
    for label in data.y.unique():
        p = data.y.value_counts()[label] / len(data.y)
        result -= p * log2(p)

    return result


def calculate_conditional_entropy(
        data: Data,
        xj: str,
        threshold: Union[int, float, None] = None,
) -> float:
    result = 0.0
    for _, dv in data.get_dv_portions(xj, threshold).items():
        result += len(dv) / len(data) * calculate_entropy(dv)

    return result


def calculate_information_gain(
        data: Data,
        xj: str,
        threshold: Union[int, float, None] = None,
) -> float:
    return calculate_entropy(data) \
           - calculate_conditional_entropy(data, xj, threshold)


def calculate_split_info(
        data: Data,
        xj: str,
        threshold: Union[int, float, None] = None,
) -> float:
    result = 0.0
    for _, dv in data.get_dv_portions(xj, threshold).items():
        ratio = len(dv) / len(data)
        result -= ratio * log2(ratio)

    return result


def calculate_gain_ration(
        data: Data,
        xj: str,
        threshold: Union[int, float, None] = None,
) -> float:
    return calculate_information_gain(data, xj, threshold) \
           - calculate_split_info(data, xj, threshold)


def calculate_gain(
        data: Data,
        xj: str,
        threshold: Union[int, float, None] = None,
        method: str = CONSTANTS.IG,
) -> float:
    if method == CONSTANTS.IG:
        return calculate_information_gain(data, xj, threshold)

    elif method == CONSTANTS.GR:
        return calculate_gain_ration(data, xj, threshold)


def find_best_threshold_gain(
        data: Data,
        xj: str,
        method: str = CONSTANTS.IG,
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
        ig = calculate_gain(data, xj, threshold, method)
        if ig > best_ig:
            best_ig = ig
            best_threshold = threshold

    return best_threshold, best_ig


def calculate_best_gain(
        data: Data,
        xj: str,
        method: str = CONSTANTS.IG,
) -> Tuple[float, Union[int, float, None]]:
    if data.feature_types[xj] == CONSTANTS.NOMINAL:
        return calculate_gain(data, xj, method=method), None

    elif data.feature_types[xj] == CONSTANTS.NUMERIC:
        threshold, ig = find_best_threshold_gain(data, xj, method)

        return ig, threshold


def choose_best_feature(
        data: Data,
        method: str = CONSTANTS.IG,
) -> Tuple[Union[str, None], Union[int, float, None]]:
    if method == CONSTANTS.IG:
        best_gain = -1
        best_feature = None
        threshold = None

        for xj in data.get_feature_names():
            ig, t = calculate_best_gain(data, xj, method)

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

    feature_values = data.feature_types[best_feature] \
        if threshold is None \
        else None

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


class DecisionTreeID3:
    def __init__(self):
        self.tree = None

    def train(
            self,
            data: Data,
            method: str = CONSTANTS.IG,
    ):
        self.tree = construct_tree(data, method)

    def test(
            self,
            test_data: Data,
            evaluation_metrics: Union[List[str], None] = None,
    ):
        if evaluation_metrics is None:
            evaluation_metrics = [CONSTANTS.ACCURACY]

        true_predictions = 0
        for i, row in test_data.X.iterrows():
            prediction = self.predict(row)
            if prediction == test_data.y.iloc[i]:
                true_predictions += 1

        results = {
            CONSTANTS.ACCURACY: true_predictions / len(test_data),
        }
        return {
            key: value
            for key, value in results.items()
            if key in evaluation_metrics
        }

    def predict(self, data_point: pd.Series):
        current_node = self.tree

        while not isinstance(current_node, LeafNode):
            current_node = current_node.run_for_point(data_point)

        return current_node.label

    def print_tree(self):
        self.tree.print_node(0)
