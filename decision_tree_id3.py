from math import log2
from typing import Tuple, Union

import CONSTANTS
from data import Data


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
    if data.X.feature_types[xj] == CONSTANTS.nominal:
        return get_information_gain(data, xj), None

    # numeric
    else:
        pass


def choose_best_feature(
        data: Data,
        method: str = 'IG',
) -> Tuple[str, Union[int, float]]:
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


def construct_tree(
        data: Data,
        method: str = 'IG',
):
    # best_feature, threshold = choose_best_feature(data, method)
    #
    # for i, column_name in enumerate(X.columns):
    #     ig = information_gain(X, column_name)
    pass


class DecisionTreeID3:
    def __init__(self):
        self.tree = None

    def train(
            self,
            data: Data,
            method: str = 'IG',
    ):
        self.tree = construct_tree(data, method)
