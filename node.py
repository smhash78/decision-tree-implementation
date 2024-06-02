from typing import List, Any, Union, Dict

import pandas as pd

import CONSTANTS
from data import Data


class Node:
    def __init__(
            self,
            selected_feature: str,
            feature_type: str,
            feature_values: Union[List[Any], None] = None,
            children: Union[Dict[Any, 'Node'], None] = None,
            threshold: float = None,
    ):
        self.selected_feature = selected_feature
        self.feature_type = feature_type

        if feature_values is None:
            self.feature_values = []
        else:
            self.feature_values = feature_values

        if children is None:
            self.children = {}
        else:
            self.children = children

        self.threshold = threshold

    def print_node(self, layer: int = 0, last_feature_value: Any = 'root'):
        indentation = '\t' * layer
        print(f"{indentation}#({last_feature_value} -> {self.selected_feature}: {self.feature_values})#")
        for key, child in self.children.items():
            child.print_node(layer + 1, key)

    def run_for_point(self, data_point: pd.Series) -> 'Node':
        # nominal/categorical
        if self.feature_type == CONSTANTS.NOMINAL:
            return self.children[data_point[self.selected_feature]]

        # numeric
        else:
            feature_value = data_point[self.selected_feature]

            if feature_value >= self.threshold:
                return self.children['above']

            else:
                return self.children['below']


class LeafNode:
    def __init__(
            self,
            label: Any = None,
    ):
        self.label = label

    def print_node(self, layer: int, last_feature_value: Any):
        indentation = '\t' * layer
        print(f"{indentation}#({last_feature_value} -> label: {self.label})#")
