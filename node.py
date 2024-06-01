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
        self.feature_values = feature_values
        self.children = children
        self.threshold = threshold

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

    def run_for_data(self, data: Data):
        return data.get_dv_portions(self.selected_feature, self.threshold)


class LeafNode:
    def __init__(
            self,
            label: Any = None,
    ):
        self.label = label
