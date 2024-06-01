from typing import List, Any

import pandas as pd

import CONSTANTS
from data import Data


class Node:
    def __init__(
            self,
            selected_feature: str,
            feature_type: str,
            children_separators: List[Any],
            children: List['Node'],
            threshold: float = None,
    ):
        self.selected_feature = selected_feature
        self.feature_type = feature_type
        self.children_separators = children_separators
        self.children = children
        self.threshold = threshold

    def run(self, data: Data):
        return data.get_dv_portions(self.selected_feature, self.threshold)

