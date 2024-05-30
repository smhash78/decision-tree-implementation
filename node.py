from typing import List, Union, Any

import pandas as pd


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

    def run(self, X: pd.DataFrame):
        split_data = {}

        if self.feature_type == 'NOM':
            grouped = X.groupby(self.selected_feature)

            for category, group in grouped:
                split_data[category] = group

        else:
            split_data['above'] = X[X[self.selected_feature] >= self.threshold]
            split_data['below'] = X[X[self.selected_feature] < self.threshold]

        return split_data


if __name__ == '__main__':
    pass
