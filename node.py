from typing import List, Union


class Node:
    def __init__(
            self,
            selected_feature: str,
            feature_type: str,
            children: List['Node'],
            thresholds: Union[List[float], List[int]] = None,
    ):
        self.selected_feature = selected_feature
        self.feature_type = feature_type
        self.children = children
        self.thresholds = thresholds


if __name__ == '__main__':
    pass
