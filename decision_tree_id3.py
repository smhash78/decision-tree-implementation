import pandas as pd

from data import Data


def choose_best_feature(
        X: Data,
        y: pd.Series,
        method: str = 'IG',
):
    pass


def construct_tree(
        X: Data,
        y: pd.Series,
        method: str = 'IG',
):
    # best_feature, thresholds = choose_best_feature(X, y, method)
    #
    # for i, column_name in enumerate(X.columns):
    #     ig = information_gain(X, column_name)
    pass


class DecisionTreeID3:
    def __init__(self):
        self.tree = None

    def train(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            method: str = 'IG',
    ):
        self.tree = construct_tree(X, y, method)
