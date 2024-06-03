from typing import Union, Dict, TypeVar

import pandas as pd

import CONSTANTS

T = TypeVar('T', bound='Data')


class Data:
    def __init__(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_types: Dict[str, str],
    ):
        self.X = X
        self.y = y
        self.feature_types = feature_types

    def __len__(self):
        return len(self.y)

    def get_feature_names(self):
        return self.X.columns.tolist()

    def get_dv_portions(
            self,
            xj: str,
            threshold: Union[int, float] = None
    ) -> Dict[str, T]:
        # nominal/categorical
        result = {}
        if self.feature_types[xj] == CONSTANTS.NOMINAL:
            for value in self.X[xj].unique():
                X_subset = self.X[self.X[xj] == value]
                y_subset = self.y[X_subset.index]

                result[value] = Data(X_subset, y_subset, self.feature_types.copy())

        # TODO numeric [done]
        else:
            if threshold is not None:
                X_subset_above = self.X[self.X[xj] >= threshold]
                y_subset_above = self.y[X_subset_above.index]

                X_subset_below = self.X[self.X[xj] < threshold]
                y_subset_below = self.y[X_subset_below.index]

                result = {
                    'above': Data(
                        X_subset_above,
                        y_subset_above,
                        self.feature_types,
                    ),
                    'below': Data(
                        X_subset_below,
                        y_subset_below,
                        self.feature_types,
                    ),
                }
            else:
                raise ValueError("The value of threshold can't be None when the feature is numeric.")

        return result

    def remove_feature(self, feature_name: str):
        self.X = self.X.drop(feature_name, axis=1)
        del self.feature_types[feature_name]
