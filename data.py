from typing import Union, Dict

import pandas as pd


class Data:
    def __init__(
            self,
            data: pd.DataFrame,
            feature_types: Dict[str, str],
    ):
        self.data = data
        self.feature_types = feature_types

    def get_dv_portions(
            self,
            xj: str,
            threshold: Union[int, float] = None
    ) -> Dict[str, pd.DataFrame]:
        # nominal/categorical
        result = {}
        if self.feature_types[xj] == 'NOM':
            for value in self.data[xj].unique():
                result[value] = self.data[self.data[xj] == value]
        # numerical
        else:
            if threshold is not None:
                result = {
                    'above': self.data[self.data[xj] >= threshold],
                    'below': self.data[self.data[xj] < threshold],
                }
            else:
                raise ValueError("The value of threshold can't be None when the feature is numerical.")

        return result

