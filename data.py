from typing import Any, Union

import pandas as pd


class Data:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def get_dv_xj(self, xj: str, value: Any):
        return self.data[self.data[xj] == value]

    def get_dv_portions(self, xj: str):
        result = {}
        for value in self.data[xj].unique():
            result[value] = self.get_dv_xj(xj, value)

        return result

    def get_dv_portions_threshold(self, xj: str, threshold: Union[int, float]):
        result = {
            'above': self.data[self.data[xj] >= threshold],
            'below': self.data[self.data[xj] < threshold],
        }

        return result
