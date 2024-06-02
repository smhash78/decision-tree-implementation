import numpy as np
import pandas as pd

import CONSTANTS

from data import Data
from decision_tree_id3 import DecisionTreeID3

if __name__ == '__main__':

    # Create the dataframe
    df = pd.DataFrame({
        'A': ['F'] + ['T'] * 3,
        'B': ['F', 'F', 'T', 'T'],
        'C': ['T', 'T', 'F', 'F'],
        'target': ['F', 'T'] * 2,
    })

    # Split into X and y
    X = df[['A', 'B', 'C']]
    y = df['target']

    decision_tree = DecisionTreeID3()

    decision_tree.train(

        Data(
            X,
            y,
            {
                'A': CONSTANTS.NOMINAL,
                'B': CONSTANTS.NOMINAL,
                'C': CONSTANTS.NOMINAL,
            }
        )
    )

    decision_tree.print_tree()
