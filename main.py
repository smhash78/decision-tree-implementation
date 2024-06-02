from time import time

import pandas as pd

import CONSTANTS

from data import Data
from decision_tree_id3 import DecisionTreeID3


def run_test_data_1():
    df = pd.DataFrame({
        'A': ['F', 'T', 'T', 'T'],
        'B': ['F', 'F', 'T', 'T'],
        'C': ['F', 'T', 'T', 'F'],
        'Y': ['F', 'T', 'F', 'T'],
    })

    X = df[['A', 'B', 'C']]
    y = df['Y']

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


def run_test_data_2():
    data = [
        ['Sunny', 'Hot', 'High', 'Light', 'No'],
        ['Sunny', 'Hot', 'High', 'Strong', 'No'],
        ['Overcast', 'Hot', 'High', 'Light', 'Yes'],
        ['Rain', 'Mild', 'High', 'Light', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Light', 'Yes'],
        ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
        ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
        ['Sunny', 'Mild', 'High', 'Light', 'No'],
        ['Sunny', 'Cool', 'Normal', 'Light', 'Yes'],
        ['Rain', 'Mild', 'Normal', 'Light', 'Yes'],
        ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
        ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
        ['Overcast', 'Hot', 'Normal', 'Light', 'Yes'],
        ['Rain', 'Mild', 'High', 'Strong', 'No']
    ]

    df = pd.DataFrame(data, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play Tennis?'])

    X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
    y = df['Play Tennis?']

    decision_tree = DecisionTreeID3()
    decision_tree.train(
        Data(
            X,
            y,
            {
                'Outlook': CONSTANTS.NOMINAL,
                'Temperature': CONSTANTS.NOMINAL,
                'Humidity': CONSTANTS.NOMINAL,
                'Wind': CONSTANTS.NOMINAL,
            }
        )
    )

    decision_tree.print_tree()


if __name__ == '__main__':
    now = time()
    run_test_data_1()
    print(f"It took {time() - now} seconds.")
    print()

    now = time()
    run_test_data_2()
    print(f"It took {time() - now} seconds.")
