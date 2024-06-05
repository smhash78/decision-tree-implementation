from time import time

import pandas as pd
from sklearn.model_selection import train_test_split

import CONSTANTS

from data import Data
from decision_tree_id3 import DecisionTreeID3


def train_and_print_test_results(
        train_data: Data,
        test_data: Data,
        sample_number: int,
) -> None:
    print(f"Sample data {sample_number}, using information gain:")

    decision_tree_ig = DecisionTreeID3()
    decision_tree_ig.train(train_data)

    decision_tree_ig.print_tree()
    print(f"Accuracy: {decision_tree_ig.test(test_data)[CONSTANTS.ACCURACY]}\n")

    print(f"Sample data {sample_number}, using gain ratio:")

    decision_tree_gr = DecisionTreeID3()
    decision_tree_gr.train(train_data)

    decision_tree_gr.print_tree()
    print(f"Accuracy: {decision_tree_gr.test(test_data)[CONSTANTS.ACCURACY]}\n")


def run_test_data_1():
    df = pd.DataFrame({
        'A': ['F', 'T', 'T', 'T'],
        'B': ['F', 'F', 'T', 'T'],
        'C': ['F', 'T', 'T', 'F'],
        'Y': ['F', 'T', 'F', 'T'],
    })

    X = df[['A', 'B', 'C']]
    y = df['Y']

    data = Data(
        X,
        y,
        {
            'A': CONSTANTS.NOMINAL,
            'B': CONSTANTS.NOMINAL,
            'C': CONSTANTS.NOMINAL,
        }
    )

    train_and_print_test_results(data, data, 1)


def run_test_data_2():
    dataset = [
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

    df = pd.DataFrame(dataset, columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play Tennis?'])

    X = df[['Outlook', 'Temperature', 'Humidity', 'Wind']]
    y = df['Play Tennis?']

    data = Data(
        X,
        y,
        {
            'Outlook': CONSTANTS.NOMINAL,
            'Temperature': CONSTANTS.NOMINAL,
            'Humidity': CONSTANTS.NOMINAL,
            'Wind': CONSTANTS.NOMINAL,
        }
    )

    train_and_print_test_results(data, data, 2)


def run_test_data_3():
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df = pd.read_csv('./data/iris.data', header=None, names=column_names)

    X = df[column_names[:-1]]
    y = df[column_names[-1]]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    X_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    train_data = Data(
        X_train,
        y_train,
        {key: CONSTANTS.NUMERIC for key in column_names[:-1]}
    )

    test_data = Data(
        X_test,
        y_test,
        {key: CONSTANTS.NUMERIC for key in column_names[:-1]}
    )

    train_and_print_test_results(train_data, test_data, 3)


if __name__ == '__main__':
    now = time()
    run_test_data_1()
    print(f"It took {time() - now} seconds.")
    print()

    now = time()
    run_test_data_2()
    print(f"It took {time() - now} seconds.")
    print()

    now = time()
    run_test_data_3()
    print(f"It took {time() - now} seconds.")
    print("\n\n")
