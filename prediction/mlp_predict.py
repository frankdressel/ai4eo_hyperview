#!/usr/bin/env python3
import joblib
import pandas
import pathlib
import sys

def predict(path_to_x: pathlib.Path):
    mlp = joblib.load('models/mlp')
    test_x = pandas.read_csv(path_to_x)

    prediction = pandas.DataFrame(mlp.predict(test_x), columns=['P', 'K', 'Mg', 'pH'])

    prediction.to_csv('data/mlp.csv', index=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(1)
    if not pathlib.Path(sys.argv[1]).exists():
        sys.exit(2)
    predict(pathlib.Path(sys.argv[1]))
