#!/usr/bin/env python3
import joblib
import os
import pandas
import pathlib
import sys

def predict(path_to_x: pathlib.Path):
    forest = joblib.load('models/randomforest')
    test_x = pandas.read_csv(path_to_x)

    prediction = pandas.DataFrame(forest.predict(test_x), columns=['P', 'K', 'Mg', 'pH'])

    prediction.to_csv(os.path.join(path_to_x.parent, 'randomforest.csv'), index=False)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(1)
    if not pathlib.Path(sys.argv[1]).exists():
        sys.exit(2)
    predict(pathlib.Path(sys.argv[1]))
