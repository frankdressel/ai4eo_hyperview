#!/usr/bin/env python3
import joblib
import pandas

from sklearn.ensemble import RandomForestRegressor

def fit():
    train_x = pandas.read_csv('data/train_x.csv')
    train_y = pandas.read_csv('data/train_y.csv')
    test_x = pandas.read_csv('data/test_x.csv')
    test_y = pandas.read_csv('data/test_y.csv')

    forest = RandomForestRegressor(max_features=int(len(train_x.columns)/3), oob_score=True)
    forest.fit(train_x, train_y)

    joblib.dump(forest, 'models/randomforest')

if __name__ == '__main__':
    fit()
