#!/usr/bin/env python3

import joblib
import numpy
import pandas
import pathlib

from sklearn.preprocessing import StandardScaler

def retransform():
    data = pandas.read_csv('data/merged.csv')[['P', 'K', 'Mg', 'pH']]
    data_forest = pandas.read_csv('data/randomforest.csv')
    data_forest_challenge = pandas.read_csv('challenge/randomforest.csv')
    test_y = pandas.read_csv('data/test_y.csv')
    for lab in ['P', 'K', 'Mg', 'pH']:
        scaler = joblib.load(f'models/scaler_{lab}')
        data_forest[[lab]] = scaler.inverse_transform(data_forest[[lab]])
        data_forest_challenge[[lab]] = scaler.inverse_transform(data_forest_challenge[[lab]])
        data[[lab]] = scaler.inverse_transform(data[[lab]])
        test_y[[lab]] = scaler.inverse_transform(test_y[[lab]])
    data_forest.to_csv('data/randomforest_final.csv', index=False)
    data_forest_challenge.to_csv('challenge/randomforest_final.csv', index=False)
    data.to_csv('data/data_final.csv', index=False)
    test_y.to_csv('data/test_y_final.csv', index=False)

if __name__== '__main__':
    retransform()
