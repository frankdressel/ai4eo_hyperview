#!/usr/bin/env python3
import joblib
import pandas

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

def fit():
    train_x = pandas.read_csv('data/train_x.csv')
    train_y = pandas.read_csv('data/train_y.csv')
    test_x = pandas.read_csv('data/test_x.csv')
    test_y = pandas.read_csv('data/test_y.csv')

    #regr = MLPRegressor(random_state=42, max_iter=500, hidden_layer_sizes=[len(test_x.columns),100, 10, 100, len(test_x.columns)])
    mlp = MLPRegressor(random_state=42, max_iter=500)
    params = {
            'hidden_layer_sizes': [(10, 10), (20, 20), (50, 50), (100, 100), (10, 10, 10), (20, 20,20)],
            'activation': ['tanh','relu'],
            'solver': ['sgd'],
            'alpha': [0.001, 0.05],
            'learning_rate': ['constant', 'adaptive']
    }
    regr = GridSearchCV(mlp, params, verbose=4)
    regr.fit(train_x, train_y)

    print(regr.best_params_)

    joblib.dump(regr, 'models/mlp')

if __name__ == '__main__':
    fit()
