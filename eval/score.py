#!/usr/bin/env python3

import pandas

def mse(data, ref):
    return ((data - ref)**2).mean()

def score():
    data = pandas.read_csv('data/merged.csv')[['P', 'K', 'Mg', 'pH']]
    pred_forest = pandas.read_csv('data/randomforest.csv')
    pred_mlp = pandas.read_csv('data/mlp.csv')
    test_y = pandas.read_csv('data/test_y.csv')

    mse_base = mse(data, data.mean())
    mse_pred_forest = mse(pred_forest, test_y)
    mse_pred_mlp = mse(pred_mlp, test_y)

    print(mse_base)
    print('Forest')
    print(mse_pred_forest)
    print((mse_pred_forest / mse_base).mean())
    print('MLP')
    print(mse_pred_mlp)
    print((mse_pred_mlp / mse_base).mean())

if __name__ == '__main__':
    score()
