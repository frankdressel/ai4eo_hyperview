#!/usr/bin/env python3
import pandas
from sklearn.model_selection import train_test_split

def split():
    df = pandas.read_csv('data/merged.csv')

    train, test = train_test_split(df)
    train_x = train.drop(labels=['P','K', 'Mg','pH', 'sample'], axis=1)
    train_y = train[['P','K', 'Mg','pH']]
    test_x = test.drop(labels=['P','K', 'Mg','pH', 'sample'], axis=1)
    test_y = test[['P','K', 'Mg','pH']]

    train_x.to_csv('data/train_x.csv',index=False)
    train_y.to_csv('data/train_y.csv',index=False)
    test_x.to_csv('data/test_x.csv',index=False)
    test_y.to_csv('data/test_y.csv',index=False)

if __name__ == '__main__':
    split()
