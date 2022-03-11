#!/usr/bin/env python3

import joblib
import numpy
import pandas
import pathlib

from sklearn.preprocessing import StandardScaler

def scale():
    pass

def normalize(arr):
    result = arr.copy()
    sum = 0
    for i in range(len(result)):
        sum = sum + result[i].mean()
    result = result / sum

    return result

def prepare_arr(arr, sample_name):
    data_dict = {}

    data_dict['sample'] = sample_name
    data_dict['size_x'] = len(arr[0])
    data_dict['size_y'] = len(arr[0][0])
    
    arr = normalize(arr)
    for i in range(len(arr)):
        data_dict[f'std_{i}'] = arr[i].std()
        data_dict[f'mean_{i}'] = arr[i].mean()
        data_dict[f'med_{i}'] = numpy.ma.median(arr[i])
    for i in range(len(arr) -1 ):
        data_dict[f'diff_{i}'] = arr[i].mean() - arr[i + 1].mean()

    return data_dict

def merge_test():
    data = pandas.DataFrame()

    test_files = pathlib.Path('test_data').glob('*.npz')
    for tf in test_files:
        with numpy.load(tf) as npz:
            arr = numpy.ma.MaskedArray(**npz)
            sample_name = tf.name.replace('.npz', '')
            data_dict = prepare_arr(arr, sample_name)
            data = pandas.concat([data, pandas.DataFrame.from_dict([data_dict])], ignore_index=True)

    data.drop(columns='sample').to_csv('challenge/test_x.csv', index=False)

def merge_train():
    gt = pandas.read_csv('train_gt.csv', converters={'sample_index': str})
    wavelengths = pandas.read_csv('wavelengths.csv')
    data = pandas.DataFrame()

    train_files = pathlib.Path('train_data').glob('*.npz')
    for tf in train_files:
        with numpy.load(tf) as npz:
            arr = numpy.ma.MaskedArray(**npz)
            sample_name = tf.name.replace('.npz', '')
            data_dict = prepare_arr(arr, sample_name)

            gt_values = gt[gt['sample_index'] == sample_name]
            data_dict['P'] = gt_values['P'].values[0]
            data_dict['K'] = gt_values['K'].values[0]
            data_dict['Mg'] = gt_values['Mg'].values[0]
            data_dict['pH'] = gt_values['pH'].values[0]

            data = pandas.concat([data, pandas.DataFrame.from_dict([data_dict])], ignore_index=True)
    data = data.apply(pandas.to_numeric, errors='ignore')
    for lab in ['P', 'K', 'Mg', 'pH']:
        scaler = StandardScaler()
        scaler.fit(data[[lab]])
        data[[lab]] = scaler.transform(data[[lab]])
        joblib.dump(scaler, f'models/scaler_{lab}')
    data.to_csv('data/merged.csv', index=False)

if __name__== '__main__':
    merge_train()
    merge_test()

