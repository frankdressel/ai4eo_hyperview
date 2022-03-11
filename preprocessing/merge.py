#!/usr/bin/env python3

import joblib
import numpy
import pandas
import pathlib

from sklearn.preprocessing import StandardScaler

def merge():
    gt = pandas.read_csv('train_gt.csv', converters={'sample_index': str})
    wavelengths = pandas.read_csv('wavelengths.csv')
    data = pandas.DataFrame()

    train_files = pathlib.Path('train_data').glob('*.npz')
    for tf in train_files:
        with numpy.load(tf) as npz:
            arr = numpy.ma.MaskedArray(**npz)
            sample_name = tf.name.replace('.npz', '')
            gt_values = gt[gt['sample_index'] == sample_name]
            data_dict = {
                'sample': sample_name,
                'size_x': len(arr[0]),
                'size_y': len(arr[0][0]),
                'P': gt_values['P'].values[0],
                'K': gt_values['K'].values[0],
                'Mg': gt_values['Mg'].values[0],
                'pH': gt_values['pH'].values[0]}
            sum = 0
            for i in range(len(arr)):
                sum = sum + arr[i].mean()
            arr = arr / sum
            for i in range(len(arr)):
                data_dict[f'std_{i}'] = arr[i].std()
                data_dict[f'mean_{i}'] = arr[i].mean()
                data_dict[f'med_{i}'] = numpy.ma.median(arr[i])
            for i in range(len(arr) -1 ):
                data_dict[f'diff_{i}'] = arr[i].mean() - arr[i + 1].mean()
            data = pandas.concat([data, pandas.DataFrame.from_dict([data_dict])], ignore_index=True)
    data = data.apply(pandas.to_numeric, errors='ignore')
    for lab in ['P', 'K', 'Mg', 'pH']:
        scaler = StandardScaler()
        scaler.fit(data[[lab]])
        data[[lab]] = scaler.transform(data[[lab]])
        joblib.dump(scaler, f'models/scaler_{lab}')
    data.to_csv('data/merged.csv', index=False)

if __name__== '__main__':
    merge()
