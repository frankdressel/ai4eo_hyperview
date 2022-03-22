import joblib
import luigi
import os
import numpy
import pandas
import pathlib
import tempfile

from ai4eo_hyperview.utils import MTimeMixin
from scipy.interpolate import UnivariateSpline 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class MergeData(MTimeMixin, luigi.Task):
    def output(self):
        return {
                'merged': luigi.LocalTarget('data/merged.csv'),
                'challenge': luigi.LocalTarget('data/challenge.csv'),
        }

    @staticmethod
    def _normalize(arr):
        result = arr.copy()
        sum = 0
        for i in range(len(result)):
            sum = sum + result[i].mean()
        result = result / sum

        return result

    @staticmethod
    def _prepare_arr(arr, sample_name):
        data_dict = {}

        data_dict['sample'] = sample_name
        data_dict['size_x'] = len(arr[0])
        data_dict['size_y'] = len(arr[0][0])
        
        means = []
        for i in range(len(arr)):
            means.append(arr[i].mean() / (arr.sum() / arr.mask[0].sum()))
        smeans = means
        for i in range(len(smeans)):
            data_dict[f'mean_{i}'] = smeans[i]
        for i in range(len(smeans)):
            if i > 0:
                data_dict[f'd{i}'] = smeans[i] - smeans[i - 1]
        data_dict['average_refl'] = arr.sum() / arr.mask[0].sum()

        return data_dict

    def run(self):
        # train data
        gt = pandas.read_csv('train_data/train_gt.csv', converters={'sample_index': str})
        _data = []

        train_files = pathlib.Path('train_data/train_data').glob('*.npz')
        for tf in train_files:
            with numpy.load(tf) as npz:
                arr = numpy.ma.MaskedArray(**npz)
                sample_name = tf.name.replace('.npz', '')
                data_dict = MergeData._prepare_arr(arr, sample_name)

                gt_values = gt[gt['sample_index'] == sample_name]
                data_dict['P'] = gt_values['P'].values[0]
                data_dict['K'] = gt_values['K'].values[0]
                data_dict['Mg'] = gt_values['Mg'].values[0]
                data_dict['pH'] = gt_values['pH'].values[0]

                _data.append(data_dict)
        data = pandas.DataFrame.from_dict(_data)
        data = data.apply(pandas.to_numeric, errors='ignore')

        with self.output()['merged'].open('w') as f:
            data.to_csv(f, index=False)

        # Challenge test data
        _data = []

        test_files = pathlib.Path('test_data').glob('*.npz')
        for tf in test_files:
            with numpy.load(tf) as npz:
                arr = numpy.ma.MaskedArray(**npz)
                sample_name = tf.name.replace('.npz', '')
                data_dict = MergeData._prepare_arr(arr, sample_name)

                _data.append(data_dict)
        data = pandas.DataFrame.from_dict(_data)
        data = data.apply(pandas.to_numeric, errors='ignore')

        with self.output()['challenge'].open('w') as f:
            data.to_csv(f, index=False)

class Split(MTimeMixin, luigi.Task):
    def requires(self):
        return {'Merge': MergeData()}

    def output(self):
        return {
                'train_x': luigi.LocalTarget('data/train_x.csv'),
                'train_y': luigi.LocalTarget('data/train_y.csv'),
                'test_x': luigi.LocalTarget('data/test_x.csv'),
                'test_y': luigi.LocalTarget('data/test_y.csv'),
        }

    def run(self):
        with self.input()['Merge']['merged'].open('r') as f:
            df = pandas.read_csv(f)

        train, test = train_test_split(df)
        train_x = train.drop(labels=['P', 'K', 'Mg','pH'], axis=1)
        train_y = train[['P', 'K', 'Mg','pH', 'sample']]
        test_x = test.drop(labels=['P', 'K', 'Mg','pH'], axis=1)
        test_y = test[['P', 'K', 'Mg','pH', 'sample']]

        with self.output()['train_x'].open('w') as f:
            train_x.to_csv(f, index=False)
        with self.output()['train_y'].open('w') as f:
            train_y.to_csv(f, index=False)
        with self.output()['test_x'].open('w') as f:
            test_x.to_csv(f, index=False)
        with self.output()['test_y'].open('w') as f:
            test_y.to_csv(f, index=False)
