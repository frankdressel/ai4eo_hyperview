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
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import FunctionTransformer

class MergeData(MTimeMixin, luigi.Task):
    def output(self):
        return {
                'merged': luigi.LocalTarget('data/merged.csv'),
                'challenge': luigi.LocalTarget('data/challenge.csv'),
        }

    @staticmethod
    def transf(df):
        s = df.filter(regex='^ddmean.*').sum().values
        ind = []
        for i in range(1, len(s)):
            if s[i] == 0 or (s[i-1] > 0 and s[i] < 0) or (s[i-1] < 0 and s[i] > 0):
                ind.append(i)
        res = df.copy()
        for i in ind:
            for j in ind:
                if i != j:
                    res[f'{i}/{j}'] = res[f'mean_{i}']/res[f'mean_{j}']
                    #for k in ind:
                    #    if k != i and k != j:
                    #        res[f'{i}/{j}+{k}'] = res[f'mean_{i}']/(res[f'mean_{j}'] + res[f'mean_{k}'])
        return res

    @staticmethod
    def _prepare_arr(arr, sample_name):
        data_dict = {}

        data_dict['sample'] = sample_name
        data_dict['size_x'] = len(arr[0])
        data_dict['size_y'] = len(arr[0, 0])
        
        means = []

        #average = arr.sum() / arr.mask[0].sum()
        average = arr.data.sum() / len(arr[0]) / len(arr[0, 0])
        for i in range(len(arr)):
            means.append(arr[i].data.mean() / average)

        ddmeans = savgol_filter(means, 5, polyorder=3, deriv=2)
        dmeans = savgol_filter(means, 5, polyorder=3, deriv=1)

        for i in range(len(ddmeans)):
            data_dict[f'ddmean_{i}'] = ddmeans[i]
            data_dict[f'dmean_{i}'] = dmeans[i]
            data_dict[f'mean_{i}'] = means[i]
        data_dict['average_refl'] = average

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

        ft = FunctionTransformer(func=MergeData.transf)
        data = ft.fit_transform(data)

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

        data = ft.fit_transform(data)

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
