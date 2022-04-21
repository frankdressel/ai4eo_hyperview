import joblib
import luigi
import os
import numpy
import pandas
import pathlib
import tempfile

from ai4eo_hyperview.utils import MTimeMixin
from scipy.interpolate import UnivariateSpline 
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import mean_squared_error

class MergeData(MTimeMixin, luigi.Task):
    def output(self):
        return {
                'merged': luigi.LocalTarget('data/merged.csv'),
                'challenge': luigi.LocalTarget('data/challenge.csv'),
        }

    @staticmethod
    def sig(x, a, b, c, d):
        return a + b / (1 + numpy.exp(-c * (x - d)))

    @staticmethod
    def lin(x, a, b):
        return a + b * x

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
        return res

    @staticmethod
    def _prepare_arr(arr, wave, sample_name):
        data_dict = {}

        data_dict['sample'] = sample_name
        data_dict['size_x'] = len(arr[0])
        data_dict['size_y'] = len(arr[0, 0])
        
        means = [numpy.ma.mean(arr[i,:,:]) for i in range(arr.shape[0])]
        popt_s, pcov_s = curve_fit(MergeData.sig, wave['wavelength'], means, bounds=([200, 500, 0.05, 500], [1000, 5000, 5, 900]))
        popt_l, pcov_l = curve_fit(MergeData.lin, wave['wavelength'], means)
        mse_s = mean_squared_error(MergeData.sig(wave["wavelength"], *popt_s), means)
        mse_l = mean_squared_error(MergeData.lin(wave["wavelength"], *popt_l), means)
        if mse_s < mse_l:
            data_dict['sig'] = 1
            data_dict['lin'] = 0
            #for i in range(len(means)):
            #    f = MergeData.sig(wave['wavelength'], *popt_s)
            #    means[i] = means[i] / f[i]
        else:
            data_dict['sig'] = 0
            data_dict['lin'] = 1
            #for i in range(len(means)):
            #    f = MergeData.lin(wave['wavelength'], *popt_l)
            #    means[i] = means[i] / f[i]

        average = arr.data.sum() / len(arr[0]) / len(arr[0, 0])

        ddmeans = savgol_filter(means, 5, polyorder=3, deriv=2)
        dmeans = savgol_filter(means, 5, polyorder=3, deriv=1)

        l = 31
        p = 6
        means = [numpy.ma.mean(arr[i,:,:]) for i in range(arr.shape[0])] - savgol_filter([numpy.ma.mean(arr[i,:,:]) for i in range(arr.shape[0])], l, p)

        for i in range(len(ddmeans)):
            data_dict[f'ddmean_{i}'] = ddmeans[i]
            data_dict[f'dmean_{i}'] = dmeans[i]
            data_dict[f'mean_{i}'] = means[i]
        data_dict['average_refl'] = average
        data_dict['ndvi'] = (means[87] - means[57]) / (means[87] + means[57])
        data_dict['area'] = len(arr[0]) * len(arr[0, 0])

        return data_dict

    def run(self):
        # train data
        gt = pandas.read_csv('train_data/train_gt.csv', converters={'sample_index': str})
        wave = pandas.read_csv('train_data/wavelengths.csv')
        _data = []

        train_files = pathlib.Path('train_data/train_data').glob('*.npz')
        for tf in train_files:
            with numpy.load(tf) as npz:
                arr = numpy.ma.MaskedArray(**npz)

                sample_name = tf.name.replace('.npz', '')
                data_dict = MergeData._prepare_arr(arr, wave, sample_name)

                gt_values = gt[gt['sample_index'] == sample_name]
                data_dict['P'] = gt_values['P'].values[0]
                data_dict['K'] = gt_values['K'].values[0]
                data_dict['Mg'] = gt_values['Mg'].values[0]
                data_dict['pH'] = gt_values['pH'].values[0]

                _data.append(data_dict)
        data = pandas.DataFrame.from_dict(_data)
        data = data.apply(pandas.to_numeric, errors='ignore')

        #ft = FunctionTransformer(func=MergeData.transf)
        #data = ft.fit_transform(data)

        with self.output()['merged'].open('w') as f:
            data.to_csv(f, index=False)

        # Challenge test data
        _data = []

        test_files = pathlib.Path('test_data').glob('*.npz')
        for tf in test_files:
            with numpy.load(tf) as npz:
                arr = numpy.ma.MaskedArray(**npz)
                sample_name = tf.name.replace('.npz', '')
                data_dict = MergeData._prepare_arr(arr, wave, sample_name)

                _data.append(data_dict)
        data = pandas.DataFrame.from_dict(_data)
        data = data.apply(pandas.to_numeric, errors='ignore')

        #data = ft.fit_transform(data)

        with self.output()['challenge'].open('w') as f:
            data.to_csv(f, index=False)
