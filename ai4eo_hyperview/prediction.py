import joblib
import luigi
import pandas

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from ai4eo_hyperview.utils import MTimeMixin, mse
from ai4eo_hyperview.preprocessing import MergeData, Split

class RandomForest(MTimeMixin, luigi.Task):
    def requires(self):
        return {'Merge': MergeData(), 'Split': Split()}

    def output(self):
        return {
                'randomforest': luigi.LocalTarget('data/randomforest', format=luigi.format.Nop),
        }

    def run(self):
        with self.input()['Split']['train_x'].open('r') as f:
            train_x = pandas.read_csv(f).drop(['sample', 'size_x', 'size_y'], axis=1)
        with self.input()['Split']['train_y'].open('r') as f:
            train_y = pandas.read_csv(f).drop(columns=['sample'])
        with self.input()['Split']['test_x'].open('r') as f:
            test_x = pandas.read_csv(f).drop(['sample', 'size_x', 'size_y'], axis=1)
        with self.input()['Split']['test_y'].open('r') as f:
            test_y = pandas.read_csv(f).drop(columns=['sample'])

        tot_x = pandas.concat([train_x, test_x])
        tot_y = pandas.concat([train_y, test_y])

        paras = {
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [None, 5, 10, 20],
                'min_weight_fraction_leaf': [0, 0.5],
                'ccp_alpha': [0, 0.5]
        }
        regr = RandomForestRegressor(n_jobs=4, n_estimators=500)

        gs = GridSearchCV(regr, paras)
        gs.fit(tot_x, tot_y)

        with self.output()['randomforest'].open('wb') as f:
            joblib.dump(gs, f)