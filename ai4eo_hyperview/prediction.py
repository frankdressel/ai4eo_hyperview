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
                'randomforest': luigi.LocalTarget('randomforest', format=luigi.format.Nop),
                'features_P': luigi.LocalTarget('features_P.csv', format=luigi.format.Nop)
        }

    def run(self):
        with self.input()['Split']['train_x'].open('r') as f:
            train_x = pandas.read_csv(f)
        with self.input()['Split']['train_y'].open('r') as f:
            train_y = pandas.read_csv(f)['P'].values
        with self.input()['Split']['test_x'].open('r') as f:
            test_x = pandas.read_csv(f)
        with self.input()['Split']['test_y'].open('r') as f:
            test_y = pandas.read_csv(f)['P'].values

        with self.input()['Merge']['merged'].open('r') as f:
            co = pandas.read_csv(f).corr()
            cop = co.drop(['P', 'K', 'Mg', 'pH'])
            features = list(cop.sort_values('P')['P'][0:5].index.values) + list(cop.sort_values('P')['P'][-5:].index.values)

        forest = RandomForestRegressor(n_jobs=2, n_estimators=500, oob_score=True)
        #params = {
        #        'max_features': [10, 20, 30, 50, 100],
        #        'max_samples': [0.1, 0.2, 0.5, 1.0],
        #        'max_depth': [2, 5, 10, 15, 20],
        #}
        #regr = GridSearchCV(forest, params, scoring='neg_mean_absolute_error')
        #regr.fit(train_x, train_y)
        forest.fit(train_x[features], train_y)

        with self.output()['randomforest'].open('wb') as f:
            #joblib.dump(regr, f)
            joblib.dump(forest, f)
        with self.output()['features_P'].open('wb') as f:
            pandas.DataFrame(features, columns=['Features']).to_csv(f)
