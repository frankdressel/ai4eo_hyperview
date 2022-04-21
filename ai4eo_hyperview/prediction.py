import joblib
import luigi
import pandas

from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

from ai4eo_hyperview.utils import MTimeMixin, mse
from ai4eo_hyperview.preprocessing import MergeData

class RandomForest(MTimeMixin, luigi.Task):
    def requires(self):
        return {'Merge': MergeData()}

    def output(self):
        return {
                'randomforest': luigi.LocalTarget('data/randomforest', format=luigi.format.Nop),
#                'randomforest_ph': luigi.LocalTarget('data/randomforest_ph', format=luigi.format.Nop),
#                'gradient_P': luigi.LocalTarget('data/gradient_P', format=luigi.format.Nop),
#                'gradient_K': luigi.LocalTarget('data/gradient_K', format=luigi.format.Nop),
#                'gradient_Mg': luigi.LocalTarget('data/gradient_Mg', format=luigi.format.Nop),
#                'gradient_pH': luigi.LocalTarget('data/gradient_pH', format=luigi.format.Nop),
        }

    def run(self):

        with self.input()['Merge']['merged'].open('r') as f:
            tot_x = pandas.read_csv(f).drop(['sample', 'size_x', 'size_y', 'P', 'K', 'Mg', 'pH'], axis=1)
        with self.input()['Merge']['merged'].open('r') as f:
            tot_y = pandas.read_csv(f).filter(regex='P|K|Mg|pH')

        paras = {
                'max_features': ['auto'],
                #'max_features': ['auto', 'sqrt', 'log2'],
                #'max_depth': [None, 5, 10, 20, 30],
                #'min_weight_fraction_leaf': [0, 0.5],
                #'ccp_alpha': [0, 0.5]
        }
        regr = RandomForestRegressor(n_estimators=500)
        gs = GridSearchCV(regr, paras, n_jobs=4)
        gs.fit(tot_x, tot_y)
        with self.output()['randomforest'].open('wb') as f:
            joblib.dump(gs, f)
