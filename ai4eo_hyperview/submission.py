import luigi
import joblib
import pandas

from ai4eo_hyperview.prediction import RandomForest
from ai4eo_hyperview.utils import MTimeMixin

class Submission(MTimeMixin, luigi.Task):
    def requires(self):
        return {'regr': RandomForest()}

    def output(self):
        return {
                'submission': luigi.LocalTarget('data/submission.csv'),
        }

    def run(self):
        with self.input()['regr']['randomforest'].open('r') as f, self.output()['submission'].open('w') as fo:
            regr = joblib.load(f)
            challenge = pandas.read_csv('data/challenge.csv').sort_values('sample').drop(['sample', 'size_x', 'size_y'], axis=1)
            submission = pandas.DataFrame(data = regr.predict(challenge), columns=["P", "K", "Mg", "pH"])
            submission.to_csv(fo, index_label="sample_index")

#        with self.input()['regr']['gradient_P'].open('r') as fp, self.input()['regr']['gradient_K'].open('r') as fk, self.input()['regr']['gradient_Mg'].open('r') as fmg, self.input()['regr']['gradient_pH'].open('r') as fph, self.output()['submission'].open('w') as fo:
#            regr_p = joblib.load(fp)
#            regr_k = joblib.load(fk)
#            regr_mg = joblib.load(fmg)
#            regr_ph = joblib.load(fph)
#            challenge = pandas.read_csv('data/challenge.csv').sort_values('sample').drop(['sample', 'size_x', 'size_y'], axis=1)
#
#            submission = pandas.DataFrame()
#            submission['P'] = regr_p.predict(challenge)
#            submission['K'] = regr_k.predict(challenge)
#            submission['Mg'] = regr_mg.predict(challenge)
#            submission['pH'] = regr_ph.predict(challenge)
#
#            submission.to_csv(fo, index_label="sample_index")
