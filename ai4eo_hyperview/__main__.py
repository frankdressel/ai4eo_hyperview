import luigi

from ai4eo_hyperview.prediction import RandomForest

if __name__=='__main__':
    luigi.build([RandomForest()], local_scheduler=True)
