import luigi

from ai4eo_hyperview.submission import Submission

if __name__=='__main__':
    luigi.build([Submission()], local_scheduler=True)
