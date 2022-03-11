#!/usr/bin/env python3

import pandas

df = pandas.read_csv('challenge/randomforest_final.csv')
df.to_csv("submission.csv", index_label="sample_index")
