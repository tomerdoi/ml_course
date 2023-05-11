import os
import pandas as pd
import numpy as np


sub_dir = '/opt/project/assignment2/submissions'
sub_files = os.listdir(sub_dir)

rows_scores = {}
for i in range(len(sub_files)):
    df = pd.read_csv(os.path.join(sub_dir, sub_files[i]))
    for i, row in df.iterrows():
        if i not in rows_scores:
            rows_scores[i] = [row['Predicted']]
        else:
            rows_scores[i].append(row['Predicted'])

with open(os.path.join(sub_dir, 'agg.csv'), 'w') as fp:
    fp.write('Id,Predicted\n')
    for i in rows_scores:
        # fp.write('%d,%0.9f,%0.9f\n' % (i, round(float(np.mean(rows_scores[i])), 9), float(np.std(rows_scores[i]))))
        fp.write('%d,%0.9f\n' % (i, round(float(np.mean(rows_scores[i])), 9)))
pass
