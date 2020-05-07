import numpy as np
import pandas as pd
from camcan.datasets import load_camcan_rest

bb = load_camcan_rest()

out = list()
for ii, fname in enumerate(bb.motion):
    data = np.genfromtxt(fname)
    df = pd.DataFrame(data)
    df['subject'] = bb.subject_id[ii][4:]
    df['n_points'] = data.shape[0]
    print(data.shape)
    out.append(df)

out = pd.concat(out)
out.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'subject']
out.to_csv('./data/motion.csv')
