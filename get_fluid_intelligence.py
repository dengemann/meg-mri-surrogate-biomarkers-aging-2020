import glob
import pandas as pd

fnames = glob.glob(
    '/Users/dengeman/cc700-scored/Cattell/release001/data/*.txt')
dfs = list()
for fname in sorted(fnames):
    df = pd.read_csv(fname, sep='\t')
    df['subject'] = fname.split('/')[-1].split('_')[1]
    dfs.append(df)
df_cattell = pd.concat(dfs)
df_cattell.to_csv('data/catell.csv')
