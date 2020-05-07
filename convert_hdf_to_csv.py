import pandas as pd

indir = './data/'
fname = 'age_stacked_importance_8.h5'

importance_keys = (
    'MEG-all-no-diag',
    'MEG all',
    'ALL',
    'ALL MRI',
    'ALL no fMRI'
)

dfs = list()
for key in importance_keys:
    df = pd.read_hdf(indir + fname, key)
    df['stack_model'] = key
    df['is_mean'] = 'F'
    df['is_mean'].values[[0, 1001, 2002]] = 'T'
    dfs.append(df)

df_out = pd.concat(dfs, axis=0)
df_out.to_csv('./viz_outputs/' + fname.replace('h5', 'csv'))
