import pandas as pd

indir = './data/'
fname = 'age_stacked_importance_5_global.h5'

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
    dfs.append(df)
df_out = pd.concat(dfs, axis=0)

df_out.to_csv(
    './viz_intermediate_files/' + fname.replace('h5', 'csv'))


fname2 = 'age_stacked_importance_8.h5'

dfs2 = list()
for key in importance_keys:
    df2 = pd.read_hdf(indir + fname2, key)
    df2 = df2.query("mod_type == 'permutation'")
    df2['fold_idx'] = pd.np.inf
    df2['repeat'] = pd.np.inf
    df2['imp_metric'] = 'permutation_insamp'
    df2['stack_model'] = key
    dfs2.append(df2)
df_out2 = pd.concat(dfs2, axis=0)
df_out2.to_csv(
    './viz_intermediate_files/' + fname2.replace('h5', 'csv'))
