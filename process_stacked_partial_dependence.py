"""Plot mean absolute error (MAE) figures.

Two types of plots are done:
    - MAE versus the chronological age,
    - MAE of one modality versus MAE of another modality.
"""
# Author: Denis A. Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

from itertools import combinations
import numpy as np
import pandas as pd
from mne.externals import h5io

IN_DEPENDENCE = './data/age_stacked_dependence_model-full.h5'
OUT_DEPENDENCE_1D = './data/age_stacked_dependence_model-full-1d.csv'
OUT_DEPENDENCE_2D = './data/age_stacked_dependence_model-full-2d.csv'

dependence = h5io.read_hdf5(IN_DEPENDENCE.format('model-full'))

# check source code for contour potting:
# https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/inspection/partial_dependence.py#L619


if False:
    preds = dependence[0]['2d'][
        'MEG envelope diag--MEG mne_envelope_cross alpha'][0]
    values = dependence[0]['2d'][
        'MEG envelope diag--MEG mne_envelope_cross alpha'][1]

    XX, YY = np.meshgrid(values[0], values[1])

    Z = preds.T
    min_pd = Z.min()
    max_pd = Z.max()

    Z_level = np.linspace(min_pd, max_pd, num=8)
    CS = plt.contour(XX, YY, Z.T[0], levels=Z_level, linewidths=0.5,
                    colors='k')
    # vmin = Z_level[0], alpha = 0.75, **contour_kw)
    plt.contourf(XX, YY, Z.T[0], levels=Z_level, vmax=Z_level[-1],
                vmin=Z_level[0], alpha=0.75)

# https: // stackoverflow.com/questions/27570221/how-to-make-a-ggplot2-contour-plot-analogue-to-latticefilled-contour

#############
# Case 1:  1d
results_1d = list()
for ii, mod in enumerate(dependence):
    for name, (pred, values) in mod['1d'].items():
        df1d = pd.DataFrame({'marker': name,
                             'model': mod['stack_model'],
                             'model_type': mod['mod_type'],
                             'pred': pred[0],
                             'value': values[0]})
        results_1d.append(df1d)

results_1d = pd.concat(results_1d, axis=0)
results_1d.to_csv(OUT_DEPENDENCE_1D)

############
# Case 2: 2d
results_12 = list()
for ii, mod in enumerate(dependence):
    for name, (pred, values) in mod['2d'].items():
        df = pd.DataFrame({'marker': name,
                           'model': mod['stack_model'],
                           'model_type': mod['mod_type'],
                           'pred': pred.ravel(0)})
        value_grid = np.array([x.ravel() for x in np.meshgrid(*values)]).T
        name_a, name_b = name.split('--')
        df['x'] = value_grid[:, 0]
        df['y'] = value_grid[:, 1]
        df['variables'] = name
        df['var_x'] = name_a
        df['var_y'] = name_b
        results_12.append(df)
results_2d = pd.concat(results_12, axis=0)
results_2d.to_csv(OUT_DEPENDENCE_2D)
