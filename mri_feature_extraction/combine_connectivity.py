"""Combine connectivity information into a single file."""
import os
from os.path import join

import numpy as np
import pandas as pd

from camcan.datasets import load_camcan_timeseries_rest
from camcan.preprocessing import extract_connectivity

# load connectivity matrices
ATLASES = ['modl256', 'basc197']
CONNECTIVITY_KINDS = ('correlation', 'tangent')
# path for the different kind of connectivity matrices
CAMCAN_TIMESERIES = '/storage/tompouce/okozynet/camcan/timeseries'
OUT_DIR = '/storage/tompouce/okozynet/camcan/connectivity'

for connect_kind in CONNECTIVITY_KINDS:
    out_file = join(OUT_DIR, f'connect_data_{connect_kind}.h5')

    # remove the output file if it exists
    if os.path.exists(out_file):
        os.remove(out_file)

    for sel_atlas in ATLASES:
        print('**************************************************************')
        print(f'Reading timeseries files for {sel_atlas}')

        dataset = load_camcan_timeseries_rest(data_dir=CAMCAN_TIMESERIES,
                                              atlas=sel_atlas)
        connectivities = extract_connectivity(dataset.timeseries,
                                              kind=connect_kind)
        connect_data = None
        subjects = tuple(s[4:] for s in dataset.subject_id)

        for i, s in enumerate(subjects):
            if connect_data is None:
                columns = np.arange(start=0, stop=len(connectivities[i]))
                connect_data = pd.DataFrame(index=subjects,
                                            columns=columns,
                                            dtype=float)
                if connect_kind == 'correlation':
                    # save and apply Fisher's transform
                    connect_data.loc[s] = np.arctanh(connectivities[i])
                else:
                    connect_data.loc[s] = connectivities[i]
            else:
                if connect_kind == 'correlation':
                    # save and apply Fisher's transform
                    connect_data.loc[s] = np.arctanh(connectivities[i])
                else:
                    connect_data.loc[s] = connectivities[i]

        connect_data.to_hdf(out_file, key=sel_atlas, complevel=9)
