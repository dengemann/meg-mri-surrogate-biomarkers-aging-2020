import mne

import numpy as np
import mne
from mne import io, read_proj, read_selection
from mne.datasets import sample
from mne.time_frequency import psd_welch

dfs = list()
for ii, psd in enumerate(psds):
    data = pd.DataFrame(
        dict(psd=psd,
            freqs=freqs)
    )
    data['channel'] = raw.ch_names[picks[ii]]
    dfs.append(data)
dfs = pd.concat(dfs, axis=0)
dfs.to_csv("./outputs/demo_meg_psd.csv")
