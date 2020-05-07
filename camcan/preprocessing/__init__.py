"""Tools to extract information from MRI, fMRI data."""
from .temporal_series import extract_timeseries
from .connectivity import extract_connectivity
from .structural import get_structural_data

__all__ = ['extract_timeseries',
           'extract_connectivity',
           'get_structural_data']
