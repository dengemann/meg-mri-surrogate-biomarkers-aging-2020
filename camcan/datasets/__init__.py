"""Utilities to load datasets."""
from .camcan import load_camcan_rest
from .camcan import load_camcan_timeseries_rest
from .camcan import load_camcan_connectivity_rest
from .camcan import load_camcan_contrast_maps
from .camcan import load_camcan_behavioural
from .camcan import load_camcan_behavioural_feature
from .camcan import load_masked_contrast_maps
from .camcan import iterate_masked_contrast_maps

__all__ = ['load_camcan_rest',
           'load_camcan_timeseries_rest',
           'load_camcan_connectivity_rest',
           'load_camcan_contrast_maps',
           'load_camcan_behavioural',
           'load_camcan_behavioural_feature',
           'load_camcan_behavioural',
           'load_masked_contrast_maps',
           'iterate_masked_contrast_maps']
