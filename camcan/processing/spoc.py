"""Implement Source Power Comodulation (SPoC) framework."""
import copy as cp

from mne import EvokedArray
import numpy as np
from scipy.linalg import pinv, eigh
from sklearn.base import TransformerMixin


def shrink(cov, alpha):
    """Shrink covariance matrix."""
    n = len(cov)
    shrink_cov = (1 - alpha) * cov + alpha * np.eye(n)
    return shrink_cov


def fstd(y):
    """Standartize data."""
    y = y.astype(np.float64)
    y -= y.mean(axis=0)
    y /= y.std(axis=0)
    return y


class SPoC(TransformerMixin):
    """Source Power Comodulation (SPoC) framework."""

    def __init__(self, covs=None, fbands=None,
                 spoc=True, n_components=2, alpha=0):
        """Create a SPoC instanse.

        Parameters
        ----------
        covs: numpy.ndarray
            Covariance matrices for every subject and frequency band.
            The array is of shape (subjects, frequency_bands,
            channels, channels).
        fbands: list(tuple(float, float))
            List of frequency bands.
        spoc: bool
            Whether to run SPoC or not, default is True.
        n_components: int
            Number of filter components to consider, default is 2.
        alpha: int
            Default is 0.

        """
        self.covs = covs
        self.fbands = fbands
        self.spoc = spoc
        self.n_components = n_components
        self.alpha = alpha

    def fit(self, X, y):
        """Fit to the training data.

        Parameters
        ----------
        X: numpy.ndarray | list | tuple
            List of subjects in the training data.
        y: float
            Target variable for every subject.

        """
        target = fstd(y)
        self.patterns_ = []
        self.filters_ = []

        if isinstance(X, np.ndarray) and\
           not np.issubdtype(X.dtype, np.integer):
            X = X.astype('int')
            X = X[:, 0]

        for fb in range(len(self.fbands)):
            if self.spoc:
                covsfb = self.covs[:, fb]
                C = covsfb[X].mean(axis=0)
                C = C / np.trace(C)
                Cz = np.mean(covsfb[X] * target[:, None, None],
                             axis=0)
                C = shrink(C, self.alpha)
                eigvals, eigvecs = eigh(Cz, C)
                eigvals = eigvals.real
                eigvecs = eigvecs.real
                ix = np.argsort(np.abs(eigvals))[::-1]
                evecs = eigvecs[:, ix].T
                self.patterns_.append(pinv(evecs).T)  # (fb, chan,chan)
                self.filters_.append(evecs)  # (fb, chan,chan) row vec
            else:
                self.patterns_.append(np.eye(self.n_components))
                self.filters_.append(np.eye(self.n_components))
        return self

    def transform(self, X):
        """
        Return age-related patterns in MEG data.

        Parameters
        ----------
        X: numpy.ndarray | list(int) | tuple(int)
            List of subject to get patterns for.

        """
        if isinstance(X, np.ndarray) and\
           not np.issubdtype(X.dtype, np.integer):
            X = X.astype('int')
            X = X[:, 0]

        Xf = np.empty((X.size, len(self.fbands), self.n_components))
        for fb in range(len(self.fbands)):
            filters = self.filters_[fb][:self.n_components]
            # (comp,chan)
            this_Xf = [np.diag(filters @ self.covs[sub, fb] @ filters.T)
                       for sub in X]
            Xf[:, fb, :] = this_Xf
        Xf = np.log10(Xf, out=Xf)  # (fb, sub, compo,)
        Xf = Xf.reshape(X.size, -1)
        self.features = Xf
        return Xf  # (sub,compo*fb)

    def plot_patterns(self, info, components=None, fband=None,
                      ch_type=None, layout=None,
                      vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                      colorbar=True, scalings=None, units='a.u.', res=64,
                      size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                      show=True, show_names=False, title=None, mask=None,
                      mask_params=None, outlines='head', contours=6,
                      image_interp='bilinear', average=None, head_pos=None,
                      axes=None):
        """Plot found patterns."""
        if components is None:
            components = np.arange(self.n_components)
        patterns = self.patterns_[self.fbands.index(fband)]

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        norm_patterns = patterns / np.linalg.norm(patterns, axis=1)[:, None]
        patterns = EvokedArray(norm_patterns.T, info, tmin=0)
        return patterns.plot_topomap(
            times=components, ch_type=ch_type, layout=layout,
            vmin=vmin, vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors,
            scalings=scalings, units=units, time_unit='s',
            time_format=name_format, size=size, show_names=show_names,
            title=title, mask_params=mask_params, mask=mask, outlines=outlines,
            contours=contours, image_interp=image_interp, show=show,
            average=average, head_pos=head_pos, axes=axes)

    def plot_filters(self, info, components=None, fband=None,
                     ch_type=None, layout=None,
                     vmin=None, vmax=None, cmap='RdBu_r', sensors=True,
                     colorbar=True, scalings=None, units='a.u.', res=64,
                     size=1, cbar_fmt='%3.1f', name_format='CSP%01d',
                     show=True, show_names=False, title=None, mask=None,
                     mask_params=None, outlines='head', contours=6,
                     image_interp='bilinear', average=None, head_pos=None,
                     axes=None):
        """Plot found filters."""
        if components is None:
            components = np.arange(self.n_components)
        filters = self.filters_[self.fbands.index(fband)]

        # set sampling frequency to have 1 component per time point
        info = cp.deepcopy(info)
        info['sfreq'] = 1.
        filters = EvokedArray(filters, info, tmin=0)
        return filters.plot_topomap(
            times=components, ch_type=ch_type, layout=layout, vmin=vmin,
            vmax=vmax, cmap=cmap, colorbar=colorbar, res=res,
            cbar_fmt=cbar_fmt, sensors=sensors, scalings=scalings, units=units,
            time_unit='s', time_format=name_format, size=size,
            show_names=show_names, title=title, mask_params=mask_params,
            mask=mask, outlines=outlines, contours=contours,
            image_interp=image_interp, show=show, average=average,
            head_pos=head_pos, axes=axes)
