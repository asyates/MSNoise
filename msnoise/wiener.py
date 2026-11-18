"""2-D Wiener filter of a CCF "image" (times x lag), applied before stacking.

The filter is the same adaptive Wiener filter as ``scipy.signal.wiener``
(local mean and variance in an M x N box; samples whose local variance is
below the noise level are replaced by the local mean), with three changes:

* the box is always centred (odd M and N; an even box in
  ``scipy.signal.wiener`` shifts the output by half a sample),
* the edges are padded by reflection instead of zeros, so the local mean is
  not pulled towards zero at the ends of a segment or of the lag axis,
* segments are split on gaps in the actual timestamps (rows that hold no
  data are not filtered and are not used as neighbours).
"""
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter


def odd_window(n):
    """Nearest odd window length >= 1 (an even box is not centred)."""
    n = max(1, int(round(n)))
    return n if n % 2 else n + 1


def find_segments(times, row_spacing, gap_threshold):
    """Split rows into continuous segments.

    :param times: sorted datetime64 array, one value per row holding data
    :param row_spacing: nominal spacing between rows (pd.Timedelta)
    :param gap_threshold: a missing span (time between two rows minus
        ``row_spacing``) of at least this duration starts a new segment
    :returns: list of integer index arrays into ``times``
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    if not len(times):
        return []
    missing = np.diff(times) - np.timedelta64(pd.Timedelta(row_spacing))
    split = np.flatnonzero(
        missing >= np.timedelta64(pd.Timedelta(gap_threshold))) + 1
    return np.split(np.arange(len(times)), split)


def wiener2d(im, size, noise=None):
    """Adaptive Wiener filter (as scipy.signal.wiener) with a centred box and
    reflected edges. ``size`` = (M, N), made odd."""
    im = np.asarray(im, dtype=float)
    size = tuple(odd_window(s) for s in size)
    l_mean = uniform_filter(im, size, mode="reflect")
    l_var = uniform_filter(im * im, size, mode="reflect") - l_mean ** 2
    l_var = np.maximum(l_var, 0.0)  # round-off can make it slightly < 0
    if noise is None:
        noise = l_var.mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = l_mean + (1.0 - noise / l_var) * (im - l_mean)
    # <= (not <): where l_var == noise the result is l_mean anyway, and it
    # avoids 0/0 on flat regions (l_var == noise == 0)
    return np.where(l_var <= noise, l_mean, out)


def wiener_filt(data, M, N, gap_threshold, row_spacing):
    """Wiener-filter the ``CCF`` variable of ``data`` (dims times x taxis),
    segment by segment.

    :param M: window length along the times axis, in rows (made odd)
    :param N: window length along the lag axis, in samples (made odd)
    :param gap_threshold: missing spans >= this (pd.Timedelta) split
        segments; shorter ones are bridged (rows either side treated as
        neighbours)
    :param row_spacing: nominal spacing of the rows (pd.Timedelta)
    """
    ccfs = data["CCF"]
    values = ccfs.values
    rows = np.flatnonzero(~np.isnan(values).all(axis=1))
    out = values.copy()
    for seg in find_segments(ccfs.times.values[rows], row_spacing,
                             gap_threshold):
        idx = rows[seg]
        out[idx, :] = wiener2d(values[idx, :], (M, N))
    filtered = data.copy(deep=True)
    filtered["CCF"] = (ccfs.dims, out)
    return filtered
