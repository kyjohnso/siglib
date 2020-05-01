#!/usr/bin/env python
import numpy as np


def frame(x, frame_length, frame_step, pad=True, pad_value=0j):
    """
    Creates frames out of the 1D array x that are frame_length
    long, and by stepping frame_step samples.

    Parameters
    ----------
    x : numpy array of shape (N,) where N is the number of samples

    frame_length : int
        Length of each frame

    frame_step : int
        Specifies how many samples to step for each frame
        this does not have to be equal to frame_length

    pad : bool (optional, default is True)
        Whether the end of x should be padded before forming the
        frames. If false, the frames will only be created out to the
        last frame that contains samples, if true it will be padded out
        so that all samples are contained in a frame but the last frame
        may contain some pad values

    pad_value : float or complex (optional, defaults to 0j)
        If ``pad`` is True, this is the value that should be used to pad

    Returns
    -------
    x_frame : a (M, frame_length) sized array
        the framed data

    Examples
    --------
    >>> import numpy as np
    >>> from siglib import frame
    >>> x = np.arange(10)
    >>> frame(x, 5, 5)
    array([[0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j],
           [5.+0.j, 6.+0.j, 7.+0.j, 8.+0.j, 9.+0.j]])
    """

    if pad:
        n_frames = int(np.ceil((x.shape[-1] - frame_length) / frame_step) + 1)
        n_pad = int((n_frames - 1) * frame_step + frame_length - x.shape[-1])
        x = np.concatenate([x, np.full(n_pad, pad_value)], axis=-1)
    else:
        n_frames = int(np.floor((x.shape[-1] - frame_length) / frame_step) + 1)
    x_frame = np.array(
        [x[i * frame_step : i * frame_step + frame_length] for i in range(n_frames)]
    )
    return x_frame


def closing(x, ntaps):
    """
    Computes the closing morphological filter over ntaps of the input x
    
    Parameters
    -----------
    x : array of shape (N,)

    ntaps : int
        The width over which the min/max functions are calculated

    Returns
    --------
    y: array of shape (N,)
        The closing function over x

    Examples
    --------
    >>> import numpy as np
    >>> from siglib import closing
    >>> x = np.arange(10)
    >>> closing(x, 5)
    array([4.+0.j, 4.+0.j, 4.+0.j, 4.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j,
           8.+0.j, 9.+0.j])
    """
    x = np.concatenate([x, np.ones(ntaps - 1) * x[-1]])
    xf = frame(x, frame_length=ntaps, frame_step=1, pad=False)
    d = np.max(xf, axis=-1)
    d = np.concatenate([np.ones(ntaps - 1) * d[0], d])
    df = frame(np.flip(d), frame_length=ntaps, frame_step=1)
    c = np.min(df, axis=-1)
    return np.flip(c)


def opening(x, ntaps):
    """
    Computes the opening morphological filter over ntaps of the input x
    
    Parameters
    -----------
    x: array of shape (N,)

    ntaps: int
        The width over which the min max functions  are calculated

    Returns
    --------
    y: array of shape (N,)
        The opening function over x

    Examples
    --------
    >>> import numpy as np
    >>> from siglib import opening
    >>> x = np.arange(10)
    >>> opening(x, 5)
    array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j, 5.+0.j, 6.+0.j, 7.+0.j,
           8.+0.j, 9.+0.j])
    """
    x = np.concatenate([x, np.ones(ntaps - 1) * x[-1]])
    xf = frame(x, frame_length=ntaps, frame_step=1, pad=False)
    e = np.min(xf, axis=-1)
    e = np.concatenate([np.ones(ntaps - 1) * e[0], e])
    ef = frame(np.flip(e), frame_length=ntaps, frame_step=1)
    o = np.max(ef, axis=-1)
    return np.flip(o)


def resample(x, idx, ntaps):
    """
    Resamples the array x to the indicies in idx using sinc interpolation with
    ntaps number of points

    Parameters
    -----------
    x: array of shape (N,)

    idx: array of shape (M,)

    ntaps: int
        How many points to use for the sinc interpolation

    Returns
    --------
    y: array of shape (M,)
        The sinc interpolated signal at the indicies idx

    Example
    -------
    >>> import numpy as np
    >>> from siglib import resample
    >>> x = np.arange(10**2)
    >>> idx = np.array([45.567])
    >>> resample(x, idx, 5)
    array([44.96565413])
    """
    d_idx = np.ceil(np.arange(-ntaps / 2, ntaps / 2))
    idx_array = np.round(idx[:, np.newaxis]) + d_idx
    idx_array = np.asarray(idx_array, dtype=np.int32)
    sinc_array = idx[:, np.newaxis] - idx_array

    y = np.sum(x[idx_array] * np.sinc(sinc_array), axis=-1)
    return y


def dcm(x, delay=1, pad=True, pad_value=1 + 0j):
    """
    Perform a delay-conjugate-multiply on
    some input signal ``x`` with delay ``delay``

    Parameters
    ----------
    x : array of shape (N,)

    delay : int (optional, defaults to 1)

    pad : bool (optional, defaults to True)
        Whether or not to pad the right end of ``x``

    pad_value : complex (optional, defaults to 1+0j)
        If ``pad`` is True, this is the value that should be used to pad.
        Note: defaults to ``1+0j`` to avoid issues in subsequent ``np.atan2``.

    Returns
    -------
    y : array of shape (N,)
        Delay-conjugate-multiply result

    Example
    -------
    >>> from siglib import dcm
    >>> x = np.array([1 + 3j, 4 + 2j, 5 + 6j, 1 + 0j])
    >>> dcm(x)
    array([10.-10.j, 32.+14.j,  5. -6.j,  1. +0.j])
    """
    if pad:
        x = np.concatenate([x, np.full(delay, pad_value)], axis=-1)
    return x[delay:] * np.conjugate(x[:-delay])
