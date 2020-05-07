#!/usr/bin/env python
import numba
import numpy as np


__all__ = ["frame", "closing", "opening", "resample", "dcm", "overlapsave", "hamming"]


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

    # Explanation for the below:
    # * ``np.arange(frame_length).reshape(1, -1)`` creates a row vector from [0, frame_length)
    # * ``np.arange(n_frames).reshape(-1, 1)`` creates a column vector from [0, n_frames)
    # * multiplying by frame_step makes the sliding window slide frame_steps at a time
    # * adding the two together gives us a n_frames x frame_length matrix,
    #   where each row is a window
    x_frame_idx = np.arange(frame_length).reshape(1, -1) + frame_step * np.arange(
        n_frames
    ).reshape(-1, 1)
    return x[x_frame_idx]


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
    df = frame(np.flip(d), frame_length=ntaps, frame_step=1, pad=False)
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
    ef = frame(np.flip(e), frame_length=ntaps, frame_step=1, pad=False)
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


@numba.njit(fastmath=True)
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
        x_pad = np.concatenate((x, np.full(delay, pad_value)), axis=-1)
        return x_pad[delay:] * np.conjugate(x_pad[:-delay])
    else:
        return x[delay:] * np.conjugate(x[:-delay])


def overlapsave(x, H, step):
    """
    Perform convolution of the 1D array x with a bank of fourier domain filters
    H. The convolution is done (as you might expect) by the overlapsave method.
    I know I could have the zero pad and fourier operations to get to H in this
    function, but this way you have to know a little about what you are doing.

    Parameters
    ----------
    x : array of shape (N,)

    H : array of shape (L,M,)
        The rows are zero padded and Fourier Transformed filters. There are L
        different filters each of length M << N. M specifies the length of the
        frames into which x is broken.

    step : int
        Specifies what the step is for each of the frame

    Returns
    -------
    xh : array of shape (L, N+M-step,)
        Each row is the convolution of x with the respective filter in H.

    Example
    -------
    >>> from siglib import overlapsave
    >>> x = np.array([ 0.+3.j, -4.+2.j, -4.+1.j, -5.+2.j, -4.-3.j,  2.+1.j,
    ...               -1.-2.j,  3.+3.j, -3.-1.j, -3.+2.j, -4.+0.j, -4.-4.j,
    ...               -3.-4.j, -4.-3.j, -4.+3.j, -3.+4.j, -4.-1.j, -5.+0.j,
    ...                4.+2.j,  2.-3.j])
    >>> h = np.array([-4.+4.j, -4.+3.j])
    >>> overlap = h.shape[-1] - 1
    >>> len_fft = int(2**(np.ceil(np.log2(8 * overlap))))
    >>> H = np.fft.fft(np.concatenate([h, np.zeros(len_fft - h.shape[-1])]))
    >>> step = H.shape[-1] - overlap
    >>> overlapsave(x, H, step)
    array([[-12.-12.j,  -1.-36.j,  22.-40.j,  25.-44.j,  42.-27.j,  13. +4.j,
          1. +6.j, -14. +5.j,  -5.-11.j,  19.-25.j,  22.-33.j,  48.-12.j,
         56. +8.j,  52. +3.j,  29.-28.j,   3.-52.j,  20.-37.j,  39.-28.j,
         -4. -7.j, -18.+24.j,   1.+18.j]])
    """
    # lets accept a 1D, 2D, or 3D H, in the first two cases we need to add a
    #   dimension for the array broadcasting in the X * H operation
    if H.ndim == 1:
        H = H[np.newaxis, np.newaxis, :]
    elif H.ndim == 2:
        H = H[:, np.newaxis, :]

    overlap = H.shape[-1] - step

    # since we will drop the first overlap samples in each row, lets zero pad
    #   x by the overlap value, this will also make it so that the output
    #   matches that of scipy.signal
    x = np.concatenate([np.zeros(overlap), x], axis=-1)

    N = x.shape[-1]

    x = frame(x, H.shape[-1], step, pad=True, pad_value=0j)
    X = np.fft.fft(x, axis=-1)
    XH = X * H
    xh = np.fft.ifft(XH, axis=-1)

    # for each row and for each filter take from index overlap till the end
    xh = xh[:, :, overlap:]

    # now reshape to get L rows and then drop the padding that happend at the
    # end during the frame operation
    xh = np.reshape(xh, (H.shape[0], xh.shape[-1] * xh.shape[-2]))
    xh = xh[:, :N]
    return xh


@numba.njit(fastmath=True)
def hamming(window_length):
    """
    A simple hamming window generator, I know, I know it is in scipy.signal
    with a bunch more OO windows and such but I just wanted something simple

    Parameters
    ----------
    window_length : int
        Length of the window

    Returns
    -------
    h : array of shape (N,)
        Contains the coeficients of the Hamming window

    Example
    -------
    >>> from siglib import hamming
    >>> hamming(5)
    array([0.08, 0.54, 1.  , 0.54, 0.08])
    """
    n = np.arange(0, window_length)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_length - 1))
