#!/usr/bin/env python

import numpy as np


def frame(x, frame_length, frame_step, pad=True, pad_value=0j):
    """
    Creates frames out of the 1D array x that are frame_length 
                    long, and by stepping frame_step samples.

    Parameters
    -----------
        x: array of shape (N,) where N is the number of samples
        frame_length: integer specifying the length of each frame
        frame_step: integer specifying how many samples to step for each frame
                        this does not have to be equal to frame_length
        pad: boolean whether the end of x should be padded before forming the 
                frames. If false, the frames will only be created out to the
                last frame that contains samples, if true it will be padded out
                so that all samples are contained in a frame but the last frame
                may contain some pad values
        pad_value: if pad is True, this is the value that should be used to pad

    Returns
    --------
        x_frame: a (M,frame_length) sized array that contains the framed data 
    """

    if pad:
        n_frames = int(np.ceil((x.shape[-1] - frame_length) / frame_step) + 1)
        n_pad = int((n_frames - 1) * frame_step + frame_length - x.shape[-1])
        x = np.concatenate([x, np.ones(n_pad) * pad_value], axis=-1)
    else:
        n_frames = int(np.floor((x.shape[-1] - frame_length) / frame_step) + 1)
    x_frame = np.array(
        [x[i * frame_step : i * frame_step + frame_length] for i in 
            range(n_frames)]
    )
    return x_frame


def closing(x, ntaps):
    """
    Computes the closing morphological filter over ntaps of the input x
    
    Parameters
    -----------
        x: array of shape (N,)
        ntaps: an integer that is the width over which the min max functions
                are calculated

    Returns
    --------
        y: array of shape (N,) that is the closing function over x
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
        ntaps: an integer that is the width over which the min max functions
                are calculated

    Returns
    --------
        y: array of shape (N,) that is the opening function over x
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
    Resamples the array x to the indecies in idx using sinc interpolation with
        ntaps number of points

    Parameters
    -----------
        x: array of shape (N,)
        idx: array of shape (M,)
        ntaps: integer specifying how many points to use for the sinc 
                interpolation

    Returns
    --------
        y: array of shape (M,) containing the sinc interpolated signal at the
            indecies idx
    """
    d_idx = np.ceil(np.arange(-ntaps / 2, ntaps / 2))
    idx_array = np.round(idx[:, np.newaxis]) + d_idx
    idx_array = np.asarray(idx_array, dtype=np.int32)
    sinc_array = idx[:, np.newaxis] - idx_array

    y = np.sum(x[idx_array] * np.sinc(sinc_array), axis=-1)
    return y
