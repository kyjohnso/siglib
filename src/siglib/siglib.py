#!/usr/bin/env python

import numpy as np

def frame(x,frame_length,frame_step,pad=True,pad_value=0j):
    """
    Description: Creates frames out of the 1D array x that are frame_length 
                    long, and by stepping frame_step samples.
    Inputs:
        x - numpy array of shape (N,) where N is the number of samples
        frame_length - integer specifying the length of each frame
        frame_step - integer specifying how many samples to step for each frame
                        this does not have to be equal to frame_length
        pad - boolean whether the end of x should be padded before forming the 
                frames. If false, the frames will only be created out to the
                last frame that contains samples, if true it will be padded out
                so that all samples are contained in a frame but the last frame
                may contain some pad values
        pad_value - if pad is True, this is the value that should be used to pad
    Outputs:
        x_frame - a (M,frame_length) sized array that contains the framed data 
    """
    if pad:
        n_frames = int(np.ceil((x.shape[-1] - frame_length)/frame_step)+1)
        n_pad = int((n_frames-1)*frame_step + frame_length-x.shape[-1])
        x = np.concatenate([x,np.ones(n_pad)*pad_value],axis=-1)
    else:
        n_frames = int(np.floor((x.shape[-1] - frame_length)/frame_step)+1)
    x_frame = np.array([x[i*frame_step:i*frame_step+frame_length] for i in range(n_frames)])
    return x_frame


def closing(x,ntaps):
    x = np.concatenate([x,np.ones(ntaps-1)*x[-1]])
    xf = frame(x,frame_length=ntaps,frame_step=1,pad=False)
    d = np.max(xf,axis=-1)
    d = np.concatenate([np.ones(ntaps-1)*d[0],d])
    df = frame(np.flip(d),frame_length=ntaps,frame_step=1)
    c = np.min(df,axis=-1)
    return np.flip(c)

def opening(x,ntaps):
    x = np.concatenate([x,np.ones(ntaps-1)*x[-1]])
    xf = frame(x,frame_length=ntaps,frame_step=1,pad=False)
    e = np.min(xf,axis=-1)
    e = np.concatenate([np.ones(ntaps-1)*e[0],e])
    ef = frame(np.flip(e),frame_length=ntaps,frame_step=1)
    o = np.max(ef,axis=-1)
    return np.flip(o)
