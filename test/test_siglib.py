#!/usr/bin/env pytest
import numpy as np
import pytest
from siglib import dcm, frame, closing, opening, resample, overlapsave


@pytest.mark.parametrize(
    "x,frame_length,frame_step,pad,pad_value,expected",
    (
        (np.arange(10), 5, 5, True, 0j, np.arange(10, dtype=np.complex).reshape(2, 5)),
        (np.arange(10), 5, 5, False, 0j, np.arange(10, dtype=np.complex).reshape(2, 5)),
    ),
)
def test_frame(x, frame_length, frame_step, pad, pad_value, expected):
    result = frame(x, frame_length, frame_step, pad=pad, pad_value=pad_value)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "x,ntaps,expected",
    (
        (np.zeros(10), 5, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (np.arange(10), 5, [4, 4, 4, 4, 4, 5, 6, 7, 8, 9]),
    ),
)
def test_closing(x, ntaps, expected):
    result = closing(x, ntaps)
    expected = np.array(expected, dtype=np.complex)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "x,ntaps,expected",
    (
        (np.zeros(10), 5, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        (np.arange(10), 5, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    ),
)
def test_opening(x, ntaps, expected):
    result = opening(x, ntaps)
    expected = np.array(expected, dtype=np.complex)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "x,idx,ntaps,expected",
    ((np.arange(10 ** 2), np.array([45.567]), 5, [44.96565413]),),
)
def test_resample(x, idx, ntaps, expected):
    result = resample(x, idx, ntaps)
    np.testing.assert_allclose(result, expected, rtol=1e-9)


@pytest.mark.parametrize(
    "x,delay,pad,pad_value,expected",
    (
        (
            [1 + 3j, 4 + 2j, 5 + 6j, 1 + 0j],
            1,
            True,
            1 + 0j,
            [10 - 10j, 32 + 14j, 5.0 - 6j, 1.0 + 0j],
        ),
        (
            [1 + 3j, 4 + 2j, 5 + 6j, 1 + 0j],
            1,
            False,
            1 + 0j,
            [10 - 10j, 32 + 14j, 5.0 - 6j],
        ),
    ),
)
def test_dcm(x, delay, pad, pad_value, expected):
    x = np.array(x)
    result = dcm(x, delay, pad=pad, pad_value=pad_value)
    expected = np.array(expected)
    np.testing.assert_equal(result, expected)


@pytest.mark.parametrize(
    "x,H,step,expected",
    (
        (
            [ 0.+3.j, -4.+2.j, -4.+1.j, -5.+2.j, -4.-3.j,  2.+1.j, 
             -1.-2.j,  3.+3.j, -3.-1.j, -3.+2.j, -4.+0.j, -4.-4.j, 
             -3.-4.j, -4.-3.j, -4.+3.j, -3.+4.j, -4.-1.j, -5.+0.j,  
              4.+2.j,  2.-3.j],
            [-8.+7.j, -4.70710678+8.94974747j,
             -1.+8.j,  0.94974747+4.70710678j,
              0.+1.j, -3.29289322-0.94974747j,
             -7.+0.j, -8.94974747+3.29289322j],
            7,
            [[-12.-12.j,  -1.-36.j,  22.-40.j,  25.-44.j,  42.-27.j,  13. +4.j,
                1. +6.j, -14. +5.j,  -5.-11.j,  19.-25.j,  22.-33.j,  48.-12.j,
               56. +8.j,  52. +3.j,  29.-28.j,   3.-52.j,  20.-37.j,  39.-28.j,
               -4. -7.j, -18.+24.j,   1.+18.j]],
        ),
        (
            [ 0.-5.j,  2.-4.j,  4.-1.j, -4.-2.j,  2.+0.j,  2.+0.j,  0.-5.j,
              0.-1.j, -2.+1.j, -4.-2.j,  0.+2.j, -5.-5.j, -5.-1.j, -2.+1.j,
              0.-1.j,  3.+4.j,  0.-2.j,  1.+0.j, -1.-2.j,  3.+3.j],
            [-6.+6.j , -3.29289322+6.53553391j,
             -1.+5.j , -0.46446609+2.29289322j,
             -2.+0.j , -4.70710678-0.53553391j,
             -7.+1.j , -7.53553391+3.70710678j],
            7,
            [[15.+20.j,  19.+32.j,  -5.+30.j,  17.+10.j,   6. -2.j, -12.+12.j,
              11.+26.j,  18.+14.j,   8. -8.j,  23.-12.j,   8.-16.j,  29. +1.j,
              48.-16.j,  18.-23.j,   4. -4.j, -21. -5.j, -12. +9.j,   2. +7.j,
               8. +8.j, -13. -2.j, -15. +3.j]],
        ),
        (
            [-2.-4.j,  4.-4.j, -1.-1.j, -5.-1.j,  4.-4.j,  1.-4.j, -5.+3.j,
              4.+3.j,  3.-3.j,  2.-4.j,  4.-5.j,  0.-4.j,  2.+1.j,  2.-4.j,
              4.+1.j, -5.-3.j,  2.+3.j,  4.+4.j,  1.+4.j, -1.-3.j],
            [[ 1.+2.j ,  2.24264069-1.j        ,
               1.-4.j , -2.        -5.24264069j,
              -5.-4.j , -6.24264069-1.j        ,
              -5.+2.j , -2.        +3.24264069j],
             [-4.+6.j , -0.29289322+6.94974747j,
               3.+5.j ,  3.94974747+1.29289322j,
               2.-2.j , -1.70710678-2.94974747j,
              -5.-1.j , -5.94974747+2.70710678j]],
            7,
            [[ 3.55271368e-15+1.0000000e+01j, -6.00000000e+00-1.4000000e+01j,
               2.50000000e+01+3.0000000e+00j,  9.00000000e+00+1.0000000e+00j,
              -2.40000000e+01-1.4000000e+01j,  1.80000000e+01+7.0000000e+00j,
               2.80000000e+01-1.0000000e+01j, -2.90000000e+01-1.6000000e+01j,
              -6.00000000e+00+2.4000000e+01j,  1.00000000e+01+6.0000000e+00j,
               5.00000000e+00+0.0000000e+00j,  2.30000000e+01+5.0000000e+00j,
               9.00000000e+00-1.6000000e+01j, -5.00000000e+00+1.5000000e+01j,
               1.10000000e+01-1.2000000e+01j,  1.60000000e+01+2.6000000e+01j,
              -7.00000000e+00-3.2000000e+01j, -7.00000000e+00+3.0000000e+00j,
               2.00000000e+00+1.5000000e+01j, -1.00000000e+01+2.2000000e+01j,
               6.00000000e+00-1.2000000e+01j],
             [ 1.00000000e+01-4.4408921e-16j,  2.60000000e+01+1.6000000e+01j,
               7.00000000e+00+2.7000000e+01j,  1.40000000e+01-1.0000000e+01j,
               2.30000000e+01-5.0000000e+00j,  1.10000000e+01+3.4000000e+01j,
               1.20000000e+01+3.0000000e+00j, -7.00000000e+00-2.4000000e+01j,
              -2.10000000e+01+1.6000000e+01j,  9.00000000e+00+2.9000000e+01j,
               1.60000000e+01+3.3000000e+01j,  1.60000000e+01+3.5000000e+01j,
               1.20000000e+01+1.5000000e+01j, -4.00000000e+00+1.3000000e+01j,
               4.00000000e+00+2.7000000e+01j, -5.00000000e+00+6.0000000e+00j,
               1.90000000e+01-1.0000000e+01j, -3.00000000e+01+3.0000000e+00j,
              -3.70000000e+01+2.0000000e+00j, -1.20000000e+01-7.0000000e+00j,
               1.50000000e+01+5.0000000e+00j]],
        ),
    ),
)
def test_overlapsave(x,H,step,expected):
    result = overlapsave(np.array(x),np.array(H),step)
    expected = np.array(expected)
    np.testing.assert_almost_equal(result,expected)
