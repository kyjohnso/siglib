#!/usr/bin/env pytest
import numpy as np
import pytest
from siglib import frame, closing, opening, resample


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
