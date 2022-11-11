#!/usr/bin/env python

import pytest
import numpy as np
from filter_bank_utils.batching_ops import *


def test_format_2D_as_batch_1D():

    x = np.array([
        [1, 2, 3, ],
        [4, 5, 6, ],
        [7, 8, 9, ],
    ])

    x_batch_1D = format_2D_as_batch_1D(x, axis=0)
    y = np.array([
        [1, 4, 7, ],
        [2, 5, 8, ],
        [3, 6, 9, ],
    ])
    assert x_batch_1D.shape == y.shape
    assert np.all(x_batch_1D == y)

    x_batch_1D = format_2D_as_batch_1D(x, axis=1)
    y = np.array([
        [1, 2, 3, ],
        [4, 5, 6, ],
        [7, 8, 9, ],
    ])
    assert x_batch_1D.shape == y.shape
    assert np.all(x_batch_1D == y)

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    with pytest.raises(AssertionError):
        x_batch_1D = format_2D_as_batch_1D(x, axis=1)


def test_format_3D_as_batch_1D():

    x = np.array([
        [
            [1, 2, 3, 4, ],
            [5, 6, 7, 8, ],
            [9, 0, 1, 2, ],
        ],
        [
            [10, 20, 30, 40, ],
            [50, 60, 70, 80, ],
            [90, 00, 10, 20, ],
        ],
    ])
    x_batch_1D = format_ND_as_batch_1D(x, axis=0)
    y = np.array([
        [1, 10, ],
        [2, 20, ],
        [3, 30, ],
        [4, 40, ],
        [5, 50, ],
        [6, 60, ],
        [7, 70, ],
        [8, 80, ],
        [9, 90, ],
        [0, 00, ],
        [1, 10, ],
        [2, 20, ],
    ])
    assert x_batch_1D.shape == y.shape
    assert x_batch_1D.shape[0] == x.shape[1] * x.shape[2]
    assert x_batch_1D.shape[1] == x.shape[0]
    assert np.all(x_batch_1D == y)

    x_batch_1D = format_ND_as_batch_1D(x, axis=1)
    y = np.array([
        [1, 5, 9],
        [2, 6, 0],
        [3, 7, 1],
        [4, 8, 2],
        [10, 50, 90],
        [20, 60, 00],
        [30, 70, 10],
        [40, 80, 20],
    ])
    assert x_batch_1D.shape == y.shape
    assert x_batch_1D.shape[0] == x.shape[0] * x.shape[2]
    assert x_batch_1D.shape[1] == x.shape[1]
    assert np.all(x_batch_1D == y)

    x_batch_1D = format_ND_as_batch_1D(x, axis=2)
    y = np.array([
        [1, 2, 3, 4, ],
        [5, 6, 7, 8, ],
        [9, 0, 1, 2, ],
        [10, 20, 30, 40, ],
        [50, 60, 70, 80, ],
        [90, 00, 10, 20, ],
    ])
    assert x_batch_1D.shape == y.shape
    assert x_batch_1D.shape[0] == x.shape[0] * x.shape[1]
    assert x_batch_1D.shape[1] == x.shape[2]
    assert np.all(x_batch_1D == y)


def test_format_batch_1D_as_2D():

    x = np.array([
        [1, 2, 3, ],
        [4, 5, 6, ],
        [7, 8, 9, ],
    ])

    x_batch_1D = format_2D_as_batch_1D(x, axis=0)
    x_re_2D = format_batch_1D_as_2D(x_batch_1D, original_axis=0)
    assert x_re_2D.shape == x.shape
    assert np.all(x_re_2D == x)

    x_batch_1D = format_2D_as_batch_1D(x, axis=1)
    x_re_2D = format_batch_1D_as_2D(x_batch_1D, original_axis=1)
    assert x_re_2D.shape == x.shape
    assert np.all(x_re_2D == x)


def test_format_batch_1D_as_ND():

    original_shape = (2, 3, 4)
    x = np.random.rand(*original_shape)

    x_batch_1D = format_ND_as_batch_1D(x, axis=0)
    x_re_ND = format_batch_1D_as_ND(
        x_batch_1D, original_shape=original_shape, original_axis=0)
    assert x_re_ND.shape == x.shape
    assert np.all(x_re_ND == x)

    x_batch_1D = format_ND_as_batch_1D(x, axis=1)
    x_re_ND = format_batch_1D_as_ND(
        x_batch_1D, original_shape=original_shape, original_axis=1)
    assert x_re_ND.shape == x.shape
    assert np.all(x_re_ND == x)

    x_batch_1D = format_ND_as_batch_1D(x, axis=2)
    x_re_ND = format_batch_1D_as_ND(
        x_batch_1D, original_shape=original_shape, original_axis=2)
    assert x_re_ND.shape == x.shape
    assert np.all(x_re_ND == x)

    original_shape = (2, 3, 4, 4, 5, 6, 2)
    x = np.random.rand(*original_shape)

    for axis in range(len(original_shape)):
        x_batch_1D = format_ND_as_batch_1D(x, axis=axis)
        x_re_ND = format_batch_1D_as_ND(
            x_batch_1D, original_shape=original_shape, original_axis=axis)
        assert x_re_ND.shape == x.shape
        assert np.all(x_re_ND == x)
