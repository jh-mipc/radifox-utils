#!/usr/bin/env python

import numpy as np
from filter_bank_utils.filter_ops import *
from filter_bank_utils.batching_ops import *


def test_apply_filter():
    x = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
    ], dtype=np.float32)

    f = np.array([1, 1, 1], dtype=np.float32)
    f = f / f.sum()

    axis = 0

    y = np.array([
        [1, 1, 1],
        [2, 2, 2],
        [5/3, 5/3, 5/3],
    ], dtype=np.float32)

    x_filt = format_batch_1D_as_2D(
        apply_filter(
            format_2D_as_batch_1D(x, axis=axis),
            f,
            device='cpu',
        ),
        original_axis=axis,
    )

    assert x_filt.shape == y.shape
    assert np.allclose(x_filt, y)

    axis = 1

    y = np.array([
        [2/3, 1, 2/3],
        [4/3, 2, 4/3],
        [2, 3, 2],
    ], dtype=np.float32)

    x_filt = format_batch_1D_as_2D(
        apply_filter(
            format_2D_as_batch_1D(x, axis=axis),
            f,
            device='cpu',
        ),
        original_axis=axis,
    )

    assert x_filt.shape == y.shape
    assert np.allclose(x_filt, y)
