#!/usr/bin/env python

import numpy as np
from filter_bank_utils.signal_ops_1d import *


def test_downsample_m():

    x = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    x_down = downsample_m(x, 1)
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    assert x_down.shape == y.shape
    assert np.all(x_down == y)

    x_down = downsample_m(x, 2)
    y = np.array([1, 3, 5, 7, ])
    assert x_down.shape == y.shape
    assert np.all(x_down == y)

    x_down = downsample_m(x, 3)
    y = np.array([1, 4, 7, ])
    assert x_down.shape == y.shape
    assert np.all(x_down == y)


def test_upsample_m():

    x = np.array([1, 2, 3, 4, ])

    x_up = upsample_m(x, 1)
    y = np.array([1, 2, 3, 4, ])
    assert x_up.shape == y.shape
    assert np.all(x_up == y)

    x_up = upsample_m(x, 2)
    y = np.array([1, 0, 2, 0, 3, 0, 4, 0, ])
    assert x_up.shape == y.shape
    assert np.all(x_up == y)

    x_up = upsample_m(x, 3)
    y = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, ])
    assert x_up.shape == y.shape
    assert np.all(x_up == y)


def test_advance():

    x = np.array([1, 2, 3, 4, ])

    x_adv = advance(x, 0)
    y = np.array([1, 2, 3, 4, ])
    assert x_adv.shape == y.shape
    assert np.all(x_adv == y)

    x_adv = advance(x, 1)
    y = np.array([2, 3, 4, 0])
    assert x_adv.shape == y.shape
    assert np.all(x_adv == y)

    x_adv = advance(x, 2)
    y = np.array([3, 4, 0, 0])
    assert x_adv.shape == y.shape
    assert np.all(x_adv == y)


def test_delay():

    x = np.array([1, 2, 3, 4, ])

    x_delayed = delay(x, 0)
    y = np.array([1, 2, 3, 4, ])
    assert x_delayed.shape == y.shape
    assert np.all(x_delayed == y)

    x_delayed = delay(x, 1)
    y = np.array([0, 1, 2, 3, ])
    assert x_delayed.shape == y.shape
    assert np.all(x_delayed == y)

    x_delayed = delay(x, 2)
    y = np.array([0, 0, 1, 2, ])
    assert x_delayed.shape == y.shape
    assert np.all(x_delayed == y)
