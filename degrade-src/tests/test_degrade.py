#!/usr/bin/env python

import numpy as np
from degrade.degrade import *


def test_fwhm_units_to_voxel_space():

    fwhm_space = 3.3
    voxel_space = 1

    assert fwhm_units_to_voxel_space(fwhm_space, voxel_space) == 3.3

    fwhm_space = 2
    voxel_space = 0.5

    assert fwhm_units_to_voxel_space(fwhm_space, voxel_space) == 4

    fwhm_space = 3
    voxel_space = 1.5

    assert fwhm_units_to_voxel_space(fwhm_space, voxel_space) == 2


def test_std_fwhm_conversion():

    for sigma in [0.3, 1, 1.45]:
        assert np.isclose(fwhm_to_std(std_to_fwhm(sigma)), sigma)

    for fwhm in [0.287, 1, 2.485]:
        assert np.isclose(std_to_fwhm(fwhm_to_std(fwhm)), fwhm)


def test_fwhm_needed():

    hr_fwhm = 1
    lr_fwhm = 1
    assert np.isclose(fwhm_needed(hr_fwhm, lr_fwhm), 0)

    hr_fwhm = 1
    lr_fwhm = 4

    assert np.isclose(fwhm_needed(hr_fwhm, lr_fwhm), 3.8729833)

    hr_fwhm = 0.8
    lr_fwhm = 2.5

    assert np.isclose(fwhm_needed(hr_fwhm, lr_fwhm), 2.368543)
