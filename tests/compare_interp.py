#!/usr/bin/env python

# pip install https://github.com/shuohan/resize.git
from resize.scipy import resize
from scipy.ndimage import zoom
from torch.nn.functional import interpolate
import torch
import numpy as np

sampling_step = 0.7
x = np.arange(6).astype(float)
ya = resize(x, (sampling_step,), order=1)
print(ya)  # [0.  0.4 1.1 1.8 2.5 3.2 3.9 4.6 5. ]

yb = zoom(x, 1 / sampling_step, order=1, mode="nearest")
print(yb)  # [0.    0.625 1.25  1.875 2.5   3.125 3.75  4.375 5.   ]

xc = torch.from_numpy(x)[None, None, ...]
yc = interpolate(xc, scale_factor=1 / sampling_step, mode="linear")
print(yc.squeeze().numpy())  # [0.   0.55 1.25 1.95 2.65 3.35 4.05 4.75]
