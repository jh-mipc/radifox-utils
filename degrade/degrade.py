"""
To degrade a signal is to blur and downsample it.

Blurring is convolution with a kernel. Some design parameters
are provided to calculate the kernel with particular FWHMs.

Most support is currently for either a user-provided kernel or
for a Gaussian kernel.
"""

import numpy as np
from resize.scipy import resize
from scipy import ndimage
from scipy.signal import windows


def fwhm_units_to_voxel_space(fwhm_space, voxel_space):
    """
    Translate the spatial resolution of the specified FWHM
    into voxel space. This is done by simply taking the ratio.

    For example, say the FWHM is specified to be 2 micrometers,
    but the resolution of each voxel is 0.5 micrometers. Therefore,
    the corresponding FWHM in 2 microns should span 4 voxels (2 / 0.5)

    Args:
        fwhm_space (float): the physical measurement of the FWHM 
        voxel_space (float): the physical measurement of the voxel resolution

    Returns:
        (float): The resultant FWHM in number of voxels
    """
    return fwhm_space / voxel_space


def std_to_fwhm(sigma):
    """
    Convert the standard deviation of a Gaussian kernel to
    its corresponding FWHM.

    Args:
        sigma (float): the standard deviation of the Gaussian kernel

    Returns:
        (float): The corresponding FWHM
    """
    return 2 * np.sqrt(2 * np.log(2)) * sigma


def fwhm_to_std(gamma):
    """
    Convert the FWHM of a Gaussian kernel to
    its corresponding standard deviation.

    Args:
        gamma (float): the FWHM of the Gaussian kernel

    Returns:
        (float): The corresponding standard deviation 
    """
    return gamma / (2 * np.sqrt(2 * np.log(2)))


def fwhm_needed(fwhm_hr, fwhm_lr):
    """
    We model the resolution of a signal by the FWHM of the PSF
    at acquisition time.

    When simulating the forward process, we want to "land on" the 
    specified resolution; this means we wish our result to have
    the specified FWHM.

    If our PSF is Gaussian, we can directly calculate the FWHM of the
    blur kernel needed to arrive at the target FWHM. When we convolve
    two Gaussian kernels, we add their variances. Thus, to find the 
    correct blur kernel, we can take a difference of the variances between
    our input FWHM and output FWHM.

    Args:
        fwhm_hr (float): The FWHM of the high resolution signal
        fwhm_lr (float): The FWHM of the low resolution signal

    Returns:
        (float): the FWHM of the blur kernel needed to bring the high resolution
                 signal to the desired low resolution
    """
    # First move specified FWHM to std
    std_hr = fwhm_to_std(fwhm_hr)
    std_lr = fwhm_to_std(fwhm_lr)

    # target std is root diff of variances
    std_target = np.sqrt(std_lr ** 2 - std_hr ** 2)

    # Translate back to FWHM space
    return std_to_fwhm(std_target)


def select_kernel(window_size, window_choice=None, fwhm=None, sym=True):
    """
    Utility function to select a blur kernel.

    Args:
        window_size (int): the number of taps for the kernel
        window_choice (string): the specific shape of the kernel; one of:
            - \'gaussian\'
            - \'hann\'
            - \'hamming\'
            - \'cosine\'
            - \'parzen\'
            - \'blackman\'
        fwhm (float): The FWHM of the kernel
        sym (bool): Whether the kernel should be symmetric

    Returns:
        (np.array): the parameterized kernel as a numpy array
    """

    WINDOW_OPTIONS = ['blackman', 'hann', 'hamming',
                      'gaussian', 'cosine', 'parzen']
    if window_choice is None:
        window_choice = np.random.choice(WINDOW_OPTIONS)
    elif window_choice not in WINDOW_OPTIONS:
        raise ValueError('Window choice (%s) is not supported.' %
                         window_choice)

    window = getattr(windows, window_choice)
    if window_choice in ['gaussian']:
        return window(window_size, fwhm_to_std(fwhm), sym)
    else:
        return window(window_size, sym)


def blur(x, blur_fwhm, axis, kernel_type='gaussian', kernel_file=None):
    """
    Blur a signal in 1D by convolution with a blur kernel along a 
    specified axis. The signal is edge-padded to keep its original size.

    Args:
        x (np.array: shape (N1, N2, ..., NN)): ND array to be blurred
        blur_fwhm (float): The FWHM of the blur kernel
        axis (int): the axis along which to blur
        kernel_type (string): The shape of the blur kernel
        kernel_file (string): The filepath to a user-specified kernel 
                              as a numpy file; must be `.npy`

    Returns:
        (np.array: shape (N1, N2, ..., NN)): The blurred signal
    """
    if kernel_file is not None:
        kernel = np.load(kernel_file)
    else:
        window_size = int(2 * round(blur_fwhm) + 1)
        kernel = select_kernel(window_size, kernel_type, fwhm=blur_fwhm)
    kernel /= kernel.sum()  # remove gain
    blurred = ndimage.convolve1d(x, kernel, mode='nearest', axis=axis)

    return blurred


def alias(img, k, down_order, up_order, axis):
    """
    Introduce aliasing in a signal in 1D by downsampling and upsampling.
    This is a phenomena which occurs in all signals when a sufficient
    bandwidth low-pass filter is NOT applied to a signal ahead of time.
    So when we downsample an image, we introduce aliasing in the frequency domain,
    which in turn affects the image domain. Upsampling does not introduce aliasing
    but is necessary to return the image to its original shape.

    Args:
        img (np.array: shape (N1, N2, ..., NN)): ND array to be aliased
        k (float): The resampling factor
        down_order (int): The order of the B-spline used to downsample. Must
                          be in the set {0, 1, 3, 5}
        up_order (int): The order of the B-spline used to upsample. Must
                          be in the set {0, 1, 3, 5}
        axis (int): the axis along which to introduce aliasing

    Returns:
        (np.array: shape (N1, N2, ..., NN)): The signal with aliasing artifacts
    """
    dxyz_down = [1.0 for _ in img.shape]
    dxyz_down[axis] = k
    dxyz_up = [1.0 for _ in img.shape]
    dxyz_up[axis] = 1 / k

    img_ds = resize(img, dxyz=dxyz_down, order=down_order)
    img_us = resize(img_ds, dxyz=dxyz_up, order=up_order,
                    target_shape=img.shape)

    return img_us
