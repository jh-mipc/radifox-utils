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
import sigpy.mri.rf as rf


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
    std_target = np.sqrt(std_lr**2 - std_hr**2)

    # Translate back to FWHM space
    return std_to_fwhm(std_target)


def pulse_based_profile(
    window_size,
    slice_thickness,
    tb=16,
    N=128,
    d1=0.01,
    d2=0.01,
    ptype="ex",
    ftype="ls",
):
    """
    Physics-based simulation of slice profile via the SLR algorithm.
    First design an RF pulse, then use ABRM to simulate the magnetization,
    and finally convert to a slice profile for use in digital convolution.

    :param [window_size]: [Number of taps in the slice profile.]
    :type [window_size]: [Int]

    :param [slice_thickness]: [Thickness of the desired slice in spatial metric units (cm, mm, um, etc.)]
    :type [slice_thickness]: [Float]

    :param [tb]: [Time-bandwidth product for the RF pulse]
    :type [tb]: [Float]

    :param [N]: [Number of samples for the RF pulse]
    :type [N]: [Int]

    :param [d1]: [Passband ripple level]
    :type [d1]: [Float]

    :param [d2]: [Stopband ripple level]
    :type [d2]: [Float]

    :param [ptype]: [Pulse type], defaults to [\'ex\']
    :type [ptype]: [String]

    :param [ftype]: [Filter type], defaults to [\'ls\']
    :type [ftype]: [String]

    :return: [A unit-energy 1D numpy array of the simulated slice profile]
    :rtype: [np.array(dtype=np.float32)]
    """

    # design pulse
    pulse = rf.slr.dzrf(N, tb, ptype, ftype, d1, d2, False)

    # simulate magnetization
    t = np.linspace(-2 * tb, 2 * tb, window_size)
    [a, b] = rf.sim.abrm(pulse, t, balanced=True)
    Mxy = 2 * np.multiply(np.conj(a), b)

    # convert to unit-energy magnitude for digital convolution
    slice_profile = np.abs(Mxy)
    slice_profile /= slice_profile.sum()

    # resample s.t. each sample of the kernel is in physical measurements
    space = t * slice_thickness / tb
    dx = space[1] - space[0]
    slice_profile = resize(
        slice_profile, dxyz=(1 / dx,), order=5, target_shape=(window_size,)
    )

    return slice_profile


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
            - \'boxcar\'
        fwhm (float): The FWHM of the kernel
        sym (bool): Whether the kernel should be symmetric

    Returns:
        (np.array): the parameterized kernel as a numpy array
    """

    WINDOW_OPTIONS = [
        "blackman",
        "hann",
        "hamming",
        "gaussian",
        "cosine",
        "parzen",
        "rect",
        "boxcar",
    ]
    if window_choice is None:
        window_choice = np.random.choice(WINDOW_OPTIONS)
    elif window_choice not in WINDOW_OPTIONS:
        raise ValueError("Window choice (%s) is not supported." % window_choice)

    # ===== Special case for rect =====
    if window_choice in ["rect", "boxcar"]:
        w = np.zeros(window_size, dtype=np.float32)
        L = window_size // 2

        for i in range(window_size):
            # shift val up
            n = np.abs(i - L)
            if n > fwhm / 2:
                w[i] = 0
            elif n == fwhm / 2:
                w[i] = 0.5
            elif n < fwhm / 2:
                w[i] = 1

        w = w / w.sum()
        return w

    # ===== Otherwise use `windows` module =====
    window = getattr(windows, window_choice)
    if window_choice in ["gaussian"]:
        return window(window_size, fwhm_to_std(fwhm), sym)
    else:
        return window(window_size, sym)


def blur(x, blur_fwhm, axis, kernel_type="gaussian", kernel_file=None):
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
    blurred = ndimage.convolve1d(x, kernel, mode="nearest", axis=axis)

    return blurred


def alias(img, k, order, axis):
    """
    Introduce aliasing in a signal in 1D by downsampling without applying a
    low-pass filter. This is a phenomena which occurs in all signals when a
    sufficient bandwidth low-pass filter is NOT applied to a signal ahead of time.
    So when we downsample an image, we introduce aliasing in the frequency domain,
    which in turn affects the image domain.

    Args:
        img (np.array: shape (N1, N2, ..., NN)): ND array to be aliased
        k (float): The downsampling factor
        order (int): The order of the B-spline used to downsample. Must
                     be in the set {0, 1, 3, 5}
        axis (int): the axis along which to introduce aliasing

    Returns:
        (np.array: shape (N1, N2, ..., NN)): The signal with aliasing artifacts
    """
    dxyz_down = [1.0 for _ in img.shape]
    dxyz_down[axis] = k

    return resize(img, dxyz=dxyz_down, order=order)
