"""
Downsample the in-plane of a volume to a target resolution or by 2.
It's most common for 3D MRI to do scanner interpolation by a factor of 2,
so that's the default for this script.

"""
import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
from ..resize.affine import update_affine


def apply_fermi(x, target_shape):
    """
    Assume x is in the Fourier domain
    """

    for i, dim in enumerate(x.shape):
        t = np.linspace(-1, 1, dim)
        scale = dim / target_shape[i]
        length = dim
        width = 20.0 / length
        fermi = 1.0 / (1.0 + np.exp((np.abs(t) - 1.0 / scale) / width))

        broadcast_idx = [np.newaxis, np.newaxis]
        broadcast_idx.insert(i, slice(None, None))
        broadcast_idx = tuple(broadcast_idx)
        fermi = fermi[broadcast_idx]

        x *= fermi

    return x


def downsample_k_space(x, target_shape):
    omega = np.fft.fftshift(np.fft.fftn(x))
    old_energy = np.prod(omega.real.shape)
    crops = [(a - b) // 2 for a, b in zip(x.shape, target_shape)]
    crops = [c if c > 0 else 0 for c in crops]
    idx = tuple([slice(c, -c) if c != 0 else slice(None, None) for c in crops])
    omega_crop = omega[idx]

    omega_filt = apply_fermi(omega_crop, target_shape)
    new_energy = np.prod(omega_filt.real.shape)
    omega_filt *= new_energy / old_energy

    y = np.fft.ifftn(np.fft.ifftshift(omega_filt))
    y = np.abs(y)
    return y


@contextmanager
def timer_context(label, verbose=True):
    if verbose:
        print(label)
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        if verbose:  # Print elapsed time only if verbose is True
            print(f"\tElapsed time: {elapsed_time:.4f}s")


def process(in_fpath, out_fpath, inplane_res=None, axis=0, verbose=False):
    obj = nib.Nifti1Image.load(in_fpath)
    img = obj.get_fdata(dtype=np.float32)

    zooms = list(obj.header.get_zooms())
    throughplane_res = zooms.pop(axis)

    if inplane_res is None:
        inplane_res = [z * 2 for z in zooms]

    target_res = inplane_res
    target_res.insert(axis, throughplane_res)
    # Update the ratio of the new resolution
    scales = [new / old for new, old in zip(inplane_res, zooms)]
    # Insert `1` for the through-plane
    scales.insert(axis, 1)

    # find the ratio of pixels
    downsample_shape = [int(np.round(d / s)) for d, s in zip(img.shape, scales)]

    with timer_context(f"=== Degrading in-plane... ===", verbose=verbose):
        img = downsample_k_space(img, target_shape=downsample_shape)
        new_affine = update_affine(obj.affine, scales)
        nib.Nifti1Image(img, affine=new_affine, header=obj.header).to_filename(
            out_fpath
        )


def main(args=None):
    # ===== Read arguments =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-fpath", type=Path, required=True)
    parser.add_argument("--out-fpath", type=Path, required=True)
    parser.add_argument("--target-inplane-res", type=float, default=None)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--verbose", action="store_true", default=False)

    parsed_args = parser.parse_args(sys.argv[1:] if args is None else args)

    for argname in ["in_fpath", "out_fpath"]:
        setattr(parsed_args, argname, getattr(parsed_args, argname).resolve())

    if not parsed_args.in_fpath.exists():
        raise ValueError("Input filepath must exist.")

    for argname in ["out_fpath"]:
        getattr(parsed_args, argname).parent.mkdir(parents=True, exist_ok=True)

    process(
        parsed_args.in_fpath,
        parsed_args.out_fpath,
        inplane_res=parsed_args.target_inplane_res,
        axis=parsed_args.axis,
    )

    if parsed_args.verbose:
        print("Done.")


if __name__ == "__main__":
    main()
