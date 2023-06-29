"""
Create LR-HR pairs at the specified resolution with the specified slice profile.
"""
import sys

import nibabel as nib
from .degrade import *
import numpy as np
from pathlib import Path
import argparse
from transforms3d.affines import compose, decompose
import time
from contextlib import contextmanager


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


def update_affine(affine, scales):
    """Updates affine matrix to take into account new resolution
    Args:
        affine (numpy.ndarray): The affine matrix to update.
        scales (tuple[float] or list[float]): Resolution scales in each direction.
            Less than 1 for upsampling. For example, ``(2.0, 0.8)`` for a 2D image
            and ``(1.3, 2.1, 0.3)`` for a 3D image.
    """
    # Decompose input affine
    tranforms, rotation, zooms, shears = decompose(affine)

    # Adjust zooms
    zooms_new = zooms * np.array(scales)

    # Calculate translation adjustment
    t_val = (
        np.where(np.abs(rotation.dot(scales)) > 1, -1, 1)
        * np.sign(tranforms)
        * np.abs(rotation.dot(zooms_new / 2 * ((1 / np.array(scales)) - 1)))
    )

    # Return the new composed affine matrix
    return compose(tranforms + t_val, rotation, zooms_new, shears)


def remove_slices(x, n, axis, crop_edge):
    if n == 0:
        return x
    crops = [slice(None, None) for _ in x.shape]
    if crop_edge == "major":
        crops[axis] = slice(None, -n)
    elif crop_edge == "minor":
        crops[axis] = slice(n, None)
    elif crop_edge == "center":
        n1 = int(np.floor(n / 2))
        n2 = int(np.ceil(n / 2))
        crops[axis] = slice(n1, -n2)
    return x[tuple(crops)]


def nearest_int_divisor_lower(a, b):
    c = a / b
    while not c.is_integer():
        a -= 1
        c = a / b
    return a


def simulate_lr(
    fpath,
    slice_profile,
    slice_thickness,
    slice_separation,
    axis,
    out_lr_fpath,
    out_hr_fpath,
    crop_edge,
    verbose,
):
    with timer_context(f"=== Loading {fpath}... ===", verbose=verbose):
        # noinspection PyTypeChecker
        obj: nib.Nifti1Image = nib.load(fpath)
        affine = obj.affine
        # noinspection PyTypeChecker
        header: nib.Nifti1Header = obj.header
        x = obj.get_fdata(dtype=np.float32)

        orig_res = round(header.get_zooms()[axis], 3)
        target_res = round(slice_thickness, 3)

    n = x.shape[axis] - nearest_int_divisor_lower(x.shape[axis], slice_separation)
    if crop_edge == "center":
        crop_str = f"{int(np.floor(n/2))} minor and {int(np.ceil(n/2))} major"
    else:
        crop_str = f"{n} {crop_edge}"
    with timer_context(f"=== Removing {crop_str} slices... ===", verbose=verbose):
        x_crop = remove_slices(x, n, axis, crop_edge)

    with timer_context(f"=== Saving HR image... ===", verbose=verbose):
        nib.Nifti1Image(x_crop, affine=affine, header=header).to_filename(out_hr_fpath)

    with timer_context(
        f"=== Degrading with {slice_profile} to {target_res} || {round(slice_separation - target_res, 3)}... ===",
        verbose=verbose,
    ):
        x_lr = apply_degrade(
            x_crop, orig_res, target_res, slice_separation, slice_profile, axis
        )

    with timer_context(f"=== Saving LR image... ===", verbose=verbose):
        scales = [1, 1, 1]
        scales[axis] = slice_separation / orig_res
        new_affine = update_affine(obj.affine, scales)
        nib.Nifti1Image(x_lr, affine=new_affine, header=header).to_filename(
            out_lr_fpath
        )


def main(args=None):
    # ===== Read arguments =====
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-fpath", type=Path, required=True)
    parser.add_argument("--out-hr-fpath", type=Path, required=True)
    parser.add_argument("--out-lr-fpath", type=Path, required=True)
    parser.add_argument("--axis", type=int, default=2)
    parser.add_argument("--slice-thickness", type=float, required=True)
    parser.add_argument("--slice-separation", type=float, required=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--slice-profile",
        type=str,
        default="rf-pulse-slr",
        choices=["rf-pulse-slr", "gaussian"],
    )
    parser.add_argument(
        "--crop-edge",
        type=str,
        default="major",
        choices=["major", "minor", "center"],
        help=('Whether to crop the major or minor indices when creating paired HR-LR data. '
              'Choose "center" to center-crop, biasing towards a major crop if odd.'),
    )

    parsed_args = parser.parse_args(sys.argv[1:] if args is None else args)

    for argname in ['in_fpath', 'out_lr_fpath', 'out_hr_fpath']:
        setattr(parsed_args, argname, getattr(parsed_args, argname).resolve())

    if not parsed_args.in_fpath.exists():
        raise ValueError('Input filepath must exist.')

    for argname in ['out_lr_fpath', 'out_hr_fpath']:
        getattr(parsed_args, argname).parent.mkdir(parents=True, exist_ok=True)

    simulate_lr(
        parsed_args.in_fpath,
        parsed_args.slice_profile,
        parsed_args.slice_thickness,
        parsed_args.slice_separation,
        parsed_args.axis,
        parsed_args.out_lr_fpath,
        parsed_args.out_hr_fpath,
        parsed_args.crop_edge,
        parsed_args.verbose,
    )
    if parsed_args.verbose:
        print("Done.")


if __name__ == "__main__":
    main()
