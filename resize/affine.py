import numpy as np
from transforms3d.affines import decompose, compose


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
    t_val = np.where(np.abs(rotation.dot(scales)) > 1, -1, 1) * rotation.dot(
        zooms_new / 2 * ((1 / np.array(scales)) - 1)
    )

    # Return the new composed affine matrix
    return compose(tranforms + t_val, rotation, zooms_new, shears)
