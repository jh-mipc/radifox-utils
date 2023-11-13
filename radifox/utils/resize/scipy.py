"""Resize with correct sampling step implemented with SciPy.

"""
import numpy as np
from scipy.ndimage import map_coordinates

from .abstract import Resize


def resize(
    image,
    dxyz,
    same_fov=True,
    target_shape=None,
    order=3,
    return_coords=False,
):
    """Wrapper function to resize an image using SciPy.

    See :class:`ResizeScipy` for more details.

    Args:
        return_coords (bool): Return sampling coordinates if ``True``.

    Returns
    -------
    result : numpy.ndarray
        The interpolated image.
    coords (optional) : numpy.ndarray
        The sampling coordinates of this image.

    """
    resizer = ResizeScipy(
        image,
        dxyz,
        same_fov=same_fov,
        target_shape=target_shape,
        order=order,
    )
    resizer.resize()
    if return_coords:
        return resizer.result, resizer.coords
    else:
        return resizer.result


class ResizeScipy(Resize):
    """Resizes the image with sampling steps dx, dy, and dz using SciPy.

    Same FOV mode:

    .. code-block::

        |   x   |   x   |   x   |
        | x | x | x | x | x | x |

    Aligning first point mode:

    .. code-block::

        |   x   |   x   |   x   |
          | x | x | x | x | x |

    Note:
        "Replication" padding is used in the interpolation.

    Args:
        image (numpy.ndarray): The image to resample.
        dxyz (tuple[float]): The sampling steps. Less than 1 for upsampling. For
            example, ``(2.0, 0.8)`` for a 2D image and ``(1.3, 2.1, 0.3)`` for
            a 3D image.
        same_fov (bool): Keep the same FOV as possible when ``True``. Otherwise,
            align the first points along each dimension between the input and
            resulting images.
        target_shape (tuple[int]): The target spatial shape if not ``None``. If
            ``same_fov`` is ``True``, additional sizes are symmetrically padded
            at/cropped from both sides of each spatial dimension. If
            ``same_fov`` is ``False``, additional sizes are padded at/cropped
            from the end of each spatial dimension.
        order (int): B-spline interpolation order.

    """

    def __init__(
        self,
        image,
        dxyz,
        same_fov=True,
        target_shape=None,
        order=3,
    ):
        self.order = order
        super().__init__(image, dxyz, same_fov, target_shape)

    def _check_shape(self):
        super()._check_shape()
        assert len(self.image.shape) == len(self.dxyz)

    @property
    def coords(self):
        """Returns the sampling coordinates.

        The shape of the coordinates array is ``(M, N)``. ``N`` is the number of
        points to sample. It is equal to product of the shape of the resulting
        image :meth:`result`. ``M`` is the dimension of these sampled points
        (e.g., 1, 2, 3). Each column is the coordinates vector of a point to
        sample relative to the input image.

        """
        return self._coords

    def _format_coords(self):
        self._coords = np.meshgrid(*self._coords, indexing="ij")
        self._coords = np.array([c.flatten() for c in self._coords])

    def _resize(self):
        self._result = map_coordinates(
            self.image,
            self.coords,
            mode="nearest",
            order=self.order,
        )
        self._result = self._result.reshape(self._new_shape)
