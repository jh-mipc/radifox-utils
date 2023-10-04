import torch

"""Resize with correct sampling step implemented with PyTorch

"""
import torch.nn.functional as F

from .abstract import Resize


def resize(
    image,
    dxyz,
    same_fov=True,
    target_shape=None,
    order=3,
    return_coords=False,
):
    """Wrapper function to resize an image using PyTorch.

    See :class:`ResizeTorch` for more details.

    Args:
        return_coords (bool): Return sampling coordinates if ``True``.

    Returns
    -------
    result : torch.Tensor
        The interpolated images.
    coords (optional) : torch.Tensor
        The sampling coordinates of these images.

    """
    resizer = ResizeTorch(
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


class ResizeTorch(Resize):
    """Resizes the image with sampling steps dx, dy, and dz in PyTorch.

    Same FOV mode:

    .. code-block::

        |   x   |   x   |   x   |
        | x | x | x | x | x | x |

    Aligning first point mode:

    .. code-block::

        |   x   |   x   |   x   |
          | x | x | x | x | x |

    Note:
        * "Replication" padding is used in interpolation.
        * For images with 3D spatial shape (5D input shape), only linear
          (``order = 1``) and nearest (``order = 0``) are supported.
        * For images with 2D spatial shape (4D input shape), cubic
          (``order = 3``) interpolation is also supported.
        * Images with 1D spatial shape is not supported.

    Args:
        image (torch.Tensor): The image to resample with shape ``(B, C,
            *spatial_size)``; ``B`` is the batch size, and ``C`` is the number
            of channels.
        dxyz (tuple[float]): The sampling steps. Less than 1 for upsampling. For
            example, ``(2.0, 0.8)`` for images with 2D spatial shape (4D input
            shape) and ``(1.3, 2.1, 0.3)`` for images with 3D spatial shape (5D
            input shape with batch and channel dimensions).
        same_fov (bool): Keep the same FOV as possible when ``True``. Otherwise,
            align the first points along each dimension between the input and
            resulting images.
        order (int): Interpolation order. 0: nearest; 1: linear; 2: cubic. See
            :func:`torch.nn.functional.grid_sample` for more details of
            the interpolation mode.
        target_shape (tuple[int]): The target spatial shape if not ``None``. If
            ``same_fov`` is ``True``, additional sizes are symmetrically padded
            at/cropped from both sides of each spatial dimension. If
            ``same_fov`` is ``False``, additional sizes are padded at/cropped
            from the end of each spatial dimension.

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
        self._mode = self._get_mode(self.order)
        super().__init__(image, dxyz, same_fov, target_shape)
        self._old_shape = self.image.shape[2:]

    def _get_mode(self, order):
        if order == 3:
            return "bicubic"
        elif order == 1:
            return "bilinear"
        elif order == 0:
            return "nearest"

    def _check_shape(self):
        super()._check_shape()
        assert len(self.image.shape) == len(self.dxyz) + 2

    @property
    def coords(self):
        """Returns the sampling coordinates.

        The shape of the coordinates array is ``(B, *spatial_size, M)`` where
        ``B`` is the batch size, and ``M`` is 3 for 3D spatial size and 2 for 2D
        spatial size. The ``spatial_size`` is the same with the resuling
        interpolated image. All ``B`` images of this batch share the same
        sampling indices.

        """
        return self._coords

    def _format_coords(self):
        # Map into the coordinates into [-1, 1] as required by F.grid_sample
        self._coords = self._normalize_coords()
        self._coords = torch.meshgrid(*self._coords, indexing="ij")
        self._coords = [c[None, ..., None] for c in self._coords]
        # Reverse the order of  coordinates for F.grid_sample
        self._coords.reverse()
        self._coords = torch.cat(self._coords, -1)
        # The first dim of coords is batch size. PyTorch needs it to be the same
        # with the images to sample.
        repeats = [self.image.shape[0]] + [1] * (self._coords.ndim - 1)
        self._coords = self._coords.repeat(repeats)

    def _normalize_coords(self):
        if self.same_fov:
            fov = [
                stop - start - 1
                for start, stop in zip(self._old_fov[0], self._old_fov[1])
            ]
        else:
            fov = self._old_fov
        return [
            torch.tensor(c / f * 2 - 1).to(self.image)
            for c, f in zip(self._coords, fov)
        ]

    def _resize(self):
        self._result = F.grid_sample(
            self.image,
            self._coords,
            mode=self._mode,
            align_corners=True,
            padding_mode="border",
        )
