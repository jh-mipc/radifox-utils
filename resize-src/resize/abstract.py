import numpy as np


class Resize:
    """Resizes the image with sampling steps dx, dy, and dz.

    Same FOV mode:

    .. code-block::

        |   x   |   x   |   x   |
        | x | x | x | x | x | x |

    Aligning first point mode:

    .. code-block::

        |   x   |   x   |   x   |
          | x | x | x | x | x |

    """

    def __init__(
        self,
        image,
        dxyz,
        same_fov=True,
        target_shape=None,
    ):
        self.image = image
        self.dxyz = dxyz
        self.same_fov = same_fov
        self.target_shape = target_shape

        self._old_shape = self.image.shape
        self._result = None
        self._coords = None
        self._check_shape()

    def _check_shape(self):
        if self.target_shape is not None:
            assert len(self.target_shape) == len(self.dxyz)

    def resize(self):
        """Call this method to resize the input ``image``."""
        self._calc_sampling_coords()
        self._format_coords()
        self._resize()

    @property
    def result(self):
        """Returns the interpolated result."""
        return self._result

    @property
    def coords(self):
        """Returns the sampling coordinates."""
        return self._coords

    def _calc_sampling_coords(self):
        if self.same_fov:
            self._old_fov = self._calc_old_fov_same_fov()
            self._new_shape = self.target_shape
            if self.target_shape is None:
                self._new_shape = self._calc_new_shape_same_fov()
            self._new_fov = self._calc_new_fov_same_fov()
            self._coords = self._calc_sampling_coords_same_fov()
        else:
            self._old_fov = self._calc_old_fov_align_first()
            self._new_shape = self.target_shape
            if self.target_shape is None:
                self._new_shape = self._calc_new_shape_align_first()
            self._new_fov = self._calc_new_fov_align_first()
            self._coords = self._calc_sampling_coords_align_first()

    def _format_coords(self):
        raise NotImplementedError

    def _resize(self):
        raise NotImplementedError

    def _calc_sampling_coords_same_fov(self):
        coords = list()
        for l, r, d in zip(self._new_fov[0], self._new_fov[1], self.dxyz):
            coords.append(np.arange(l + d / 2, r - d / 4, d))
        return coords

    def _calc_sampling_coords_align_first(self):
        return [np.arange(0, f + d / 4, d) for f, d in zip(self._new_fov, self.dxyz)]

    def _calc_old_fov_same_fov(self):
        """Calculates the FOV of the original image.

        Suppose the left boundaries are at (-0.5, -0.5, -0.5), then the first
        voxel is at (0, 0, 0). Assume the step size is 1.

        """
        step_size = 1
        left = (-0.5,) * len(self._old_shape)
        right = tuple(l + s * step_size for l, s in zip(left, self._old_shape))
        return left, right

    def _calc_old_fov_align_first(self):
        """Assumes the FOV starts from the first and ends at the last point."""
        step_size = 1
        fov = tuple((s - 1) * step_size for s in self._old_shape)
        return fov

    def _calc_new_fov_same_fov(self):
        """Calculates the FOV of the resulting image.

        Assume the old and new FOV have the same center, then the new FOV is
        shifted from the old FOV by half of the size difference.

        """
        old_size = [r - l for l, r in zip(*self._old_fov)]
        new_size = [s * d for s, d in zip(self._new_shape, self.dxyz)]
        size_diff = [(o - n) / 2 for o, n in zip(old_size, new_size)]
        left = tuple(l + sd for l, sd in zip(self._old_fov[0], size_diff))
        right = tuple(r - sd for r, sd in zip(self._old_fov[1], size_diff))
        return left, right

    def _calc_new_fov_align_first(self):
        """The new FOV also starts from the first point."""
        fov = tuple((s - 1) * d for s, d in zip(self._new_shape, self.dxyz))
        return fov

    def _calc_new_shape_same_fov(self):
        return tuple(int(round(s / d)) for s, d in zip(self._old_shape, self.dxyz))

    def _calc_new_shape_align_first(self):
        """The largest number of points to ensure new FOV is within the old."""
        return tuple(
            int(np.floor((s - 1) / d) + 1) for s, d in zip(self._old_shape, self.dxyz)
        )
