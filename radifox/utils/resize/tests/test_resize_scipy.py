import numpy as np
from resize.scipy import resize


def test_scipy():
    # Same FOV

    x1 = np.arange(6)[:, None, None]
    x2 = np.arange(7)[None, :, None]
    x3 = np.arange(5)[None, None, :]
    x = (x1 + x2 + x3).astype(float)
    dxyz = (1.5, 2, 0.3)
    y, coords = resize(x, dxyz, order=1, return_coords=True)
    assert y.shape == (4, 4, 17)
    assert np.array_equal(y[:, 0, 0], [0.25, 1.75, 3.25, 4.75])
    assert np.array_equal(y[0, :, 0], [0.25, 2.25, 4.25, 6.25])
    assert np.allclose(
        y[0, 0, :],
        [
            0.25,
            0.25,
            0.45,
            0.75,
            1.05,
            1.35,
            1.65,
            1.95,
            2.25,
            2.55,
            2.85,
            3.15,
            3.45,
            3.75,
            4.05,
            4.25,
            4.25,
        ],
    )

    y2 = resize(x, dxyz, order=1, target_shape=(3, 8, 12))
    assert y2.shape == (3, 8, 12)
    assert np.allclose(y2[:, 0, 0], [1.35, 2.85, 4.35])
    assert np.allclose(y2[0, :, 0], [1.35, 1.35, 1.35, 3.35, 5.35, 7.35, 7.35, 7.35])
    assert np.allclose(
        y2[0, 0, :],
        [1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05, 4.35, 4.65],
    )

    x1 = np.arange(5).astype(float)
    y1 = resize(x1, (0.7,), order=1)
    assert y1.shape == (7,)
    assert np.allclose(y1, np.array((0, 0.6, 1.3, 2.0, 2.7, 3.4, 4)))

    # Align first

    y = resize(x, dxyz, order=1, same_fov=False)
    assert y.shape == (4, 4, 14)
    assert np.allclose(y[:, 0, 0], [0, 1.5, 3, 4.5])
    assert np.allclose(y[0, :, 0], [0, 2, 4, 6])
    assert np.allclose(
        y[0, 0, :], [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]
    )

    y1 = resize(x1, (0.7,), order=1, same_fov=False)
    assert np.allclose(y1, [0, 0.7, 1.4, 2.1, 2.8, 3.5])

    y = resize(x, dxyz, order=1, same_fov=False, target_shape=(3, 8, 12))
    assert np.allclose(y[:, 0, 0], [0, 1.5, 3])
    assert np.allclose(y[0, :, 0], [0, 2, 4, 6, 6, 6, 6, 6])
    assert np.allclose(
        y[0, 0, :], [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3]
    )

    print("successful")


if __name__ == "__main__":
    test_scipy()
