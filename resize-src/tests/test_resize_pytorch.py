import torch
from resize.pytorch import resize


def test_pytorch():
    # Same FOV
    x1 = torch.arange(5).float()[None, None, :, None]
    x2 = torch.arange(7).float()[None, None, None, :]
    x = x1 + x2
    y = resize(x, (0.7, 2.4), order=0)
    y = resize(x, (0.7, 2.4), order=3)
    y = resize(x, (0.7, 2.4), order=1)
    assert y.shape == (1, 1, 7, 3)
    assert torch.allclose(
        y[0, 0, :, 0], torch.tensor((0.6, 1.2, 1.9, 2.6, 3.3, 4, 4.6))
    )
    assert torch.allclose(y[0, 0, 0, :], torch.tensor((0.6, 3, 5.4)))

    x1 = torch.arange(6)[:, None, None]
    x2 = torch.arange(7)[None, :, None]
    x3 = torch.arange(5)[None, None, :]
    x = (x1 + x2 + x3).float()[None, None, ...]
    dxyz = (1.5, 2, 0.3)
    y, coords = resize(x, dxyz, return_coords=True, order=1)
    assert y.shape == (1, 1, 4, 4, 17)
    assert torch.allclose(
        y[0, 0, :, 0, 0], torch.tensor([0.25, 1.75, 3.25, 4.75]).float()
    )
    assert torch.allclose(
        y[0, 0, 0, :, 0], torch.tensor([0.25, 2.25, 4.25, 6.25]).float()
    )
    assert torch.allclose(
        y[0, 0, 0, 0, :],
        torch.tensor(
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
            ]
        ).float(),
    )

    y2 = resize(x, dxyz, order=1, target_shape=(3, 8, 12))
    assert y2.shape == (1, 1, 3, 8, 12)
    assert torch.allclose(y2[0, 0, :, 0, 0], torch.tensor([1.35, 2.85, 4.35]).float())
    assert torch.allclose(
        y2[0, 0, 0, :, 0],
        torch.tensor([1.35, 1.35, 1.35, 3.35, 5.35, 7.35, 7.35, 7.35]).float(),
    )
    assert torch.allclose(
        y2[0, 0, 0, 0, :],
        torch.tensor(
            [1.35, 1.65, 1.95, 2.25, 2.55, 2.85, 3.15, 3.45, 3.75, 4.05, 4.35, 4.65]
        ).float(),
    )

    # Align first

    y = resize(x, dxyz, order=1, same_fov=False, target_shape=(3, 8, 12))
    assert torch.allclose(y[0, 0, :, 0, 0], torch.tensor([0, 1.5, 3]).float())
    assert torch.allclose(
        y[0, 0, 0, :, 0], torch.tensor([0, 2, 4, 6, 6, 6, 6, 6]).float()
    )
    assert torch.allclose(
        y[0, 0, 0, 0, :],
        torch.tensor(
            [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3]
        ).float(),
    )

    y = resize(x, dxyz, same_fov=False, order=1)
    assert y.shape == (1, 1, 4, 4, 14)
    assert torch.allclose(y[0, 0, :, 0, 0], torch.tensor([0, 1.5, 3, 4.5]))
    assert torch.allclose(y[0, 0, 0, :, 0], torch.tensor([0, 2, 4, 6]).float())
    assert torch.allclose(
        y[0, 0, 0, 0, :],
        torch.tensor(
            [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9]
        ),
    )
    x1 = torch.arange(5).float()[None, None, :, None]
    x2 = torch.arange(7).float()[None, None, None, :]
    x = x1 + x2
    y = resize(x, (0.7, 2.4), same_fov=False, order=1)
    assert torch.allclose(y[0, 0, :, 0], torch.tensor([0, 0.7, 1.4, 2.1, 2.8, 3.5]))
    assert torch.allclose(y[0, 0, 0, :], torch.tensor([0, 2.4, 4.8]))

    print("successful")


if __name__ == "__main__":
    test_pytorch()
