import numpy as np
from transforms3d.affines import compose
from transforms3d.euler import euler2mat
from resize.affine import update_affine

def test_affine():
    original_affine = np.eye(4)
    new_affine = update_affine(original_affine, [1, 1, 4])
    expected_diff = np.zeros((4,4))
    expected_diff[2, 2] = 3.0
    expected_diff[2, 3] = 1.5
    assert np.allclose(new_affine - original_affine, expected_diff)
    
    T = np.array([14.0, 3.8, -39.1])
    R = euler2mat(np.pi/16, -np.pi/16, np.pi/16)
    Z = np.array([0.8, 0.8, 1.0])
    S = np.array([0.0, 0.0, 0.0])
    original_affine = compose(T, R, Z, S)
    new_affine = update_affine(original_affine, [1, 1, 3])
    expected_diff = np.zeros((4, 4))
    expected_diff[:, 2] = [-0.29921, -0.45734, 1.92388, 0.0]
    expected_diff[:, 3] = [0.1496, 0.22867, 0.96194, 0.0]
    assert np.allclose(new_affine - original_affine, expected_diff, atol=1e-5)
    
    print('successful')

if __name__ == '__main__':
    test_affine()
