import numpy as np

import torch

def distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute Euclidean distance between two tensors (supports multi-environment batch).
    """
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"

    dist = torch.norm(a - b, dim=-1)  # 计算欧氏距离
    dist = torch.round(dist * 1e6) / 1e6  # 保留 1e-6 精度
    return dist


def angle_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the geodesic distance between two array of angles. This function is vectorized.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.

    Returns:
        np.ndarray: The geodesic distance between the angles.
    """
    assert a.shape == b.shape
    dist = 1 - np.inner(a, b) ** 2
    return dist
