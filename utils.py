"""
Utility functions for terrain processing.
"""

import numpy as np
from typing import Tuple, List


def apply_convolution(matrix, kernel=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32)):
    """Apply a 2D convolution operation to a matrix using a given square kernel."""
    assert kernel.ndim == 2 and kernel.shape[0] == kernel.shape[1], "Kernel must be square"
    assert kernel.shape[0] % 2 == 1, "Kernel size must be odd"

    kernel = kernel.astype(np.float32)
    kernel /= kernel.sum() if kernel.sum() != 0 else 1

    k = kernel.shape[0] // 2
    padded = np.pad(matrix, pad_width=k, mode='edge')
    output = np.zeros_like(matrix)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            region = padded[i:i + 2 * k + 1, j:j + 2 * k + 1]
            output[i, j] = np.sum(region * kernel)

    return output


def apply_sobel_magnitude(matrix):
    """Applies Sobel X and Y kernels to compute edge magnitude."""
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [0,  0,  0],
        [1,  2,  1]
    ], dtype=np.float32)

    gx = apply_convolution(matrix, sobel_x)
    gy = apply_convolution(matrix, sobel_y)

    return np.sqrt(gx ** 2 + gy ** 2)


def dig_path(
    matrix: np.ndarray,
    kernel: np.ndarray,
    start_cell: Tuple[int, int],
    max_cells: int,
    vert_thresh: float
) -> np.ndarray:
    """
    Digs a path through a grid by applying a kernel to selected cells,
    starting from `start_cell`, and proceeding to neighbors whose height
    difference is within `vert_thresh`, using depth-first traversal.
    """
    h, w = matrix.shape
    dug_matrix = matrix.copy()
    visited = np.zeros_like(matrix, dtype=bool)
    frontier: List[Tuple[int, int]] = [start_cell]

    kh, kw = kernel.shape
    k_center_h = kh // 2
    k_center_w = kw // 2

    dug_count = 0
    dug_limit = max_cells

    def get_valid_neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = cell
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                vert_diff = abs(dug_matrix[nr, nc] - dug_matrix[r, c])
                if vert_diff <= vert_thresh:
                    neighbors.append((nr, nc))

        return neighbors

    while frontier and dug_count < dug_limit:
        r, c = frontier.pop()
        if visited[r, c]:
            continue

        visited[r, c] = True
        dug_count += 1

        r_start = max(0, r - k_center_h)
        r_end = min(h, r + k_center_h + 1)
        c_start = max(0, c - k_center_w)
        c_end = min(w, c + k_center_w + 1)

        region = dug_matrix[r_start:r_end, c_start:c_end]
        convolved_region = apply_convolution(region, kernel)
        dug_matrix[r_start:r_end, c_start:c_end] = convolved_region

        frontier.extend(get_valid_neighbors((r, c)))

    return dug_matrix


emboss = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]
], dtype=np.float32)

sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
], dtype=np.float32)

laplacian = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
], dtype=np.float32)

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float32)

box_blur_3x3 = np.ones((3, 3), dtype=np.float32)
box_blur_3x3 /= box_blur_3x3.sum()

box_blur_7x7 = np.ones((7, 7), dtype=np.float32)
box_blur_7x7 /= box_blur_7x7.sum()

box_blur_11x11 = np.ones((11, 11), dtype=np.float32)
box_blur_11x11 /= box_blur_11x11.sum()

box_blur_25x25 = np.ones((25, 25), dtype=np.float32)
box_blur_25x25 /= box_blur_25x25.sum()

gaussian_kernel_3x3 = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)
gaussian_kernel_3x3 /= gaussian_kernel_3x3.sum()

gaussian_kernel_5x5 = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
], dtype=np.float32)
gaussian_kernel_5x5 /= gaussian_kernel_5x5.sum()
