"""
Perlin noise implementation.
MIT License - Copyright (c) 2019 Pierre Vigier
"""

import numpy as np


def interpolant(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """Generate a 2D numpy array of perlin noise.

    Args:
        shape: The shape of the generated array (tuple of two ints).
        res: The number of periods of noise to generate along each axis.
        tileable: If the noise should be tileable along each axis.
        interpolant: The interpolation function.

    Returns:
        A numpy array of shape shape with the generated noise.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])

    x = np.arange(shape[0])
    y = np.arange(shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')

    cell_x = (X * res[0] / shape[0]).astype(int) % res[0]
    cell_y = (Y * res[1] / shape[1]).astype(int) % res[1]

    tx = (X * res[0] / shape[0]) % 1
    ty = (Y * res[1] / shape[1]) % 1

    angles = 2 * np.pi * np.random.rand(res[0], res[1])
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    if tileable[0]:
        gradients = np.concatenate([gradients, gradients[0:1, :]], axis=0)
    if tileable[1]:
        gradients = np.concatenate([gradients, gradients[:, 0:1]], axis=1)

    g00 = gradients[cell_x, cell_y]
    g10 = gradients[(cell_x + 1) % res[0], cell_y]
    g01 = gradients[cell_x, (cell_y + 1) % res[1]]
    g11 = gradients[(cell_x + 1) % res[0], (cell_y + 1) % res[1]]

    v00 = g00[:, :, 0] * tx + g00[:, :, 1] * ty
    v10 = g10[:, :, 0] * (tx - 1) + g10[:, :, 1] * ty
    v01 = g01[:, :, 0] * tx + g01[:, :, 1] * (ty - 1)
    v11 = g11[:, :, 0] * (tx - 1) + g11[:, :, 1] * (ty - 1)

    t = interpolant(np.stack([tx, ty], axis=-1))
    n0 = v00 * (1 - t[:, :, 0]) + t[:, :, 0] * v10
    n1 = v01 * (1 - t[:, :, 0]) + t[:, :, 0] * v11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant
):
    """Generate a 2D numpy array of fractal noise.

    Args:
        shape: The shape of the generated array.
        res: The base number of periods of noise along each axis.
        octaves: The number of octaves in the noise.
        persistence: The scaling factor between two octaves.
        lacunarity: The frequency factor between two octaves.
        tileable: If the noise should be tileable along each axis.
        interpolant: The interpolation function.

    Returns:
        A numpy array of fractal noise of shape shape.
    """
    noise = np.zeros(shape, dtype=np.float32)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        freq_res = (res[0] * frequency, res[1] * frequency)
        octave_noise = generate_perlin_noise_2d(
            shape, freq_res, tileable, interpolant
        )
        noise += amplitude * octave_noise
        frequency *= lacunarity
        amplitude *= persistence
    return noise
