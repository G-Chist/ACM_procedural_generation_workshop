"""
Terrain generation utilities.
"""

import numpy as np
from perlin_numpy import generate_fractal_noise_2d
from utils import apply_convolution


def generate_terrain_noise(
        size=1024,
        res=(8, 8),
        octaves=8,
        random_seed=123,
        sea_level=0.5,
        sky_level=1.0,
        sea_roughness=0.3,
        layers=0,
        trend=None,
        terrace=False,
        terrace_steepness=11,
        terrace_frequency=10,
        kernels=None
):
    """Generate filtered Perlin-based terrain noise with optional convolution kernels."""
    min_amplitude = 0.0
    max_amplitude = 1.0

    shape = (size, size)
    np.random.seed(random_seed)

    noise = generate_fractal_noise_2d(
        shape=shape, res=res, tileable=(True, True), octaves=octaves)

    noise_filtered = np.interp(
        noise, (noise.min(), noise.max()), (min_amplitude, max_amplitude))
    rand_range = (max_amplitude - min_amplitude) * 0.01

    noise_filtered = np.where(
        noise_filtered < sky_level, noise_filtered, sky_level)

    noise_filtered = np.where(
        noise_filtered > sea_level,
        noise_filtered,
        sea_level + np.random.uniform(
            -rand_range * sea_roughness,
            rand_range * sea_roughness,
            noise_filtered.shape
        )
    )

    for _ in range(layers):
        noise_filtered += noise_filtered

    if trend is not None:
        noise_filtered += trend

    noise_filtered = np.interp(noise_filtered, (noise_filtered.min(
    ), noise_filtered.max()), (min_amplitude, max_amplitude))

    if terrace:
        freq = terrace_frequency
        step = np.round(noise_filtered * freq) / freq
        noise_filtered = np.sin((noise_filtered - step)
                                * 2.45) ** terrace_steepness + step

    if kernels is None:
        kernels = []
    elif not isinstance(kernels, (list, tuple)):
        kernels = [kernels]

    for kernel in kernels:
        noise_filtered = apply_convolution(
            matrix=noise_filtered, kernel=kernel)

    return noise_filtered
