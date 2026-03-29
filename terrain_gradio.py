"""
Gradio interface for terrain generation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import gradio as gr
from terrain_utils import generate_terrain_noise
from utils import (
    box_blur_3x3, box_blur_7x7, box_blur_11x11, box_blur_25x25,
    gaussian_kernel_3x3, gaussian_kernel_5x5, sharpen, emboss, laplacian
)


KERNEL_MAP = {
    "None": None,
    "Box Blur 3x3": box_blur_3x3,
    "Box Blur 7x7": box_blur_7x7,
    "Box Blur 11x11": box_blur_11x11,
    "Box Blur 25x25": box_blur_25x25,
    "Gaussian 3x3": gaussian_kernel_3x3,
    "Gaussian 5x5": gaussian_kernel_5x5,
    "Sharpen": sharpen,
    "Emboss": emboss,
    "Laplacian": laplacian,
}


def plot_single_3d(heightmap, elev, azim, elev_scale, save_path=None):
    dpi = 150
    fig = plt.figure(figsize=(14, 12), dpi=dpi)
    
    h, w = heightmap.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.get_cmap('terrain')
    
    ax = fig.add_subplot(111, projection="3d")
    rgb = ls.shade(heightmap, cmap=cmap, blend_mode='soft', vert_exag=elev_scale)
    ax.plot_surface(X, Y, heightmap, facecolors=rgb, linewidth=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, elev_scale))
    
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    return fig


def plot_terrain_3d(noise, z_scale):
    fig1 = plot_single_3d(noise, 30, 45, z_scale)
    fig2 = plot_single_3d(noise, 15, 135, z_scale)
    
    return fig1, fig2


def create_terrain(
    size: int,
    res_x: int,
    res_y: int,
    octaves: int,
    random_seed: int,
    sea_level: float,
    sky_level: float,
    sea_roughness: float,
    layers: int,
    terrace: bool,
    terrace_steepness: int,
    terrace_frequency: int,
    kernel: str,
    z_scale: float,
):
    kernel_val = KERNEL_MAP[kernel]
    kernels = [kernel_val] if kernel_val is not None else None

    noise = generate_terrain_noise(
        size=size,
        res=(res_x, res_y),
        octaves=octaves,
        random_seed=random_seed,
        sea_level=sea_level,
        sky_level=sky_level,
        sea_roughness=sea_roughness,
        layers=layers,
        terrace=terrace,
        terrace_steepness=terrace_steepness,
        terrace_frequency=terrace_frequency,
        kernels=kernels,
    )

    heightmap = (noise * 255).astype(np.uint8)
    fig1, fig2 = plot_terrain_3d(noise, z_scale)

    return heightmap, fig1, fig2


demo = gr.Interface(
    fn=create_terrain,
    inputs=[
        gr.Slider(64, 512, step=64, value=256, label="Size"),
        gr.Slider(1, 32, step=1, value=2, label="Resolution X"),
        gr.Slider(1, 32, step=1, value=2, label="Resolution Y"),
        gr.Slider(1, 16, step=1, value=6, label="Octaves"),
        gr.Number(label="Random Seed", value=42),
        gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="Sea Level"),
        gr.Slider(0.0, 1.0, step=0.05, value=1.0, label="Sky Level"),
        gr.Slider(0.0, 1.0, step=0.05, value=0.3, label="Sea Roughness"),
        gr.Slider(0, 10, step=1, value=0, label="Layers"),
        gr.Checkbox(label="Terrace", value=False),
        gr.Slider(1, 20, step=1, value=11, label="Terrace Steepness"),
        gr.Slider(1, 20, step=1, value=10, label="Terrace Frequency"),
        gr.Dropdown(
            list(KERNEL_MAP.keys()),
            value="None",
            label="Post-process Kernel"
        ),
        gr.Slider(0.01, 2.0, step=0.01, value=0.2, label="Z Scale"),
    ],
    outputs=[
        gr.Image(label="Heightmap", type="numpy"),
        gr.Plot(label="Front"),
        gr.Plot(label="Side"),
    ],
    title="Procedural Terrain Generator",
    description="Generate terrain using fractal Perlin noise with various parameters.",
)

if __name__ == "__main__":
    demo.launch()
