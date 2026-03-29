"""
Perlin Noise Explained - Step by Step Visualization
Based on: https://en.wikipedia.org/wiki/Perlin_noise
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import gradio as gr


def smoothstep(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def generate_gradients(res_x, res_y, seed=42):
    np.random.seed(seed)
    angles = 2 * np.pi * np.random.rand(res_y + 1, res_x + 1)
    return np.stack([np.cos(angles), np.sin(angles)], axis=-1)


STEP_DESCRIPTIONS = [
    """**Random Gradient Vectors**

A lattice is placed. At every lattice intersection, we assign a random **unit vector** (same length, random direction).

These arrows are not heights. They define local directional tendency (a slope field) that nearby pixels will project onto.

With a fixed seed, this gradient grid is deterministic, so reruns produce the same noise pattern.""",
    
    """**Dot Products at Each Corner**

Inside each cell, represent the pixel by local coordinates `tx, ty in [0, 1)`.
Build four offset vectors from each corner to that pixel:

- bottom-left: `(tx, ty)`
- bottom-right: `(tx-1, ty)`
- top-left: `(tx, ty-1)`
- top-right: `(tx-1, ty-1)`

Then compute each corner contribution with:
`dot(gradient, offset) = gx*ox + gy*oy`

- Positive (red) = pixel aligns with gradient direction
- Negative (blue) = pixel opposes gradient direction
- Larger magnitude = stronger influence from that corner""",
    
    """**Interpolate Horizontally (Bottom Row)**

Blend the two bottom corner contributions across x:
`n0 = (1-s(tx))*d00 + s(tx)*d10`

where `s(t)` is Perlin's fade curve:
`s(t) = 6t^5 - 15t^4 + 10t^3`

Linear `t` is not used because `s(t)` has zero slope at `t=0` and `t=1`, reducing visible seams at cell borders.

`n0` is a smooth signal along the bottom edge of each cell.""",
    
    """**Interpolate Horizontally (Top Row)**

Do the same interpolation for the top corners:
`n1 = (1-s(tx))*d01 + s(tx)*d11`

Now we have two x-blended values per pixel:

- `n0`: bottom-edge blend
- `n1`: top-edge blend

The final step is to blend these two values vertically.""",
    
    """**Interpolate Vertically (Y direction)**

Blend the two horizontal results along y:
`noise = (1-s(ty))*n0 + s(ty)*n1`

This is smooth bilinear interpolation in two stages (x, then y).

At the bottom edge (`ty=0`), output is exactly `n0`.
At the top edge (`ty=1`), output is exactly `n1`.
Between them, contributions transition smoothly.

Because adjacent cells share lattice corners and the same fade curve, the field stays continuous across cell boundaries.""",
    
    """**Complete Perlin Noise**

The result is one octave of 2D Perlin noise.

To build fractal (self-similar) maps, stack multiple octaves (noise maps of different frequency)."""
]


def perlin_step_by_step(width, height, res_x, res_y, seed=42):
    gradients = generate_gradients(res_x, res_y, seed)
    
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    
    cell_x = (X * res_x / width).astype(int)
    cell_y = (Y * res_y / height).astype(int)
    
    tx = (X * res_x / width) % 1
    ty = (Y * res_y / height) % 1
    
    x0 = cell_x
    x1 = (cell_x + 1) % (res_x + 1)
    y0 = cell_y
    y1 = (cell_y + 1) % (res_y + 1)
    
    g00 = gradients[y0, x0]
    g10 = gradients[y0, x1]
    g01 = gradients[y1, x0]
    g11 = gradients[y1, x1]
    
    d00 = g00[:, :, 0] * tx + g00[:, :, 1] * ty
    d10 = g10[:, :, 0] * (tx - 1) + g10[:, :, 1] * ty
    d01 = g01[:, :, 0] * tx + g01[:, :, 1] * (ty - 1)
    d11 = g11[:, :, 0] * (tx - 1) + g11[:, :, 1] * (ty - 1)
    
    t_x = smoothstep(tx)
    t_y = smoothstep(ty)
    
    n0 = d00 * (1 - t_x) + d10 * t_x
    n1 = d01 * (1 - t_x) + d11 * t_x
    noise = (1 - t_y) * n0 + t_y * n1
    
    cell_w = width / res_x
    cell_h = height / res_y
    
    # STEP 1: GRADIENT GRID
    fig1 = plt.figure(figsize=(8, 8))
    ax = fig1.add_subplot(111)
    ax.set_aspect('equal')
    for i in range(res_y + 1):
        for j in range(res_x + 1):
            gx, gy = gradients[i, j]
            ax.arrow(j * cell_w, i * cell_h, 
                     gx * cell_w * 0.4, gy * cell_h * 0.4,
                     head_width=cell_w*0.1, head_length=cell_h*0.05,
                     fc='red', ec='red', linewidth=2)
            ax.plot(j * cell_w, i * cell_h, 'k.', markersize=8)
    for i in range(res_y + 1):
        ax.axhline(i * cell_h, color='gray', linewidth=0.5)
    for j in range(res_x + 1):
        ax.axvline(j * cell_w, color='gray', linewidth=0.5)
    ax.set_xlim(-cell_w, width + cell_w)
    ax.set_ylim(-cell_h, height + cell_h)
    ax.invert_yaxis()
    ax.axis('off')
    
    # STEP 2: DOT PRODUCTS
    fig2 = plt.figure(figsize=(12, 10))
    ax2 = fig2.add_subplot(221)
    ax2.set_title("d00 = dot(G00, offset)", fontsize=11)
    ax2.imshow(d00, cmap='RdBu', vmin=-1, vmax=1)
    ax2.axis('off')
    ax2 = fig2.add_subplot(222)
    ax2.set_title("d10 = dot(G10, offset)", fontsize=11)
    ax2.imshow(d10, cmap='RdBu', vmin=-1, vmax=1)
    ax2.axis('off')
    ax2 = fig2.add_subplot(223)
    ax2.set_title("d01 = dot(G01, offset)", fontsize=11)
    ax2.imshow(d01, cmap='RdBu', vmin=-1, vmax=1)
    ax2.axis('off')
    ax2 = fig2.add_subplot(224)
    ax2.set_title("d11 = dot(G11, offset)", fontsize=11)
    ax2.imshow(d11, cmap='RdBu', vmin=-1, vmax=1)
    ax2.axis('off')
    
    # STEP 3: INTERPOLATE X (BOTTOM)
    fig3 = plt.figure(figsize=(12, 10))
    ax3 = fig3.add_subplot(131)
    ax3.set_title("d00 (bottom-left)", fontsize=10)
    ax3.imshow(d00, cmap='RdBu', vmin=-1, vmax=1)
    ax3.axis('off')
    ax3 = fig3.add_subplot(132)
    ax3.set_title("d10 (bottom-right)", fontsize=10)
    ax3.imshow(d10, cmap='RdBu', vmin=-1, vmax=1)
    ax3.axis('off')
    ax3 = fig3.add_subplot(133)
    ax3.set_title("n0 = blend(d00, d10)", fontsize=10)
    ax3.imshow(n0, cmap='RdBu', vmin=-1, vmax=1)
    ax3.axis('off')
    
    # STEP 4: INTERPOLATE X (TOP)
    fig4 = plt.figure(figsize=(12, 10))
    ax4 = fig4.add_subplot(131)
    ax4.set_title("d01 (top-left)", fontsize=10)
    ax4.imshow(d01, cmap='RdBu', vmin=-1, vmax=1)
    ax4.axis('off')
    ax4 = fig4.add_subplot(132)
    ax4.set_title("d11 (top-right)", fontsize=10)
    ax4.imshow(d11, cmap='RdBu', vmin=-1, vmax=1)
    ax4.axis('off')
    ax4 = fig4.add_subplot(133)
    ax4.set_title("n1 = blend(d01, d11)", fontsize=10)
    ax4.imshow(n1, cmap='RdBu', vmin=-1, vmax=1)
    ax4.axis('off')
    
    # STEP 5: INTERPOLATE Y
    fig5 = plt.figure(figsize=(12, 10))
    ax5 = fig5.add_subplot(131)
    ax5.set_title("n0 (bottom edge)", fontsize=10)
    ax5.imshow(n0, cmap='RdBu', vmin=-1, vmax=1)
    ax5.axis('off')
    ax5 = fig5.add_subplot(132)
    ax5.set_title("n1 (top edge)", fontsize=10)
    ax5.imshow(n1, cmap='RdBu', vmin=-1, vmax=1)
    ax5.axis('off')
    ax5 = fig5.add_subplot(133)
    ax5.set_title("noise = blend(n0, n1)", fontsize=10)
    ax5.imshow(noise, cmap='RdBu', vmin=-1, vmax=1)
    ax5.axis('off')
    
    # STEP 6: FINAL RESULT
    fig6 = plt.figure(figsize=(16, 8))
    ax6 = fig6.add_subplot(121)
    ls = LightSource(azdeg=315, altdeg=45)
    cmap = plt.cm.get_cmap('terrain')
    rgb = ls.shade(noise, cmap=cmap, vert_exag=0.5)
    ax6.imshow(rgb)
    ax6.axis('off')
    ax6 = fig6.add_subplot(122, projection='3d')
    X, Y = np.meshgrid(np.arange(width), np.arange(height))
    rgb3d = ls.shade(noise, cmap=cmap, vert_exag=0.5)
    ax6.plot_surface(X, Y, noise, facecolors=rgb3d, linewidth=0)
    ax6.view_init(elev=30, azim=45)
    ax6.set_axis_off()
    ax6.set_box_aspect((1, 1, 0.3))
    
    return fig1, fig2, fig3, fig4, fig5, fig6


def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Perlin Noise Explained")
        gr.Markdown("Step-by-step visualization of the Perlin noise algorithm")
        
        inputs = [
            gr.Slider(64, 256, step=16, value=128, label="Width"),
            gr.Slider(64, 256, step=16, value=128, label="Height"),
            gr.Slider(2, 16, step=1, value=4, label="Resolution X"),
            gr.Slider(2, 16, step=1, value=4, label="Resolution Y"),
            gr.Number(label="Seed", value=42),
        ]
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Gradient Vectors")
                gr.Markdown(STEP_DESCRIPTIONS[0])
            with gr.Column(scale=2):
                plot1 = gr.Plot()
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 2. Dot Products")
                gr.Markdown(STEP_DESCRIPTIONS[1])
            with gr.Column(scale=2):
                plot2 = gr.Plot()
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 3. Interpolate X (Bottom)")
                gr.Markdown(STEP_DESCRIPTIONS[2])
            with gr.Column(scale=2):
                plot3 = gr.Plot()
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 4. Interpolate X (Top)")
                gr.Markdown(STEP_DESCRIPTIONS[3])
            with gr.Column(scale=2):
                plot4 = gr.Plot()
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 5. Interpolate Y")
                gr.Markdown(STEP_DESCRIPTIONS[4])
            with gr.Column(scale=2):
                plot5 = gr.Plot()
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 6. Final Result")
                gr.Markdown(STEP_DESCRIPTIONS[5])
            with gr.Column(scale=2):
                plot6 = gr.Plot()
        
        btn = gr.Button("Generate")
        
        def update_all(*args):
            figs = perlin_step_by_step(*args)
            return figs
        
        btn.click(
            fn=update_all,
            inputs=inputs,
            outputs=[plot1, plot2, plot3, plot4, plot5, plot6]
        )
        
        for inp in inputs:
            inp.change(
                fn=update_all,
                inputs=inputs,
                outputs=[plot1, plot2, plot3, plot4, plot5, plot6]
            )
    
    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
