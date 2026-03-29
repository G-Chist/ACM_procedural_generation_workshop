"""
Gradio interface for cave generation using binary Perlin noise.
"""

import numpy as np
import matplotlib.pyplot as plt
import gradio as gr
from collections import deque
from perlin_numpy import generate_fractal_noise_2d


def find_accessible_point(cave, from_end=False):
    if from_end:
        for y in range(cave.shape[0] - 1, -1, -1):
            for x in range(cave.shape[1] - 1, -1, -1):
                if cave[y, x] == 1:
                    return (x, y)
    else:
        for y in range(cave.shape[0]):
            for x in range(cave.shape[1]):
                if cave[y, x] == 1:
                    return (x, y)
    return None


def bfs_path(cave, start, end):
    if start is None or end is None:
        return None
    
    h, w = cave.shape
    visited = set()
    queue = deque([(start, [start])])
    visited.add(start)
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    while queue:
        (x, y), path = queue.popleft()
        
        if (x, y) == end:
            return path
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and cave[ny, nx] == 1 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), path + [(nx, ny)]))
    
    return None


def generate_cave(
    size: int,
    res_x: int,
    res_y: int,
    octaves: int,
    random_seed: int,
    cutoff: float,
    invert: bool,
):
    np.random.seed(random_seed)
    
    noise = generate_fractal_noise_2d(
        shape=(size, size),
        res=(res_x, res_y),
        octaves=octaves,
    )
    
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    if invert:
        cave = (noise < cutoff).astype(np.uint8)
    else:
        cave = (noise >= cutoff).astype(np.uint8)
    
    start = find_accessible_point(cave, from_end=False)
    end = find_accessible_point(cave, from_end=True)
    path = bfs_path(cave, start, end)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cave, cmap='gray')
    
    if path is None:
        ax.text(cave.shape[1] // 2, cave.shape[0] // 2, "CAVE UNPASSABLE",
                ha='center', va='center', fontsize=24, color='red',
                fontweight='bold', transform=ax.transData)
    elif start is not None and end is not None:
        path_x, path_y = zip(*path)
        ax.plot(path_x, path_y, color='red', linewidth=2, alpha=0.8)
        ax.plot(start[0], start[1], 'go', markersize=10, label='Start')
        ax.plot(end[0], end[1], 'bo', markersize=10, label='End')
        ax.legend(loc='upper right')
    
    ax.set_axis_off()
    
    return fig


demo = gr.Interface(
    fn=generate_cave,
    inputs=[
        gr.Slider(64, 512, step=64, value=256, label="Size"),
        gr.Slider(1, 32, step=1, value=4, label="Resolution X"),
        gr.Slider(1, 32, step=1, value=4, label="Resolution Y"),
        gr.Slider(1, 16, step=1, value=6, label="Octaves"),
        gr.Number(label="Random Seed", value=42),
        gr.Slider(0.0, 1.0, step=0.01, value=0.5, label="Cutoff Threshold"),
        gr.Checkbox(label="Invert (caves <-> solid)", value=False),
    ],
    outputs=[
        gr.Plot(label="Cave Map"),
    ],
    title="Cave Generator",
    description="Generate cave-like mazes using binary fractal Perlin noise.",
)

if __name__ == "__main__":
    demo.launch()
