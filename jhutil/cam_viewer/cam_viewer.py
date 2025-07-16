import sys
import os
import sys
import argparse
import numpy as np
import torch

from .src.visualizer import CameraVisualizer
from .src.loader import load_quick, load_nerf, load_colmap
from .src.utils import load_image


def get_example_datas(format="quick", root_path="inputs/quick/cam_c2w/", image_size=256, no_images=False):
    current_file_path = os.path.abspath(__file__)
    root_path = os.path.join(os.path.dirname(current_file_path), root_path)

    if format == 'quick':
        poses, legends, colors, image_paths = load_quick(root_path, "c2w")
    elif format == 'nerf':
        poses, legends, colors, image_paths = load_nerf(root_path)
    elif format == 'colmap':
        poses, legends, colors, image_paths = load_colmap(root_path)
    else:
        raise ValueError(f'Unknown format {format}')

    images = None
    if not no_images:
        images = []
        for fpath in image_paths:

            if fpath is None:
                images.append(None)
                continue

            if not os.path.exists(fpath):
                images.append(None)
                print(f'Image not found at {fpath}')
                continue

            images.append(load_image(fpath, sz=image_size))

    return poses, legends, colors, images


def visualize_camera(poses, legends=None, pcds=[], colors=["gray"], images=None, scene_size=5, show_indices=None):
    n_camera = len(poses)
    if show_indices is None:
        show_indices = range(len(poses))

    if images is not None:
        if isinstance(images[0], torch.Tensor):
            images = [image.cpu().numpy() for image in images]
            # redce resolution
        if images[0].shape[0] == 4:
            images = [image[:3] for image in images]
        if images[0].shape[0] == 3:
            images = [image.transpose(1, 2, 0) for image in images]
        if images[0].max() <= 1:
            images = [(image * 255).astype(np.uint8) for image in images]
        if images[0].shape[0] > 128:
            stride = images[0].shape[0] // 128
            images = [image[::stride, ::stride] for image in images]
    if legends is None:
        legends = [""] * len(poses)
    if len(poses) > len(colors):
        colors = [colors[i % len(colors)] for i in range(len(poses))]

    if images is not None:
        images = [images[i] for i in range(n_camera) if i in show_indices]
    if pcds is not None:
        pcds = [pcd for i, pcd in enumerate(pcds) if i in show_indices]
    poses = [poses[i] for i in range(n_camera) if i in show_indices]
    colors = [colors[i] for i in range(n_camera) if i in show_indices]
    legends = [legends[i] for i in range(n_camera) if i in show_indices]

    viz = CameraVisualizer(poses, legends, colors, images=images, pcds=pcds)
    fig = viz.update_figure(scene_bounds=scene_size, base_radius=1, zoom_scale=1,
                            show_grid=False, show_ticklabels=True, show_background=False)

    fig.show()


if __name__ == "__main__":
    poses, legends, colors, images = get_example_datas()
    visualize_camera(poses, images=images)
