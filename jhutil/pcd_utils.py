import numpy as np
import plotly.graph_objs as go
import trimesh
import matplotlib.pyplot as plt
import torch
import os
import math
import random
import jhutil
import pandas as pd
from typing import Union, List
from pyntcloud import PyntCloud
from .freq_utils import to_cpu


PALATTE = np.array([[204, 0, 0], [204, 204, 0], [0, 204, 0], [76, 76, 204], [127, 76, 127],
                    [0, 127, 127], [76, 153, 0], [153, 0, 76], [76, 0, 153], [153, 76, 0], [76, 0, 153], [153, 0, 76], [204, 51, 127], [204, 51, 127], [51, 204, 127], [51, 127, 204], [127, 51, 204], [127, 204, 51], [76, 76, 178], [76, 178, 76], [178, 76, 76]])


def show_point_clouds(point_clouds: List[torch.Tensor], colors=None, show_indices=None):
    if show_indices is None:
        show_indices = range(len(point_clouds))
    point_clouds = to_cpu(point_clouds)

    # Filter the point clouds
    n_pcd = len(point_clouds)
    point_clouds = [point_clouds[i] for i in range(n_pcd) if i in show_indices]
    lens = [len(pcd) for pcd in point_clouds]
    
    point_clouds = np.concatenate(point_clouds, axis=0)

    if colors is None:
        palatte = PALATTE[show_indices]
        colors = np.repeat(palatte, lens, axis=0)
    else:
        colors = np.concatenate(colors, axis=0)

    result = pd.DataFrame()
    result["x"] = point_clouds[:, 0]
    result["y"] = point_clouds[:, 1]
    result["z"] = point_clouds[:, 2]

    assert colors.shape == point_clouds.shape
    result["red"] = colors[:, 0]
    result["green"] = colors[:, 1]
    result["blue"] = colors[:, 2]

    PyntCloud(result).plot(return_scene=True, backend="pythreejs", initial_point_size=0.5)


def sample_point_cloud_from_mesh(mesh_file, num_points):
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Sample the point cloud
    point_cloud, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    return point_cloud


def transform_pcd_with_matrix(point_cloud, transform_matrix):
    # Convert the point cloud to homogeneous coordinates
    num_points = point_cloud.shape[0]
    ones = torch.ones((num_points, 1), dtype=torch.float)
    point_cloud_h = torch.cat((point_cloud, ones), dim=1).t()

    # Perform the transformation
    transformed_point_cloud_h = torch.matmul(transform_matrix.float(), point_cloud_h.float())

    # Convert back to 3D coordinates
    transformed_point_cloud = transformed_point_cloud_h[:3, :].t()

    return transformed_point_cloud


def get_matrix_from_quat_trans(quaternion, translation):
    from pytorch3d.transforms import quaternion_to_matrix
    rotation_matrix = quaternion_to_matrix(torch.tensor(quaternion))
    translation = translation.reshape(3, 1)

    last_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    transform_matrix = torch.cat((rotation_matrix, translation), dim=1)
    transform_matrix = torch.cat((transform_matrix, last_row), dim=0)
    return transform_matrix


def transform_pcd_with_quat_trans(point_cloud, quaternion, translation):
    transform_matrix = get_matrix_from_quat_trans(quaternion, translation)
    return transform_pcd_with_matrix(point_cloud, transform_matrix)
