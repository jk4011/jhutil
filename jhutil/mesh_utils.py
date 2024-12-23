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
from typing import Union
from pyntcloud import PyntCloud
from .pcd_utils import transform_pcd_with_matrix, PALATTE

def _random_rotate(vertex):
    theta_x = random.uniform(0, math.pi * 2)
    theta_y = random.uniform(0, math.pi * 2)
    theta_z = random.uniform(0, math.pi * 2)
    rotation_x = np.array([
        [1, 0, 0],
        [0, math.cos(theta_x), -math.sin(theta_x)],
        [0, math.sin(theta_x), math.cos(theta_x)]
    ])
    rotation_y = np.array([
        [math.cos(theta_y), 0, math.sin(theta_y)],
        [0, 1, 0],
        [-math.sin(theta_y), 0, math.cos(theta_y)]
    ])
    rotation_z = np.array([
        [math.sin(theta_z), math.cos(theta_z), 0],
        [0, 0, 1]
    ])
    rotation = np.matmul(rotation_z, np.matmul(rotation_y, rotation_x))
    vertex = np.matmul(vertex, rotation)
    return vertex


def show_multiple_objs(obj_files, colors=None, is_random_rotate=False, library="go", transformations=None, scale=1):
    if transformations is not None:
        assert len(obj_files) == len(transformations)
    else:
        transformations = [torch.eye(4, dtype=torch.float32)] * len(obj_files)

    if library == "meshplot":
        import meshplot as mp
        v, f = read_obj_file(obj_files[0])
        p = mp.plot(v, f)

        for i, obj_file in enumerate(obj_files):
            if i == 0:
                continue
            vertices, faces = read_obj_file(obj_file)
            vertices *= scale
            if is_random_rotate:
                vertices = _random_rotate(vertices)
            # color = colors[i] if colors and len(colors) > i else None
            color = np.array(colors[i]) if colors and len(colors) > i else None
            p.add_mesh(vertices, faces, c=color)  # shading={"wireframe": True}
    elif library == "go":
        meshes = []

        for idx, obj_file in enumerate(obj_files):
            vertices, faces = read_obj_file(obj_file)
            vertices *= scale
            vertices = transform_pcd_with_matrix(torch.tensor(vertices), transformations[idx])
            x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

            def rgb_to_hex(rgb):
                return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
            if colors is None:
                color = rgb_to_hex(PALATTE[idx % len(PALATTE)])
            else:
                color = rgb_to_hex(colors[idx])
            mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k, color=color)
            meshes.append(mesh)

        layout = go.Layout(
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                xaxis=dict(visible=False),  # Disable x-axis
                yaxis=dict(visible=False),  # Disable y-axis
                zaxis=dict(visible=False)   # Disable z-axis
            ),
        )

        fig = go.Figure(data=meshes, layout=layout)
        fig.show()


def show_obj(obj_file, color=[1, 0, 0], library="go"):
    vertices, faces = read_obj_file(obj_file)
    if library == "meshplot":
        import meshplot as mp
        mp.plot(vertices, faces, c=color)
    elif library == "go":
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        mesh = go.Mesh3d( 
            x=x, y=y, z=z, i=i, j=j, k=k, 
            opacity=0.5,
        )

        fig = go.Figure(data=[mesh])
        fig.update_layout(
            scene=dict(
                xaxis_title='X Axis',
                yaxis_title='Y Axis',
                zaxis_title='Z Axis'
            ),
            title='3D Mesh Visualization'
        )

        fig.show()


def read_obj_file(file_path):
    mesh = trimesh.load(file_path)
    vertices = mesh.vertices
    faces = mesh.faces

    return vertices, faces


def show_meshes(folder_path, indices=None):
    file_list = os.listdir(folder_path)
    file_list = [os.path.join(folder_path, file) for file in file_list]
    if indices is None:
        show_multiple_objs(file_list)
    else:
        file_list = [file_list[idx] for idx in indices]
        show_multiple_objs(file_list, PALATTE[indices])

    jhutil.jhprint(0000, file_list, endline='\n', list_in_one_line=False)


def get_mesh_intersection(mesh1, mesh2, file_loc):
    # Save intersection of two meshes
    import pymeshlab
    ms = pymeshlab.MeshSet()

    # Add the input meshes to the MeshSet
    ms.load_new_mesh(mesh1)
    ms.load_new_mesh(mesh2)

    # Apply the boolean intersection filter
    ms.apply_filter(
        'generate_boolean_intersection',
        first_mesh=0,
        second_mesh=1,
        transfer_face_color=False,
        transfer_face_quality=False,
        transfer_vert_color=False,
        transfer_vert_quality=False)

    ms.save_current_mesh(file_loc)


def calculate_mesh_volume(mesh_file):
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)
    return mesh.volume
