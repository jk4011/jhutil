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


PALATTE = np.array([[0, 204, 0], [204, 0, 0], [0, 0, 204], [127, 127, 0], [127, 0, 127], [0, 127, 127], [76, 153, 0], [153, 0, 76], [76, 0, 153], [153, 76, 0], [76, 0, 153], [
    153, 0, 76], [204, 51, 127], [204, 51, 127], [51, 204, 127], [51, 127, 204], [127, 51, 204], [127, 204, 51], [76, 76, 178], [76, 178, 76], [178, 76, 76]])


def sample_point_cloud_from_mesh(mesh_file, num_points):
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Sample the point cloud
    point_cloud, face_indices = trimesh.sample.sample_surface(mesh, num_points)

    return point_cloud


def show_point_clouds(point_clouds, colors=None):
    n_pcd = len(point_clouds)
    lens = [len(pcd) for pcd in point_clouds]
    point_clouds = np.concatenate(point_clouds, axis=0)
    
    if colors is None:
        palatte = PALATTE[:n_pcd]
        colors = np.repeat(palatte, lens, axis=0)
    
    result = pd.DataFrame()
    result["x"] = point_clouds[:, 0]
    result["y"] = point_clouds[:, 1]
    result["z"] = point_clouds[:, 2]

    # TODO: change it into log (debug)
    import jhutil; jhutil.jhprint(1111, colors.shape)
    import jhutil; jhutil.jhprint(2222, point_clouds.shape)
    
    assert colors.shape == point_clouds.shape
    result["red"] = colors[:, 0]
    result["green"] = colors[:, 1]
    result["blue"] = colors[:, 2]
    
    PyntCloud(result).plot(return_scene=True, backend="pythreejs", initial_point_size=0.5)


def _random_rotate(vertices):
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
    vertices = np.matmul(vertices, rotation)
    return vertices


def show_multiple_objs(obj_files, colors=None, is_random_rotate=False, library="go", transformations=None, scale=1):
    if transformations is not None:
        assert len(obj_files) == len(transformations)
    else:
        transformations = [torch.eye(4)] * len(obj_files)

    if library == "meshplot":
        import meshplot as mp
        v, f = parse_obj_file(obj_files[0])
        p = mp.plot(v, f)

        for i, obj_file in enumerate(obj_files):
            if i == 0:
                continue
            vertices, faces = parse_obj_file(obj_file)
            vertices *= scale
            if is_random_rotate:
                vertices = _random_rotate(vertices)
            # color = colors[i] if colors and len(colors) > i else None
            color = np.array(colors[i]) if colors and len(colors) > i else None
            p.add_mesh(vertices, faces, c=color)  # shading={"wireframe": True}
    elif library == "go":
        meshes = []

        for idx, obj_file in enumerate(obj_files):
            vertices, faces = parse_obj_file(obj_file)
            vertices *= scale
            vertices = matrix_transform(transformations[idx], torch.tensor(vertices))
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


def parse_obj_file(filename):
    positions = []
    faces = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('v '):
                # Extract vertex position
                parts = line.split()
                position = [float(parts[1]), float(parts[2]), float(parts[3])]
                positions.append(position)
            elif line.startswith('f '):
                # Extract face indices
                parts = line.split()
                # Subtract 1 because OBJ indices are 1-based
                v1 = int(parts[1].split('/')[0]) - 1
                v2 = int(parts[2].split('/')[0]) - 1
                v3 = int(parts[3].split('/')[0]) - 1
                faces.append([v1, v2, v3])

    positions = np.array(positions, dtype=np.float32)
    faces = np.array(faces, dtype=np.int32)

    return positions, faces


def show_obj(obj_file, color=[1, 0, 0], library="go"):
    vertices, faces = parse_obj_file(obj_file)
    if library == "meshplot":
        import meshplot as mp
        mp.plot(vertices, faces, c=color)
    elif library == "go":
        x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
        i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

        mesh = go.Mesh3d(x=x, y=y, z=z, i=i, j=j, k=k)
        layout = go.Layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                                      aspectmode='data',
                                      aspectratio=dict(x=1, y=1, z=1)))

        fig = go.Figure(data=[mesh], layout=layout)
        fig.show()


def show_meshes(folder_path, indices=None):
    file_list = os.listdir(folder_path)
    file_list = [os.path.join(folder_path, file) for file in file_list]
    if indices is None:
        show_multiple_objs(file_list)
    else:
        file_list = [file_list[idx] for idx in indices]
        show_multiple_objs(file_list, PALATTE[indices])

    jhutil.jhprint(0000, file_list, endline='\n', list_one_line=False)


# transformation

def matrix_transform(transform_matrix, point_cloud):
    # Convert the point cloud to homogeneous coordinates
    num_points = point_cloud.shape[0]
    ones = torch.ones((num_points, 1), dtype=torch.float)
    point_cloud_h = torch.cat((point_cloud, ones), dim=1).t()

    # Perform the transformation
    transformed_point_cloud_h = torch.matmul(transform_matrix, point_cloud_h)

    # Convert back to 3D coordinates
    transformed_point_cloud = transformed_point_cloud_h[:3, :].t()

    return transformed_point_cloud


def matrix_from_quat_trans(quaternion, translation):
    from pytorch3d.transforms import quaternion_to_matrix
    rotation_matrix = quaternion_to_matrix(torch.tensor(quaternion))
    translation = translation.reshape(3, 1)

    last_row = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    transform_matrix = torch.cat((rotation_matrix, translation), dim=1)
    transform_matrix = torch.cat((transform_matrix, last_row), dim=0)
    return transform_matrix


def quat_trans_transform(quaternion, translation, point_cloud):
    transform_matrix = matrix_from_quat_trans(quaternion, translation)
    return matrix_transform(transform_matrix, point_cloud)


def mesh_intersection(mesh1, mesh2, file_loc):
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
