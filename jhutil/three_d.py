import numpy as np
import plotly.graph_objs as go
import trimesh
import numpy as np
import matplotlib.pyplot as plt
import torch


def sample_point_cloud_from_mesh(mesh_file, num_points):
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Sample the point cloud
    point_cloud, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    return point_cloud


def show_point_clouds(point_clouds, colors=None, normals=None, is_random_rotate=False, s=None):
    
    if s is None:
        s = [0.3] * len(point_clouds)
        
    if isinstance(point_clouds, torch.Tensor):
        point_clouds = point_clouds.numpy()
        
    pcds_concated = np.concatenate(point_clouds, axis=0)
    
    mean = pcds_concated.mean(axis=0).tolist()
    std = pcds_concated.std()
    range = [[mean[0] - std * 2, mean[0] + std * 2], 
            [mean[1] - std * 2, mean[1] + std * 2],
            [mean[2] - std * 2, mean[2] + std * 2]]

    if colors is None:
        def rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
        colors = [[0, 204, 0], [204, 0, 0], [0, 0, 204], [127, 127, 0], [127, 0, 127], [0, 127, 127], [76, 153, 0], [153, 0, 76], [76, 0, 153], [153, 76, 0], [76, 0, 153], [153, 0, 76], [204, 51, 127], [204, 51, 127], [51, 204, 127], [51, 127, 204], [127, 51, 204], [127, 204, 51], [76, 76, 178], [76, 178, 76], [178, 76, 76]]
        colors = [rgb_to_hex(color) for color in colors]
        colors = colors[:len(point_clouds)]
    
    # Set up the figure and axis for the 3D plot
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')

    if normals is None:
        # Scatter plot for the 3D coordinates
        for points, color, s_ in zip(point_clouds, colors, s):
            if is_random_rotate:
                points = random_rotate(points)
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], '.', color=color, s=s_)
    else:
        # Scatter plot for the 3D coordinates
        for points, normal, color, s_ in zip(point_clouds, normals, colors, s):
            if is_random_rotate:
                points = random_rotate(points)
            
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], '.', color=color, s=s_)
            ax.quiver(points[:, 0], points[:, 1], points[:, 2], normal[:, 0], normal[:, 1], normal[:, 2], length=0.05 ,normalize=True, color=color, alpha=0.2)
            location_mean = points.mean(axis=0)
            normal_mean = normal.mean(axis=0)
            ax.quiver(location_mean[0], location_mean[1], location_mean[2], 
                      normal_mean[0], normal_mean[1], normal_mean[2], length=0.5, color=color, alpha=1)

    # Set labels for the axis
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    # Set the range for each axis
    ax.set_xlim(range[0])
    ax.set_ylim(range[1])
    ax.set_zlim(range[2])
    
    # Display the plot
    plt.show()