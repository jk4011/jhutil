import numpy as np
import plotly.graph_objs as go
import numpy as np
import trimesh
import numpy as np
import matplotlib.pyplot as plt


def sample_point_cloud_from_mesh(mesh_file, num_points):
    # Load the mesh
    mesh = trimesh.load_mesh(mesh_file)

    # Sample the point cloud
    point_cloud, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    return point_cloud


def show_point_clouds(point_clouds, colors, normals=None, is_random_rotate=False, s=None, range=((-1.1), (-1.1), (-1.1))):
    # type check point_clouds is list
    if isinstance(point_clouds, list):
        point_clouds = np.array(point_clouds)
    if s is None:
        s = [0.3] * len(point_clouds)


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