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
import plotly.io as pio

# VS Code에서 외부 브라우저로 띄우고 싶다면 한 줄 추가
pio.renderers.default = "browser"


PALATTE = np.array(
    [
        [204, 0, 0],
        [204, 204, 0],
        [0, 204, 0],
        [76, 76, 204],
        [127, 76, 127],
        [0, 127, 127],
        [76, 153, 0],
        [153, 0, 76],
        [76, 0, 153],
        [153, 76, 0],
        [76, 0, 153],
        [153, 0, 76],
        [204, 51, 127],
        [204, 51, 127],
        [51, 204, 127],
        [51, 127, 204],
        [127, 51, 204],
        [127, 204, 51],
        [76, 76, 178],
        [76, 178, 76],
        [178, 76, 76],
    ]
)
PALATTE = np.concatenate([PALATTE for _ in range(10)], axis=0)

# append black color at last
PALLETE = np.concatenate([PALATTE, np.array([[0, 0, 0]])], axis=0)


def _to_cpu_nparray(t):
    """torch.Tensor → np.ndarray(cpu) 변환 헬퍼"""
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def show_point_clouds(
    point_clouds: List[torch.Tensor], colors=None, show_indices=None, framework="plotly", point_size=2
):

    if framework == "plotly":
        np_pcls = [_to_cpu_nparray(p) for p in point_clouds]

        # ② 시각화 대상 선택
        if show_indices is None:
            show_indices = range(len(np_pcls))
        show_indices = list(show_indices)

        vis_pcls = [np_pcls[i] for i in show_indices]
        n_points = [len(p) for p in vis_pcls]
        xyz = np.concatenate(vis_pcls, axis=0)  # (N, 3)

        # ③ 색상 준비
        if colors is None:
            palette = PALATTE[show_indices]  # (#pcd, 3)
            rgb_arr = np.repeat(palette, n_points, axis=0)  # (N, 3)
        else:
            rgb_arr = np.concatenate(colors, axis=0)  # (N, 3)

        assert xyz.shape == rgb_arr.shape, f"{xyz.shape=} {rgb_arr.shape=}"

        rgb_strings = [f"rgb({r},{g},{b})" for r, g, b in rgb_arr.astype(int)]

        # ④ Plotly Figure 생성
        fig = go.Figure(
            data=go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=rgb_strings),
            )
        )
        # 동일 축 비율(option)
        fig.update_layout(scene=dict(aspectmode="data"))

        # ⑤ 시각화
        fig.show()

    else:

        for i in range(len(point_clouds)):
            pcd = point_clouds[i]
            if isinstance(pcd, torch.Tensor):
                point_clouds[i] = pcd.detach().cpu()

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

        assert (
            colors.shape == point_clouds.shape
        ), f"colors.shape: {colors.shape}, point_clouds.shape: {point_clouds.shape}"
        result["red"] = colors[:, 0]
        result["green"] = colors[:, 1]
        result["blue"] = colors[:, 2]

        PyntCloud(result).plot(
            return_scene=True, backend="pythreejs", initial_point_size=0.01
        )


def show_color_pcds(
    pcl_list,  # List[torch.Tensor] | (Ni, 3)
    colors=None,  # Optional[List[np.ndarray]] | (Ni, 3)
    show_indices=None,  # Optional[Iterable[int]]
    point_size=2,  # 마커 크기(px)
):
    """
    Plotly로 컬러 포인트 클라우드 시각화
    Args:
        pcl_list  : point cloud Tensor들의 리스트 [(Ni, 3), ...]
        colors    : 각 point cloud에 대응하는 RGB 배열 리스트  [(Ni, 3), ...]  (uint8, 0~255)
                     None이면 PALATTE에서 자동 할당
        show_indices : 시각화할 pcl 인덱스(미지정 시 전체)
        point_size   : 포인트 표시 크기(px)
    """
    # 1) 텐서를 CPU numpy로 변환
    np_pcls = []
    for pcl in pcl_list:
        if isinstance(pcl, torch.Tensor):
            pcl = pcl.detach().cpu().numpy()
        np_pcls.append(pcl)  # (Ni, 3)
        
    # 2) 시각화 대상 선택
    if show_indices is None:
        show_indices = range(len(np_pcls))
    show_indices = list(show_indices)

    vis_pcls = [np_pcls[i] for i in show_indices]
    len_each = [len(p) for p in vis_pcls]
    xyz = np.concatenate(vis_pcls, axis=0)  # (N, 3)

    # 3) 색상 준비
    if colors is None:
        palette = PALATTE[show_indices]  # (#shown, 3)
        rgb_arr = np.repeat(palette, len_each, axis=0)  # (N, 3)
    else:
        np_colors = []
        for color in colors:
            if isinstance(color, torch.Tensor):
                color = color.detach().cpu().numpy()
            np_colors.append(color)
        rgb_arr = np.concatenate(np_colors, axis=0)  # (N, 3)
        if rgb_arr.max() < 1.0:
            rgb_arr = (rgb_arr * 255).astype(np.uint8)

    assert xyz.shape == rgb_arr.shape, f"{xyz.shape=} {rgb_arr.shape=}"

    # 4) 'rgb(r,g,b)' 문자열로 변환
    rgb_strings = [f"rgb({r},{g},{b})" for r, g, b in rgb_arr.astype(int)]

    # 5) Plotly Scatter3d 생성
    fig = go.Figure(
        data=go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode="markers",
            marker=dict(size=point_size, color=rgb_strings),
        )
    )
    # 축 비율 고정(옵션)
    fig.update_layout(scene=dict(aspectmode="data"))

    # 6) 시각화
    fig.show()


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
    transformed_point_cloud_h = torch.matmul(
        transform_matrix.float(), point_cloud_h.float()
    )

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
