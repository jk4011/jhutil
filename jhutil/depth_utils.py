import torch
import numpy as np
from .pcd_utils import show_point_clouds


def depth_to_points(depth_tensor, fov=60):
    """
    Convert depth map to 3D points.

    Parameters:
    depth_tensor (torch.Tensor): Depth map of shape (H, W)
    fov (float): Field of view in degrees
    image_width (int): Width of the image
    image_height (int): Height of the image

    Returns:
    torch.Tensor: 3D points of shape (H, W, 3) where each point has (X, Y, Z) coordinates
    """
    
    image_height, image_width = depth_tensor.shape
    
    # Compute focal length from the field of view
    focal_length = (image_width / 2) / torch.tan(torch.tensor(fov / 2 * (torch.pi / 180)))

    # Create a meshgrid for pixel coordinates
    i, j = torch.meshgrid(torch.arange(image_height), torch.arange(image_width), indexing='ij')

    # Normalize pixel coordinates
    i_normalized = (i - image_height / 2) / focal_length
    j_normalized = (j - image_width / 2) / focal_length

    # Compute 3D coordinates (X, Y, Z)
    Z = depth_tensor
    X = j_normalized * Z
    Y = i_normalized * Z

    # Stack to get (H, W, 3) tensor for 3D points
    points_3d = torch.stack([X, Y, Z], dim=-1)

    # convert into (H*W, 3)
    points_3d = points_3d.view(-1, 3)
    
    return points_3d


def show_depth_3d(depth, rgb, fov=60, subsample=10):
    # n, 3
    points = depth_to_points(depth, fov=fov)
    
    # n, 3
    rgb = rgb.permute(1, 2, 0)
    rgb = rgb.reshape(-1, 3)
    
    show_point_clouds([points[::subsample]], colors=[rgb[::subsample] * 255])
    
    
def show_depth(depth):
    depth = depth.squeeze()
    return get_scale_depth(depth).chans


def get_scale_depth(depth, min=0, max=1):
    depth = depth.squeeze()
    depth_scaled = ((depth - depth.min()) / (depth.max() - depth.min()))
    return depth_scaled * (max - min) + min


def show_gt_dense_and_sparse(depth_gt, depth_sparse):
    depth_gt = depth_gt.squeeze()
    depth_sparse = depth_sparse.squeeze()
    trans = depth_gt.min()
    scale = depth_gt.max() - depth_gt.min()
    depth_scaled = (depth_gt - trans) / scale
    depth_sparse_scaled = (depth_sparse - trans) / scale
    depth_sparse_scaled[depth_sparse == 0] = 0
    return torch.cat([depth_scaled, depth_sparse_scaled], dim=1).chans


def show_depth_all(depth_gt, depth_sparse, depth_pred):
    depth_gt = depth_gt.squeeze()
    depth_sparse = depth_sparse.squeeze()
    depth_pred = depth_pred.squeeze()
    trans = depth_gt.min()
    scale = depth_gt.max() - depth_gt.min()
    depth_scaled = (depth_gt - trans) / scale
    depth_sparse_scaled = (depth_sparse - trans) / scale
    depth_sparse_scaled[depth_sparse == 0] = 0
    return torch.cat([depth_scaled, depth_sparse_scaled, depth_pred], dim=1).chans


# copy from MiDaS
def compute_scale_and_shift(prediction, target, mask):
    if prediction.ndim == 2:
        prediction = prediction[None, :]
    if prediction.ndim == 2:
        target = target[None, :]
    if prediction.ndim == 2:
        mask = mask[None, :]

    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = det.nonzero()

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def get_depth_diff(depth_gt, depth_sparse, depth_pred):
    depth_alligned = get_depth_alligned(depth_pred, depth_sparse)
    return depth_gt - depth_alligned


def get_depth_alligned(depth_pred, depth_gt_sparse):
    if depth_gt_sparse.shape != depth_pred.shape:
        h, w = depth_pred.shape
        depth_gt_sparse = resize_sparse_depth(depth_gt_sparse, h2=h, w2=w)
    assert depth_gt_sparse.shape == depth_pred.shape

    mask = depth_gt_sparse != 0
    scale, shift = compute_scale_and_shift(depth_pred[None, :], depth_gt_sparse[None, :], mask[None, :])
    scaled_depth_pred = depth_pred * scale + shift

    return scaled_depth_pred


def resize_sparse_depth(sparse_depth, h2=576, w2=768):
    device = sparse_depth.device
    h1, w1 = sparse_depth.shape
    assert h1 <= h2 and w1 <= w2, "New shape must be larger than the original shape."

    # Find the indices and values of the non-zero elements
    nz_indices = sparse_depth.nonzero()
    nz_values = sparse_depth[nz_indices[:, 0], nz_indices[:, 1]]

    # Calculate new positions based on linear interpolation
    nz_indices_float = nz_indices.float()
    new_i = nz_indices_float[:, 0] * (h2 - 1) / (h1 - 1)
    new_j = nz_indices_float[:, 1] * (w2 - 1) / (w1 - 1)

    # Round the positions to nearest integer (optional, depending on how you want to handle subpixel locations)
    new_i = new_i.round().long()
    new_j = new_j.round().long()

    # Create a new depth map with zeros
    new_depth = torch.zeros((h2, w2), dtype=sparse_depth.dtype, device=device)

    # Place the sparse values into the new depth map
    new_depth[new_i, new_j] = nz_values

    return new_depth


def get_depth_map(caminfo, pcd, H=None, W=None):
    transformed_pcd = pcd @ caminfo.R + caminfo.T
    X, Y, Z = transformed_pcd.T

    if H is None or W is None:
        H, W, _ = np.array(caminfo.image).shape

    reg_X = (X / Z) / np.tan(caminfo.FovX / 2)
    reg_Y = (Y / Z) / np.tan(caminfo.FovY / 2)
    depths = Z

    x_coors = reg_X * W / 2 + W / 2
    y_coors = reg_Y * H / 2 + H / 2

    depth_map = torch.zeros((H, W))

    for x, y, depth in zip(x_coors, y_coors, depths):
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        if depth_map[round(y), round(x)] == 0:
            depth_map[round(y), round(x)] = depth
        else:
            depth_map[round(y), round(x)] = min(depth_map[round(y), round(x)], depth)

    return depth_map


def inverse_3x4(campose):
    campose = np.concatenate((campose, np.array([[0, 0, 0, 1]])), axis=0)
    inverse_campose = np.linalg.inv(campose)
    return inverse_campose[:3]


def get_pcd(depth_map, pose, fovX=1, fovY=1, is_w2c=True):
    depth_map = torch.tensor(depth_map)
    fovX = torch.tensor(fovX)
    fovY = torch.tensor(fovY)

    # Get the depth map dimensions
    height, width = depth_map.shape

    # Generate pixel coordinates
    x = torch.linspace(0, width - 1, width)
    y = torch.linspace(0, height - 1, height)
    xv, yv = torch.meshgrid(x, y, indexing='xy')

    # Reshape the pixel coordinates
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)

    # Convert pixel coordinates to normalized device coordinates (NDC)
    x_ndc = (xv / (width - 1)) * 2 - 1
    y_ndc = (yv / (height - 1)) * 2 - 1

    # Convert NDC to camera coordinates
    x_cam = x_ndc * torch.tan(fovX / 2)
    y_cam = y_ndc * torch.tan(fovY / 2)
    Z = depth_map.reshape(-1)
    X = x_cam * Z
    Y = y_cam * Z

    # Stack camera coordinates
    cam_coords = torch.stack((X, -Y, -Z, torch.ones(len(X))), dim=1)
    cam_coords = cam_coords[Z != 0]

    if is_w2c:
        pcd = cam_coords @ pose.T
    else:
        pcd = cam_coords @ inverse_3x4(pose).T

    return pcd
