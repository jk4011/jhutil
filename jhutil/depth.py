import torch


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
