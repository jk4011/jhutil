import torch


def show_depth(depth_map):
    return ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())).chans


def show_depth_and_sparse(depth_map, depth_map_sparse):
    trans = depth_map.min()
    scale = depth_map.max() - depth_map.min()
    depth_map_scaled = (depth_map - trans) / scale
    depth_map_sparse_scaled = (depth_map_sparse - trans) / scale
    depth_map_sparse_scaled[depth_map_sparse == 0] = 0
    return torch.cat([depth_map_scaled, depth_map_sparse_scaled], dim=1).chans
