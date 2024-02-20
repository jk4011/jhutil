import torch

def show_depth(depth_map):
    return ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())).chans