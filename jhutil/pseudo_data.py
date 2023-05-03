import torch
from torch.utils.data import Dataset, DataLoader
import random

class GeoTransformerPseudoData(Dataset):
    def __len__(self):
        return 10
    
    def __getitem__(self, idx):
        num_ref = random.randint(5000, 10000)
        num_src = random.randint(5000, 10000)
        data = {
            "scene_name": f"hello_{idx}",
            "ref_frame": 0,
            "src_frame": idx+1,
            "overlap": 0.7996478089653904,
            "ref_points": torch.randn(num_ref, 3),# "array[18977, 3] f32 n=56931 (0.2Mb) x∈[-1.446, 3.482] μ=0.634 σ=1.356",
            "src_points": torch.randn(num_src, 3),# "array[19082, 3] f32 n=57246 (0.2Mb) x∈[-1.434, 3.302] μ=0.599 σ=1.253",
            "ref_feats": torch.ones(num_ref, 1),# "array[18977, 1] f32 74Kb x∈[1.000, 1.000] μ=1.000 σ=0.",
            "src_feats": torch.ones(num_src, 1),# "array[19082, 1] f32 75Kb x∈[1.000, 1.000] μ=1.000 σ=0.",
            "transform": torch.randn(4, 4),# "array[4, 4] f32 n=16 x∈[-0.116, 1.000] μ=0.247 σ=0.436"
        }
        return data