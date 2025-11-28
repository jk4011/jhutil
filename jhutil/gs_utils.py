import torch

def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to Spherical Harmonics degree 0 (SH0)

    Args:
        rgb (torch.Tensor): RGB tensor of shape (B, 3, H, W) or (N, 3)
            - (B, 3, H, W): Batch of images with channels first
            - (N, 3): Point cloud colors

    Returns:
        torch.Tensor: SH0 tensor of shape (N, 1, 3)
            where N = B*H*W for image input or N for point cloud input
    """
    C0 = 0.28209479177387814

    if rgb.ndim == 4:
        # (B, 3, H, W) -> (N, 3)
        B, C, H, W = rgb.shape
        assert C == 3, f"Expected 3 channels, got {C}"
        # Permute to (B, H, W, 3) then reshape to (N, 3)
        rgb = rgb.permute(0, 2, 3, 1).reshape(-1, 3)
    elif rgb.ndim == 2:
        # (N, 3) -> keep as is
        assert rgb.shape[1] == 3, f"Expected shape (N, 3), got {rgb.shape}"
    else:
        raise ValueError(f"Expected rgb of shape (B, 3, H, W) or (N, 3), got {rgb.shape}")

    # Convert RGB to SH0: sh0 = (rgb - 0.5) / C0
    sh0 = (rgb - 0.5) / C0

    # Reshape to (N, 1, 3)
    sh0 = sh0.unsqueeze(1)

    return sh0

