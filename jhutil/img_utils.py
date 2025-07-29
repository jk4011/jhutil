import torch
import math
from .freq_utils import save_img

def get_masked_image(img, mask, blur_ratio=0.7, mask_color=[1., 1., 1.]):
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img, dtype=torch.float32)
    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.bool)
    
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).repeat(img.shape[0], 1, 1)
    masked_img = img.clone()
    mask_color = torch.tensor(mask_color).repeat((~mask).sum() // 3)
    masked_img[~mask] = (1 - blur_ratio) * masked_img[~mask] + blur_ratio * mask_color
    return masked_img


def rgb(image, subsample=1):
    if image.dim() == 3:
        if image.shape[0] in [3, 4]:
            image = image.permute(2, 0, 1)
        return image[:, ::subsample, ::subsample].rgb
    elif image.dim() == 4:
        if image.shape[-1] in [3, 4]:
            image = image.permute(0, 3, 1, 2)
        return image[:, :, ::subsample, ::subsample].rgb

def get_bbox(alpha: torch.Tensor):
    nonzero_coords = (alpha > 0).nonzero(as_tuple=False)
    if nonzero_coords.numel() == 0:
        return None  # fully transparent

    min_y, min_x = (
        nonzero_coords[:, 0].min().item(),
        nonzero_coords[:, 1].min().item(),
    )
    max_y, max_x = (
        nonzero_coords[:, 0].max().item(),
        nonzero_coords[:, 1].max().item(),
    )

    return (min_x, min_y, max_x + 1, max_y + 1)


def crop_two_image_with_alpha(img1: torch.Tensor, img2: torch.Tensor):
    assert img1.dim() == 3
    assert img2.dim() == 3

    # 1. Extract alpha channel (4th channel) and compute bounding boxes
    alpha1 = img1[3]
    alpha2 = img2[3]
    bbox1 = get_bbox(alpha1)
    bbox2 = get_bbox(alpha2)

    # 2. Determine the union bounding box
    if bbox1 is None and bbox2 is None:
        return img1, img2
    elif bbox1 is None:
        union_bbox = bbox2
    elif bbox2 is None:
        union_bbox = bbox1
    else:
        # Union bounding box
        left1, top1, right1, bottom1 = bbox1
        left2, top2, right2, bottom2 = bbox2
        union_left = min(left1, left2)
        union_top = min(top1, top2)
        union_right = max(right1, right2)
        union_bottom = max(bottom1, bottom2)
        union_bbox = (union_left, union_top, union_right, union_bottom)

    # 3. Crop images using the union bounding box
    left, top, right, bottom = union_bbox
    # Note: For slicing [C, H, W], we slice in [ : , top:bottom, left:right]
    cropped_img1 = img1[:, top:bottom, left:right]
    cropped_img2 = img2[:, top:bottom, left:right]

    return union_bbox, cropped_img1, cropped_img2


def crop_image_with_alpha(img: torch.Tensor):
    assert img.dim() == 3

    alpha = img[3]
    bbox = get_bbox(alpha)
    cropped_image = img[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
    return bbox, cropped_image


def crop_two_image_with_background(
    img1: torch.Tensor, img2: torch.Tensor, background="black", shape='chw'
):
    assert img1.dim() == 3
    assert img2.dim() == 3
    assert shape in ["chw", "hwc"]

    if shape == "hwc":
        img1 = img1.permute(2, 0, 1)
        img2 = img2.permute(2, 0, 1)

    assert img1.shape[0] == 3
    assert background in ["black", "white"]

    if background == "black":
        alpha1 = 1 - (img1 == 0).all(dim=0).int()
        alpha2 = 1 - (img2 == 0).all(dim=0).int()
    elif background == "white":
        alpha1 = 1 - (img1 == 1).all(dim=0).int()
        alpha2 = 1 - (img2 == 1).all(dim=0).int()

    img1_alpha = torch.cat([img1, alpha1.unsqueeze(0)], dim=0)
    img2_alpha = torch.cat([img2, alpha2.unsqueeze(0)], dim=0)

    union_bbox, cropped_img1, cropped_img2 = crop_two_image_with_alpha(img1_alpha, img2_alpha)
    cropped_img1 = cropped_img1[:3]
    cropped_img2 = cropped_img2[:3]

    if shape == "hwc":
        cropped_img1 = cropped_img1.permute(1, 2, 0)
        cropped_img2 = cropped_img2.permute(1, 2, 0)

    return union_bbox, cropped_img1, cropped_img2


def crop_image_with_background(img: torch.Tensor, background="black"):
    assert img.dim() == 3

    if background == "black":
        alpha = 1 - (img == 0).all(dim=0).int()
    elif background == "white":
        alpha = 1 - (img == 1).all(dim=0).int()
    bbox = get_bbox(alpha)
    return img[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]


def show_xy(xy):
    assert xy.shape[1] == 2
    assert xy.dim() == 2

    xy_long = xy.long()
    w = torch.max(xy_long[:, 0]).item() + 1
    h = torch.max(xy_long[:, 1]).item() + 1

    plane = torch.zeros((h, w), dtype=torch.float32)
    coord = [xy_long[:, 1], xy_long[:, 0]]
    plane[coord] = True

    return plane.chans


def save_motion_img(
    images: torch.Tensor,
    path=None,
    *,
    alpha_range=(0.5, 1.0),
    alpha_mode="exponential",
    stride: int = 0,
) -> torch.Tensor:
    # 1) (N, H, W, 4) → (N, 4, H, W)
    if images.ndim == 4 and images.shape[-1] in (3, 4):
        images = images.permute(0, 3, 1, 2).contiguous()

    N, C, H, W = images.shape
    if C != 4:
        raise ValueError("Input must be RGBA (4-channel) images")

    if alpha_mode not in ("linear", "exponential"):
        raise ValueError("alpha_mode must be 'linear' or 'exponential'")

    # 2) per-image alpha scale (oldest → newest)
    if N == 1:
        weights = torch.tensor([1.0], device=images.device)
    else:
        if alpha_mode == "linear":
            weights = torch.linspace(
                alpha_range[0], alpha_range[1], steps=N, device=images.device
            )
        else:  # exponential
            t = torch.linspace(
                alpha_range[0], alpha_range[1], steps=N, device=images.device
            )
            weights = (torch.exp(t) - 1.0) / (math.e - 1.0)

    # 3) 캔버스 폭 = W + stride*(N-1)  (이미지 잘리지 않도록)
    W_canvas = W + max(stride, 0) * (N - 1)
    canvas = torch.zeros((4, H, W_canvas), dtype=images.dtype, device=images.device)

    # 4) Older → Newer 순으로 blending (좌→우 이동)
    for i in range(N):
        img      = images[i]                # (4, H, W)
        alpha    = img[3] * weights[i]      # (H, W)
        alpha_ex = alpha.unsqueeze(0)       # (1, H, W) for broadcasting

        x0 = stride * i                     # 왼쪽 시작 위치
        x1 = x0 + W                         # 오른쪽 끝 위치
        # canvas slice와 블렌딩
        canvas[:, :, x0:x1] = img * alpha_ex + canvas[:, :, x0:x1] * (1.0 - alpha_ex)

    if path is not None:
        save_img(canvas, path)
    return canvas
