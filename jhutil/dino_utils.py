import io
import requests
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, Dinov2Model
from .freq_utils import cache_output

device = "cuda" if torch.cuda.is_available() else "cpu"

processor, model = None, None
upsampler = None

def load_dino():
    global processor, model
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", use_fast=True, token=False)
    model = Dinov2Model.from_pretrained("facebook/dinov2-base", token=False).to(device).eval()

def load_upsampler():
    global upsampler
    upsampler = torch.hub.load("wimmerth/anyup", "anyup", verbose=False).to(device).eval()


# @cache_output(func_name="_dino_inference", override=False)
@torch.no_grad()
def _dino_inference(image_path_list, subsample_ratio=1, upsample=False):
    if isinstance(image_path_list, str):
        image_path_list = [image_path_list]

    images = []
    for img_path in image_path_list:
        img = Image.open(img_path)

        W, H = img.size
        # resize image
        Hr, Wr = (H // (14 * subsample_ratio)) * 14, (W // (14 * subsample_ratio)) * 14
        h, w = Hr // 14, Wr // 14
        img = img.resize((Wr, Hr))
        images.append(img)
    
    if processor is None:
        load_dino()

    inputs = processor(
        images=images,
        do_resize=True,
        size={"shortest_edge": min(Hr, Wr)},
        do_center_crop=True,
        crop_size={"height": Hr, "width": Wr},
        return_tensors="pt",
    )
    hr_image = inputs["pixel_values"].to(device)  # (1, 3, 448, 448), ImageNet-normalized

    with torch.no_grad():
        out = model(pixel_values=hr_image)
        tokens = out.last_hidden_state[:, 1:, :]  # drop [CLS]

    B, N, C = tokens.shape
    assert h * w == N, (h, w, N)
    lr_features = tokens.reshape(B, h, w, C).permute(0, 3, 1, 2).contiguous()  # (1, C, 32, 32)

    if not upsample:
        return lr_features, None

    if upsampler is None:
        load_upsampler()

    with torch.no_grad():
        hr_features = upsampler(hr_image, lr_features, q_chunk_size=256)  # (1, C, 448, 448)

    return lr_features.cpu(), hr_features.cpu()


@torch.no_grad()
def dino_inference(image_path_list, subsample=1, upsample=False, visualize=False, load_gpu=True):
    lr_features, hr_features = _dino_inference(image_path_list, subsample, upsample)
    B, C, h, w = lr_features.shape
    
    # 4) Joint PCA
    if visualize:
        if upsample:
            B, C, Hr, Wr = hr_features.shape
            
            lr_flat = lr_features[0].permute(1, 2, 0).reshape(-1, C)
            hr_flat = hr_features[0].permute(1, 2, 0).reshape(-1, C)
            all_feats = torch.cat([lr_flat, hr_flat], dim=0)

            mean = all_feats.mean(dim=0, keepdim=True)
            X = all_feats - mean

            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            pcs = Vh[:3].T

            proj_all = X @ pcs

            # Split back and min-max normalize jointly for consistent coloring
            n_lr = h * w
            proj_lr = proj_all[:n_lr].reshape(h, w, 3)
            proj_hr = proj_all[n_lr:].reshape(Hr, Wr, 3)

            cmin = proj_all.min(dim=0).values
            cmax = proj_all.max(dim=0).values
            crng = (cmax - cmin).clamp(min=1e-6)

            lr_rgb = ((proj_lr - cmin) / crng).cpu().numpy()
            hr_rgb = ((proj_hr - cmin) / crng).cpu().numpy()

            # 5) Plot
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            axs[0].imshow(lr_rgb)
            axs[0].set_title(f"LR DINOv2 tokens ({h}×{w})")
            axs[0].axis("off")
            axs[1].imshow(hr_rgb)
            axs[1].set_title(f"AnyUp upsampled ({Hr}×{Wr})")
            axs[1].axis("off")
            plt.tight_layout()
            plt.show()

        else:
            # visualize only lr_features
            lr_flat = lr_features[0].permute(1, 2, 0).reshape(-1, C)
            
            mean = lr_flat.mean(dim=0, keepdim=True)
            X = lr_flat - mean
            
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            pcs = Vh[:3].T
            
            proj_lr = X @ pcs
            
            # Min-max normalize
            cmin = proj_lr.min(dim=0).values
            cmax = proj_lr.max(dim=0).values
            crng = (cmax - cmin).clamp(min=1e-6)
            
            lr_rgb = ((proj_lr - cmin) / crng).reshape(h, w, 3).cpu().numpy()
            
            # Plot
            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            ax.imshow(lr_rgb)
            ax.set_title(f"LR DINOv2 tokens ({h}×{w})")
            ax.axis("off")
            plt.tight_layout()
            plt.show()
    
    if load_gpu:
        lr_features = lr_features.cuda()
        if hr_features is not None:
            hr_features = hr_features.cuda()
    
    return lr_features, hr_features