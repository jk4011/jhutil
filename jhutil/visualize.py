import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
from base64 import b64encode
from IPython.display import HTML
from ipywidgets import interact
import io
from .pcd_utils import PALATTE
from einops import rearrange




def show_video(video_path):
    mp4 = open(video_path, "rb").read()
    data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
    return HTML(
        """
    <video width=600 controls>
      <source src="%s" type="video/mp4">
    </video>
    """
        % data_url
    )

def get_mask_diff(pred_mask, gt_mask):
    mask_diff = torch.stack([pred_mask, pred_mask & gt_mask, gt_mask]).float()
    return mask_diff


def show_mask_diff_with_plt(pred_mask, gt_mask):
    mask_diff = get_mask_diff(pred_mask, gt_mask)
    show_image_with_plt(mask_diff)

def show_image_with_plt(img_tensor):
    if img_tensor.ndim == 2:
        raise ValueError("img_tensor should be 3 or 4 dim")
    
    if img_tensor.ndim == 4:
        img_tensor = rearrange(img_tensor, 'b c h w -> h w b c')
        # if it is multi image, show the image with 4 columns
        if img_tensor.shape[0] > 1:
            fig, axes = plt.subplots(1, 4, figsize=(10, 6))
            for i in range(4):
                axes[i].imshow(img_tensor[i])
                axes[i].axis("off")
            plt.show()
        else:
            plt.imshow(img_tensor[0])
            plt.axis("off")
            plt.show()
    elif img_tensor.ndim == 3:
        img_tensor = rearrange(img_tensor, 'c h w -> h w c').cpu().detach().numpy()
        plt.imshow(img_tensor)
        plt.axis("off")
        plt.show()



def show_images(img_paths):
    """
    Display images in a 4-column grid layout

    Args:
        img_paths: List of image paths to display
    """
    if len(img_paths) == 0:
        print("No images found in the list.")
        return

    plt.close("all")

    # Calculate grid dimensions (4 columns)
    n_cols = 4
    n_images = len(img_paths)
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each image
    for idx, img_path in enumerate(img_paths):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        img = Image.open(img_path)
        ax.imshow(np.asarray(img))
        ax.set_title(f"Image {idx}")
        ax.axis('off')

    # Hide unused subplots
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()


def show_matching(
    img0,
    img1,
    mkpts0=torch.empty(0, 2),
    mkpts1=torch.empty(0, 2),
    color=None,
    kpts0=None,
    kpts1=None,
    dpi=75,
    bbox=None,
    skip_line=False,
    linewidth=1,
):
    # Permute channels if tensor is in CxHxW and convert to numpy
    if img0.shape[0] in [3, 4]:
        img0 = img0.permute(1, 2, 0)
        img1 = img1.permute(1, 2, 0)
    if isinstance(img0, torch.Tensor):
        img0 = img0.cpu().detach().numpy()
        img1 = img1.cpu().detach().numpy()
    if isinstance(mkpts0, torch.Tensor):
        mkpts0 = mkpts0.cpu().detach().numpy()
        mkpts1 = mkpts1.cpu().detach().numpy()
    mkpts0 = mkpts0.copy()
    mkpts1 = mkpts1.copy()

    # Generate random colors if none provided
    if color is None:
        generator = torch.Generator()
        color = plt.cm.jet(torch.rand(mkpts0.shape[0], generator=generator))

    # Apply bounding box cropping
    if bbox is not None:
        w_from, h_from, w_to, h_to = bbox
        img0 = img0[h_from:h_to, w_from:w_to]
        img1 = img1[h_from:h_to, w_from:w_to]
        mkpts0[:, 0] -= w_from
        mkpts0[:, 1] -= h_from
        mkpts1[:, 0] -= w_from
        mkpts1[:, 1] -= h_from

    assert mkpts0.shape[0] == mkpts1.shape[0], \
        f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"

    # Create figure with transparent background
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.patch.set_alpha(0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Display images (RGBA or RGB)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    plt.tight_layout(pad=1)

    # Plot keypoints if provided
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    # Plot matches
    if mkpts0.shape[0] > 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        if not skip_line:
            fig.lines = [
                matplotlib.lines.Line2D(
                    (fkpts0[i, 0], fkpts1[i, 0]),
                    (fkpts0[i, 1], fkpts1[i, 1]),
                    transform=fig.transFigure,
                    c=color[i],
                    linewidth=linewidth,
                )
                for i in range(len(mkpts0))
            ]
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # Convert figure to PIL Image, preserving transparency
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img_pil = Image.open(buf).convert("RGBA")
    return img_pil


def show_groups(
    img0,
    img1,
    mkpts0=torch.empty(0, 2),
    mkpts1=torch.empty(0, 2),
    groups=None,
    kpts0=None,
    kpts1=None,
    dpi=75,
    bbox=None,
):
    # Permute channels if tensor is in CxHxW and convert to numpy
    if img0.shape[0] in [3, 4]:
        img0 = img0.permute(1, 2, 0)
        img1 = img1.permute(1, 2, 0)
    if isinstance(img0, torch.Tensor):
        img0 = img0.cpu().detach().numpy()
        img1 = img1.cpu().detach().numpy()
    if isinstance(mkpts0, torch.Tensor):
        mkpts0 = mkpts0.cpu().detach().numpy()
        mkpts1 = mkpts1.cpu().detach().numpy()
    mkpts0 = mkpts0.copy()
    mkpts1 = mkpts1.copy()

    # Apply bounding box cropping
    if bbox is not None:
        w_from, h_from, w_to, h_to = bbox
        img0 = img0[h_from:h_to, w_from:w_to]
        img1 = img1[h_from:h_to, w_from:w_to]
        mkpts0[:, 0] -= w_from
        mkpts0[:, 1] -= h_from
        mkpts1[:, 0] -= w_from
        mkpts1[:, 1] -= h_from

    assert mkpts0.shape[0] == mkpts1.shape[0], \
        f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"

    # Create figure with transparent background
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.patch.set_alpha(0)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Display images (RGBA or RGB)
    axes[0].imshow(img0)
    axes[1].imshow(img1)
    plt.tight_layout(pad=1)

    # Plot keypoints if provided
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    # Plot groups with alpha colors
    if mkpts0.shape[0] > 0 and groups is not None:
        for i, group in enumerate(groups):
            base_color = (PALATTE[i] / 255).tolist()
            rgba_color = base_color + [0.1]
            axes[0].scatter(
                mkpts0[group][:, 0], mkpts0[group][:, 1],
                c=[rgba_color], s=50, marker='s'
            )
            axes[1].scatter(
                mkpts1[group][:, 0], mkpts1[group][:, 1],
                c=[rgba_color], s=50, marker='s'
            )

    # Convert figure to PIL Image, preserving transparency
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    buf.seek(0)
    img_pil = Image.open(buf).convert("RGBA")
    return img_pil
