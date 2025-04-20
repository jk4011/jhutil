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


def show_images(img_paths):
    def plot_img(img_paths, index=0):
        img_path = img_paths[index]
        img = Image.open(img_path)
        plt.imshow(np.asarray(img))
        plt.axis("off")
        plt.show()

    # Create an interactive slider to browse through the images
    if len(img_paths) > 0:
        interact(plot_img, img_paths=[img_paths], index=(0, len(img_paths) - 1))
    else:
        print("No images found in the list.")


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

    if color is None:
        generator = torch.Generator()  # for fixing seed
        color = plt.cm.jet(torch.rand(mkpts0.shape[0], generator=generator))

    if bbox is not None:
        w_from, h_from, w_to, h_to = bbox
        img0 = img0[h_from:h_to, w_from:w_to]
        img1 = img1[h_from:h_to, w_from:w_to]
        mkpts0[:, 0] = mkpts0[:, 0] - w_from
        mkpts0[:, 1] = mkpts0[:, 1] - h_from
        mkpts1[:, 0] = mkpts1[:, 0] - w_from
        mkpts1[:, 1] = mkpts1[:, 1] - h_from

    assert (
        mkpts0.shape[0] == mkpts1.shape[0]
    ), f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
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

    # Convert figure to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    img_pil = Image.open(buf)

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

    if bbox is not None:
        w_from, h_from, w_to, h_to = bbox
        img0 = img0[h_from:h_to, w_from:w_to]
        img1 = img1[h_from:h_to, w_from:w_to]
        mkpts0[:, 0] = mkpts0[:, 0] - w_from
        mkpts0[:, 1] = mkpts0[:, 1] - h_from
        mkpts1[:, 0] = mkpts1[:, 0] - w_from
        mkpts1[:, 1] = mkpts1[:, 1] - h_from

    assert (
        mkpts0.shape[0] == mkpts1.shape[0]
    ), f"mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}"
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap="gray")
    axes[1].imshow(img1, cmap="gray")
    for i in range(2):  # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c="w", s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c="w", s=2)

    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        for i in range(len(groups)):
            group = groups[i]
            color = (PALATTE[i] / 255).tolist()
            color = color + [0.1]
            fig.canvas.draw()

            axes[0].scatter(mkpts0[group][:, 0], mkpts0[group][:, 1], c=color, s=50, marker='s')
            axes[1].scatter(mkpts1[group][:, 0], mkpts1[group][:, 1], c=color, s=50, marker='s')

    # Convert figure to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format="PNG", bbox_inches="tight", pad_inches=0)
    plt.close(fig)  # Close the figure to free memory
    buf.seek(0)
    img_pil = Image.open(buf)

    return img_pil
