import os
import torch
import argparse
import time
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np
import cv2
import functools
import hashlib


def load_img(path, downsample=1):
    img = ToTensor()(Image.open(path))
    if downsample > 1:
        img = img[:, ::downsample, ::downsample]
    return img


def save_img(tensor, path):
    tensor = tensor.squeeze().detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] in [3, 4]:
        tensor = tensor.permute(1, 2, 0)

    tensor = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(tensor).save(path)


def convert_to_gif(image_folder, fps=50):
    output_path = os.path.join(image_folder, "0_output.gif")
    try:
        duration = int(1000 / fps)

        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(".png")]
        )

        if not image_files:
            raise ValueError("No PNG files found in the specified folder.")

        images = [Image.open(os.path.join(image_folder, img)) for img in image_files]

        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
        )
        print(f"GIF saved to {output_path} with {fps} FPS.")

    except Exception as e:
        print(f"Error: {e}")


def convert_to_mp4(image_folder, fps=50):
    output_path = os.path.join(image_folder, "output.mp4")
    try:
        image_files = sorted(
            [f for f in os.listdir(image_folder) if f.endswith(".png")]
        )

        if not image_files:
            raise ValueError("No PNG files found in the specified folder.")

        first_image_path = os.path.join(image_folder, image_files[0])
        first_image = cv2.imread(first_image_path)
        height, width, _ = first_image.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            frame = cv2.imread(image_path)
            video_writer.write(frame)

        video_writer.release()
        print(f"MP4 video saved to {output_path} with {fps} FPS.")

    except Exception as e:
        print(f"Error: {e}")


def to_cuda(x):
    r"""Move all tensors to cuda."""
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cuda()
    return x


def to_cpu(x):
    r"""Move all tensors to cpu."""
    if isinstance(x, list):
        x = [to_cpu(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cpu(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cpu(value) for key, value in x.items()}
    elif isinstance(x, torch.Tensor):
        x = x.cpu()
    return x


def is_jupyter():
    import __main__ as main

    return not hasattr(main, "__file__")


tmp = []


def hold_gpus(gpu_idxs):
    global tmp
    for gpu in gpu_idxs:
        gpu = int(gpu)
        tmp.append(torch.randn(1000000000).cuda(gpu))


def release_gpus():
    global tmp
    tmp = []
    torch.cuda.empty_cache()


def cache_model_output(func):
    def wrapper(*args, **kwargs):
        hash_sum = 0
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = np.array(arg.cpu())
            if isinstance(arg, np.ndarray):
                hash_int = int(hashlib.md5(arg.tobytes()).hexdigest(), 16)
            else:
                hash_int = hash(arg)
            hash_sum += hash_int

        if not os.path.exists(".cache"):
            os.mkdir(".cache")
        cache_path = f".cache/{hash_sum}.pt"
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        else:
            result = func(*args, **kwargs)
            torch.save(result, cache_path)
            return result
    return wrapper
