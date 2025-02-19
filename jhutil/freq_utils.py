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


# wrapper function
def cache_output(func_name=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            hash_sum = 0
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    arg = np.array(arg.cpu())
                if isinstance(arg, np.ndarray):
                    hash_int = int(hashlib.md5(arg.tobytes()).hexdigest(), 16)
                else:
                    hash_int = int(hashlib.md5(str(arg).encode()).hexdigest(), 16)
                hash_sum += hash_int
                
            if func_name != "":
                hash_sum += int(hashlib.md5(str(func_name).encode()).hexdigest(), 16)

            if not os.path.exists(".cache"):
                os.mkdir(".cache")
            cache_path = f".cache/{hash_sum}.pt"
            if os.path.exists(cache_path):
                from jhutil import color_log; color_log("cccc", f"load cached output, skipping {func_name}")
                return torch.load(cache_path)
            else:
                from jhutil import color_log; color_log("aaaa", f"no cached output, executing {func_name}")
                result = func(*args, **kwargs)
                torch.save(result, cache_path)
                return result
        return wrapper
    return decorator



def get_img_diff(img1, img2):
    diff_img = torch.zeros_like(img1).cuda()
    
    diff_max = (img1 - img2).max(dim=0).values
    diff_min = (img1 - img2).min(dim=0).values
    
    diff_img[0][diff_max > 0] = diff_max[diff_max > 0]
    diff_img[2][diff_min < 0] = - diff_min[diff_min < 0]
    
    diff_img_concat = torch.concat([img1, diff_img, img2], dim=2)
    return diff_img_concat
