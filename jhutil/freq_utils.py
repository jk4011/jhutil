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
from functools import lru_cache
from contextlib import contextmanager
import json
import pickle
import random
import inspect
import shutil
import torch.nn.functional as F


def reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path):
    return json.load(open(path, "r", encoding="utf-8"))


def load_img(path, max_resolution=None):
    img = ToTensor()(Image.open(path))
    if max_resolution is not None and (img.shape[1] > max_resolution or img.shape[2] > max_resolution):
        resize_scale = min(max_resolution / img.shape[1], max_resolution / img.shape[2])
        H, W = int(img.shape[1] * resize_scale), int(img.shape[2] * resize_scale)
        img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False)[0]
    return img


def bkgd2white(rgba):
    new_rgb = (rgba[:3] + (1 - rgba[3])[None, :])
    new_rgba = torch.cat([new_rgb, rgba[3:4]], dim=0)
    return new_rgba


def save_img(tensor, path):
    tensor = tensor.squeeze().detach().cpu()
    if tensor.ndim == 3 and tensor.shape[0] in [3, 4]:
        tensor = tensor.permute(1, 2, 0)

    tensor = (tensor.numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(tensor).save(path)


def save_video(image_list: torch.Tensor, filename: str, fps: int = 30, normalize: bool = True):
    
    if isinstance(image_list, list):
        image_list = torch.stack(image_list)
    image_list = image_list.detach().cpu()

    if image_list.ndim != 4:
        raise ValueError("Input tensor must have 4 dimensions. (B, C, H, W) or (B, H, W, C)")

    b, c, h, w = image_list.shape if image_list.shape[1] <= 4 else (image_list.shape[0], image_list.shape[3], image_list.shape[1], image_list.shape[2])

    if image_list.shape[1] == c and image_list.shape[2] == h and image_list.shape[3] == w:
        image_list = image_list.permute(0, 2, 3, 1)

    if normalize:
        image_list = (image_list * 255.0).clamp(0, 255).byte()
    else:
        image_list = image_list.clamp(0, 255).byte()

    frames_np = image_list.numpy()

    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    
    for i in range(frames_np.shape[0]):
        frame = frames_np[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)

    out.release()
    print(f"Video has been saved to '{filename}'.")




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
        x = x.cuda().detach()
    return x

def to_tensor(x):
    if isinstance(x, list):
        x = [to_cuda(item) for item in x]
    elif isinstance(x, tuple):
        x = (to_cuda(item) for item in x)
    elif isinstance(x, dict):
        x = {key: to_cuda(value) for key, value in x.items()}
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
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



def check_disk_space(path, required_gb=5):
    """Check if there is enough disk space available.

    Args:
        path: Path to check disk space for
        required_gb: Required space in GB (default: 5GB)

    Returns:
        bool: True if enough space is available, False otherwise
    """
    try:
        stat = shutil.disk_usage(os.path.dirname(path) if os.path.isfile(path) or not os.path.exists(path) else path)
        required_bytes = required_gb * 1024 * 1024 * 1024
        return stat.free > required_bytes
    except Exception as e:
        print(f"Warning: Could not check disk space: {e}")
        return True  # Proceed if we can't check


def load_cache_file(cache_path):
    if cache_path.endswith('.pt'):
        return torch.load(cache_path, weights_only=False)

    elif cache_path.endswith('.pkl'):
        with open(cache_path, 'rb') as f:
            obj = pickle.load(f)
        return obj

    else:
        raise ValueError("Unsupported file format. Use .pt or .pkl files.")



def save_cache_file(results, cache_path):
    if not check_disk_space(cache_path, required_gb=5):
        print(f"Warning: Not enough disk space (< 5GB available). Skipping cache save to {cache_path}")
        return False

    if cache_path.endswith('.pt'):
        torch.save(results, cache_path)
    elif cache_path.endswith('.pkl'):
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
    else:
        raise ValueError("Unsupported file format. Use .pt or .pkl files.")

    return True


def tensor_to_hash(tensor, light=False):
    if isinstance(tensor, torch.Tensor):
        tensor = np.array(tensor.detach().cpu())
    return int(hashlib.md5(tensor.tobytes()).hexdigest(), 16)


# check environment variable JHUTIL_CACHE_OFF
is_cache_off = os.environ.get("JHUTIL_CACHE_OFF") == "1"
if is_cache_off:
    from jhutil import color_log; color_log(1111, "cache is off")

is_save_off = False

@contextmanager
def cache_off():
    """Context manager to temporarily disable cache.
    
    Usage:
        with cache_off():
            # cache is disabled here
            result = my_cached_function()
        # cache is restored to original state
    """
    global is_cache_off
    original_value = is_cache_off
    is_cache_off = True
    try:
        yield
    finally:
        is_cache_off = original_value

@contextmanager
def save_off():
    global is_save_off
    original_value = is_save_off
    is_save_off = True
    try:
        yield
    finally:
        is_save_off = original_value


# wrapper function
def cache_output(func_name="", override=False, verbose=True, folder_path=".cache", use_pickle=False):
    def decorator(func):
        fullspec = inspect.getfullargspec(func)
        is_method = len(fullspec.args) > 0 and fullspec.args[0] == 'self'

        def wrapper(*args, **kwargs):
            # ============================================
            # ① self를 class name으로 치환
            # ============================================
            if is_method:
                cls_name = args[0].__class__.__name__
                # 첫 인자를 class name 문자열로 바꿔줌
                new_args = (cls_name,) + args[1:]
            else:
                new_args = args

            all_args = list(new_args) + list(kwargs.values())

            # ============================================
            # ② hashing 계산
            # ============================================
            hash_sum = 0
            for i, arg in enumerate(all_args):
                if isinstance(arg, torch.Tensor):
                    arg = np.array(arg.detach().cpu())
                if isinstance(arg, np.ndarray):
                    hash_int = int(hashlib.md5(arg.tobytes()).hexdigest(), 16)
                else:
                    hash_int = int(hashlib.md5(str(arg).encode()).hexdigest(), 16)
                hash_sum += hash_int * (i + 1)

            if func_name != "":
                hash_sum += int(hashlib.md5(str(func_name).encode()).hexdigest(), 16)
                subfolder_path = os.path.join(folder_path, func_name)
            else:
                subfolder_path = folder_path

            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

            cache_path = f"{subfolder_path}/{hash_sum}.pkl" if use_pickle else f"{subfolder_path}/{hash_sum}.pt"

            # ============================================
            # cache hit
            # ============================================
            if not override and os.path.exists(cache_path) and not is_cache_off:
                if verbose:
                    from jhutil import color_log; color_log("cccc", f"cache file found, skipping {func_name}")
                try:
                    return load_cache_file(cache_path)
                except:
                    if verbose:
                        from jhutil import color_log; color_log("aaaa", f"cache file {cache_path} corrupted, executing {func_name}")
                    result = func(*args, **kwargs)
                    if not is_save_off:
                        save_cache_file(result, cache_path)
                    return result

            # ============================================
            # cache miss
            # ============================================
            if verbose:
                from jhutil import color_log; color_log("aaaa", f"executing {func_name} for caching")

            result = func(*args, **kwargs)

            if not is_save_off:
                save_cache_file(result, cache_path)
            return result

        return wrapper
    return decorator



def get_img_diff(img1, img2):

    diff_img = torch.zeros_like(img1).cuda()
    
    diff_max = (img1 - img2).max(dim=0).values
    diff_min = (img1 - img2).min(dim=0).values
    
    diff_img[0][diff_max > 0] = diff_max[diff_max > 0]
    diff_img[2][diff_min < 0] = - diff_min[diff_min < 0]
    if img1.shape[0] == 4:
        diff_img[3] = 1
    
    diff_img_concat = torch.concat([img1, diff_img, img2], dim=2)
    return diff_img_concat
