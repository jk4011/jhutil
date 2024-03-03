import importlib

import torch
import numpy as np

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont

import os
import os.path as osp


def is_jupyter():
    import __main__ as main
    return not hasattr(main, '__file__')


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
