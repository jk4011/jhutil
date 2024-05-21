import torch
import argparse
import torch
import time


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
    return not hasattr(main, '__file__')


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
