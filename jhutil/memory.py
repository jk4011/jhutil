from pytorch_memlab.utils import readable_size
import torch


def get_memory_allocated():
    return readable_size(torch.cuda.memory_allocated())
