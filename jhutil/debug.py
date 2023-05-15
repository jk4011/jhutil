from pytorch_memlab import MemReporter
from pytorch_memlab.utils import readable_size
import torch


def memory():
    return readable_size(torch.cuda.memory_allocated())


def memory_reporter():
    reporter = MemReporter()
    reporter.report()
