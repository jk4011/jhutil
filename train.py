1111 9 1
1111 9 2
import torch
import time
import sys

gpus = sys.argv[1]
gpus = gpus.split(",")

for i in range(2000):
    for gpu in gpus:
        a = torch.randn((10000, 10000), device=f'cuda:{gpu}')
        b = torch.randn((300, 300), device=f'cuda:{gpu}')
        for i in range(10):
            b = b @ b

        print(1111, i, gpu)
        del a
    torch.cuda.empty_cache()