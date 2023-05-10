import torch
import time
import sys

gpus = sys.argv[1]
gpus = gpus.split(",")

for i in range(2000):
    for gpu in gpus: 
        a = torch.randn((1000, 1000), device=f'cuda:{gpu}')
        for i in range(10):
            a = a @ a
        del a
    torch.cuda.empty_cache()
    time.sleep(3)