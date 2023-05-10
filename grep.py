import torch
import time
import sys

gpus = sys.argv[1]
gpus = gpus.split(",")

for i in range(2000):
    for gpu in gpus: 
        a = torch.randn(200000, device=f'cuda:{gpu}')
        a = a @ a
    time.sleep(5)