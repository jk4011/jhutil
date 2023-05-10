import torch
import time
import sys

gpus = sys.argv[1]
try:
    duration = sys.argv[2]
except:
    duration = 60
gpus = gpus.split(",")

# time check 
start = time.time()

while True:
    for gpu in gpus:
        a = torch.randn((10000, 10000), device=f'cuda:{gpu}')
        b = torch.randn((100, 100), device=f'cuda:{gpu}')
        for i in range(10):
            b = b @ b
        del a
    # if start time is greater than 60 miniutes, break
    if time.time() - start > 60 * duration:
        break
    time.sleep(0.5)
    torch.cuda.empty_cache()
