import argparse
import torch
import time
if __name__ == "__main__":
    # Instantiate the ArgumentParser object
    parser = argparse.ArgumentParser()

    # Define the arguments you want to get
    parser.add_argument('--gpus', type=list, help='Your name')
    
    args = parser.parse_args()
    tmp = []
    for gpu in args.gpus:
        if gpu not in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            continue
        gpu = int(gpu)
        tmp.append(torch.randn(1e10).cuda(gpu))
    time.sleep(10000)