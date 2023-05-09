import torch
import jhutil

def knn(src, dst, k=1, is_naive=False):
    from knn_cuda import KNN
    """return k nearest neighbors"""

    if not isinstance(src, torch.Tensor):
        src = torch.tensor(src)
    if not isinstance(dst, torch.Tensor):
        dst = torch.tensor(dst)
    
    assert(len(src.shape) == 2)
    assert(len(dst.shape) == 2)
    assert(src.shape[-1] == dst.shape[-1])
    
    # cpu or gpu, memory inefficient
    if is_naive: 
        src = src.reshape(-1, 1, src.shape[-1])
        distance = torch.norm(src - dst, dim=-1)

        knn = distance.topk(k, largest=False)
        distance = knn.values
        indices = knn.indices
    
    if len(src) * len(dst) > 10e8:
        _batched_knn(src, dst, k)
        
    # gpu 
    else:
        num_gpus = torch.cuda.device_count()
        src = src.cuda(num_gpus - 1).contiguous()
        dst = dst.cuda(num_gpus - 1).contiguous()

        from knn_cuda import KNN
        knn = KNN(k=1, transpose_mode=True)
        distance, indices = knn(dst[None, :], src[None, :]) 
    
    tmp = distance.ravel()
    distance = tmp.cpu()
    indices = indices.ravel().cpu()

    return distance, indices


def _batched_knn(xs, ys, k=1, memory_gb=1):
    
    knn = KNN(k=k, transpose_mode=True)
    n, m = len(xs), len(ys)
    b = memory_gb * 5000000 // n
    if b == 0:
        b = 1
        
    distance_lst, indices_lst = [], []
    for i in range(0, m, b):
        b_ = min(m - i, b)
        y = ys[i : i + b_].cuda(non_blocking=True) # b ki
        distance, indices = knn(xs[None, :], y[None, :]) # b k
        distance_lst.append(distance)
        indices_lst.append(indices)
        
    distance = torch.cat(distance_lst, dim=1) # m k
    indices = torch.cat(indices_lst, dim=1) # m k
    
    
    return distance, indices
