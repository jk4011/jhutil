import torch
import jhutil


def knn(src, dst, k=1, is_naive=False, is_sklearn=False):
    """return k nearest neighbors"""

    assert (len(src.shape) == 2)
    assert (len(dst.shape) == 2)
    assert (src.shape[-1] == dst.shape[-1])

    if is_sklearn:
        from sklearn.neighbors import NearestNeighbors

        neigh = NearestNeighbors(n_neighbors=k)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel()

    if not isinstance(src, torch.Tensor):
        src = torch.tensor(src)
    if not isinstance(dst, torch.Tensor):
        dst = torch.tensor(dst)

    # cpu or gpu, memory inefficient
    if is_naive:
        src = src.reshape(-1, 1, src.shape[-1])
        distance = torch.norm(src - dst, dim=-1)

        knn = distance.topk(k, largest=False)
        distance = knn.values
        indices = knn.indices

    # gpu
    else:
        src = src.cuda().contiguous()
        dst = dst.cuda().contiguous()

        from knn_cuda import KNN
        knn = KNN(k=1, transpose_mode=True)
        distance, indices = knn(dst[None, :], src[None, :])

    tmp = distance.ravel()
    distance = tmp.cpu()
    indices = indices.ravel().cpu()

    return distance, indices
