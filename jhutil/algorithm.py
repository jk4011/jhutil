import torch
import jhutil
from jhutil import cache_output


@cache_output(func_name="knn")
def knn(src, dst, k=1, is_naive=False, is_sklearn=False, device="cuda", chunk_size=1e5):
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
        knn = KNN(k=k, transpose_mode=True)
        src_chunks = torch.chunk(src, int(src.shape[0] * dst.shape[0] // chunk_size) + 1, 0)
        distance = []
        indices = []    
        for src_chunk in src_chunks:
            dist_chunk, indices_chunk = knn(dst[None, :], src_chunk[None, :])
            distance.append(dist_chunk)
            indices.append(indices_chunk)
        distance = torch.cat(distance, dim=1)
        indices = torch.cat(indices, dim=1)

    distance = distance.squeeze(0).to(device)
    indices = indices.squeeze(0).to(device)

    return distance, indices


if __name__ == "__main__":
    src = torch.rand(100000, 3)
    dst = torch.rand(100000, 3)
    distance, indices = knn(src, dst)

    import jhutil; jhutil.color_log(1111, distance)
    import jhutil; jhutil.color_log(2222, indices)
