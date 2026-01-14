import torch
from jhutil import cache_output


@cache_output(func_name="knn", verbose=False, override=True, folder_path=".cache")
def knn(src, dst, k=1, is_naive=False, backend="sklearn", device="cuda", chunk_size=1e5, exclude_first=False):
    """return k nearest neighbors"""

    assert (len(src.shape) == 2)
    assert (len(dst.shape) == 2)
    assert (src.shape[-1] == dst.shape[-1])

    if not isinstance(src, torch.Tensor):
        src = torch.tensor(src)
    if not isinstance(dst, torch.Tensor):
        dst = torch.tensor(dst)

    # faiss is better suited when d > 20
    # faiss FAQ에서 저차원이면 scikit-learn 쓰는게 좋다고 함.
    # https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-index-2d-or-3d-data
    if backend == "faiss":
        import faiss
        import faiss.contrib.torch_utils  # PyTorch 텐서 지원 활성화
        d = src.shape[-1]
        res = faiss.StandardGpuResources()

        config = faiss.GpuIndexFlatConfig()
        config.useFloat16 = True  # float16 연산 활성화 - 속도 차이가 없음.
        
        index = faiss.GpuIndexFlatL2(res, d, config)
        index.add(dst)

        distances, indices = index.search(src, k)

    # sklearn is better suited when d < 10
    elif backend == "sklearn":
        from sklearn.neighbors import NearestNeighbors
        
        src = src.cpu().detach().numpy()
        dst = dst.cpu().detach().numpy()
        neigh = NearestNeighbors(n_neighbors=k, algorithm="kd_tree", n_jobs=-1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        distances = torch.tensor(distances, device=device, dtype=torch.float32)
        indices = torch.tensor(indices, device=device, dtype=torch.int32)


    # cpu or gpu, memory inefficient
    elif is_naive:
        src = src.reshape(-1, 1, src.shape[-1])
        distances = torch.norm(src - dst, dim=-1)

        knn = distances.topk(k, largest=False)
        distances = knn.values
        indices = knn.indices

    # gpu
    else:
        src = src.cuda().contiguous()
        dst = dst.cuda().contiguous()

        from knn_cuda import KNN
        knn = KNN(k=k, transpose_mode=True)
        src_chunks = torch.chunk(src, int(src.shape[0] // chunk_size) + 1, 0)
        distances = []
        indices = []    
        for src_chunk in src_chunks:
            dist_chunk, indices_chunk = knn(dst[None, :], src_chunk[None, :])
            distances.append(dist_chunk)
            indices.append(indices_chunk)
        distances = torch.cat(distances, dim=1)
        indices = torch.cat(indices, dim=1)

    distances = distances.to(device)
    indices = indices.to(device)

    if exclude_first:
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    
    return distances, indices


def ball_query(src, dst, r, k=10, concat=False):
    k = min(k, dst.size(0))
    distance, knn_indices = knn(src, dst, k)
    mask = distance < r
    
    if concat:
        ball_indices = knn_indices[mask].unique()
    else:
        ball_indices = []
        for i in range(src.size(0)):
            ball_indices.append(knn_indices[i][mask[i]])
    
    return ball_indices


def dbscan(src, is_sklearn=True, eps=0.05, min_samples=3):
    
    if is_sklearn:
        device = src.device
        src = src.cpu().numpy()
        # 예시 데이터 생성 (반달 모양)
        from sklearn.cluster import DBSCAN

        # DBSCAN 적용
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(src)
        labels = torch.tensor(labels, device=device, dtype=torch.int32)
    
    return labels




if __name__ == "__main__":
    src = torch.rand(100000, 3)
    dst = torch.rand(100000, 3)
    distance, indices = knn(src, dst)

    import jhutil; jhutil.color_log(1111, distance)
    import jhutil; jhutil.color_log(2222, indices)
