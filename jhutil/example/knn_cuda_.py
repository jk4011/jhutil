

# ------------------------------------------------------------------------------------------------------------------------------ #
# install command: 
# pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# wget -P /usr/bin https://github.com/unlimblue/KNN_CUDA/raw/master/ninja
# ------------------------------------------------------------------------------------------------------------------------------ #
# site : https://github.com/unlimblue/KNN_CUDA
# ------------------------------------------------------------------------------------------------------------------------------ #

from knn_cuda import KNN
import torch

# Make sure your CUDA is available.
assert torch.cuda.is_available()

"""
if transpose_mode is True, 
    src   is Tensor [bs x nr x dim]
    dst is Tensor [bs x nq x dim]
    
    return 
        dist is Tensor [bs x nq x k]
        indx is Tensor [bs x nq x k]
else
    src   is Tensor [bs x dim x nr]
    dst is Tensor [bs x dim x nq]
    
    return 
        dist is Tensor [bs x k x nq]
        indx is Tensor [bs x k x nq]
"""

# knn

knn = KNN(k=1, transpose_mode=True)

src = torch.rand(200, 3).cuda()
dst = torch.rand(100, 3).cuda()

dist, indx = knn(src[None, :], dst[None, :])  # 32 x 50 x 10
import jhutil;jhutil.jhprint(1111, dist)


# naive 

src = src.reshape(-1, 1, src.shape[-1])
distance = torch.norm(src - dst, dim=-1)

knn = distance.topk(1, largest=False)
distance = knn.values.ravel().cpu()
indices = knn.indices.ravel().cpu()
import jhutil;jhutil.jhprint(2222, dist)
