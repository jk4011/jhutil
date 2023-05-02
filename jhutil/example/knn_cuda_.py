

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
    ref   is Tensor [bs x nr x dim]
    query is Tensor [bs x nq x dim]
    
    return 
        dist is Tensor [bs x nq x k]
        indx is Tensor [bs x nq x k]
else
    ref   is Tensor [bs x dim x nr]
    query is Tensor [bs x dim x nq]
    
    return 
        dist is Tensor [bs x k x nq]
        indx is Tensor [bs x k x nq]
"""

knn = KNN(k=10, transpose_mode=True)

ref = torch.rand(32, 1000, 5).cuda()
query = torch.rand(32, 50, 5).cuda()

dist, indx = knn(ref, query)  # 32 x 50 x 10
import jhutil;jhutil.jhprint(1111, dist)