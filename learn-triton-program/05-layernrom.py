# 05-layernrom.py
# 2024.12.11 周三
# 
# 参照 https://triton.hyper.ai/docs/getting-started/tutorials/layer-normalization 
# 实现的layernrom，期待能实现 matmul + bias + layernrom
#
# 另发现flash-attn中有所实现但很复杂 https://blog.csdn.net/just_sort/article/details/136087403
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py


import torch
import triton
import triton.language as tl

def torch_cuda_identify(print_info = True):
    if torch.cuda.is_available():
        if print_info:
            print(' PyTorch version:', torch.__version__)
            print(' CUDA version \t:', torch.version.cuda)
            print(' GPU cuda:({}) \t: {}'.format(torch.cuda.current_device(), torch.cuda.get_device_name()),'\n', "-" * 50)
        return torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        print('cuda is not avaliable !')
        return torch.device('cpu')
    

@triton.jit
def layer_norm_fwd_fused_kernel(
    x_ptr, y_ptr, w_ptr, bias_ptr,  
    mean_ptr,  rstd_ptr,  
    stride_xm,  
    N,    # X 的列数
    eps,  # 用于避免除以 0 的 epsilon
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(axis=0) # 每个块对应一行
    x_ptr += row * stride_xm
    y_ptr += row * stride_xm
    
    # 计算均值
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(x_ptr + cols, mask=cols<N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    
    # 计算标准差； 方差是var
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + cols, mask=cols<N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x-mean, 0.)
        _var += x*x
    var  = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps) 
    
    # 写入 mean / rstd
    tl.store(mean_ptr + row, mean)
    tl.store(rstd_ptr + row, rstd)
    
    # 归一化并应用线性变换
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w    = tl.load(w_ptr + cols, mask=mask)
        bias = tl.load(bias_ptr + cols, mask=mask)
        x    = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + bias  
        
        y = y.to(tl.float16)
        tl.store(y_ptr + cols, y, mask=mask)  


def triton_layernrom_fwd(x, w_shape, weight, bias, eps):
    y = torch.empty_like(x)
    
    # 输入张量尺寸改成2D
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    
    # 每个特征少于 64KB , 入队融合内核 ?
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_SIZE:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # 对 warp 数量的启发算法
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    
    # 入队融合内核
    layer_norm_fwd_fused_kernel[(M, )](  #
            x_arg, y, weight, bias, mean, rstd,  #
            x_arg.stride(0), N, eps,  #
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, num_ctas=1)
    return y
    


if __name__ == "__main__":
    torch_cuda_identify()
    torch.manual_seed(0)
    seq_len = 1024
    DATA_TYPE = torch.float16
    M = seq_len
    N = seq_len * 2
    eps = 1e-5
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=DATA_TYPE, device="cuda")
    bias   = torch.rand(w_shape, dtype=DATA_TYPE, device="cuda")
    x = -2.3 + 0.5 * torch.randn(x_shape,  dtype=DATA_TYPE, device="cuda")
    
    torch_output  = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(DATA_TYPE)
    triton_output = triton_layernrom_fwd(x, w_shape, weight, bias, eps)
    
    print(f"torch_output={torch_output}")
    print(f"triton_output={triton_output}") 
    
    print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(torch_output - triton_output))))
    
    