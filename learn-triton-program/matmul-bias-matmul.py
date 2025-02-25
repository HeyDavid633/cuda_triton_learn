# 2024.12.09 周一 
# 没有成功 --- 中间需要等待一下的问题
# matmul（bias + act）+ matmul 写在一个kernel
# a(M, K) * b(K, N) = C(M, N);  C(M, N)*D(N, H) = E(M, H)

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


def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'], # 这个值的变化会带来 调优配置变化
)

@triton.jit
def matmul_bias_matmul_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr, e_ptr, bias1_ptr,  # a(M, K) * b(K, N) = C(M, N);  C(M, N)*D(N, H) = E(M, H)
    M, N, K, H,             
    stride_am, stride_ak, # 对于矩阵a来说，stride_am 表示为了访问下一行，需要在a_ptr上相对增加多少
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_dn, stride_dh,
    stride_em, stride_eh,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,  
    ACTIVATION: tl.constexpr, 
):    
    pid = tl.program_id(axis=0)    # 块id,而不是线程id,  其值为[0，9）最大值为8；因为启动的时候grid 3*3启动的
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 每个红框里，有多少程序pid = 9*3=27个，此时认为 GROUP_SIZE_M=9
    pid_m = (pid % num_pid_in_group) % GROUP_SIZE_M     # 红框中的程序的行 id
    pid_n = (pid % num_pid_in_group) // GROUP_SIZE_M    # 红框中的程序的列 id
    
    
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k  = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    
    # 主要计算，沿着K的维度以（BLOCK_SIZE_M, BLOCK_SIZE_N）的计算量计算 -----------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # 一次性的计算结果为一个块
    for k in range (0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    bias_ptrs = bias1_ptr + (offs_am[:, None] * stride_am + offs_bn[None, :] * stride_bn)
    bias = tl.load(bias_ptrs)
    accumulator += bias
    mid_c = accumulator # 直接给 c 赋值accumulator 作为中间结果以float32存
    
    # if ACTIVATION == "leaky_relu":
    #     accumulator = leaky_relu(accumulator)
    # c = accumulator.to(tl.float16)
    
    # 中间结果c 少了tl.store(c_ptrs, c)写回而已
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_dh = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_h = tl.arange(0, BLOCK_SIZE_K)
    c_ptrs  = c_ptr + offs_cm[:, None] * stride_cm + offs_h[None, :] * stride_cn
    d_ptrs  = d_ptr + offs_h[:, None] * stride_dn + offs_dh[None, :] * stride_dh
    tl.store(c_ptrs, mid_c)
    # 这里必须要同步一下 ？ 
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range (0, tl.cdiv(K, BLOCK_SIZE_K)):
        c = tl.load(c_ptrs)
        d = tl.load(d_ptrs)
        accumulator = tl.dot(c, d, accumulator)
        c_ptrs += BLOCK_SIZE_K * stride_cn
        d_ptrs += BLOCK_SIZE_K * stride_dn
        
    e = accumulator.to(tl.float16)
    
    offs_em = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_eh = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    e_ptrs  = e_ptr + offs_em[:, None] * stride_em + offs_eh[None, :] * stride_eh
    tl.store(e_ptrs, e)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)
    
def triton_matmul_bias_matmul(a, b, d, bias1, activation=""):
    M, N, K, H = a.shape[0], b.shape[1], a.shape[1], d.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=DATA_TYPE)
    e = torch.empty((M, H), device=a.device, dtype=DATA_TYPE)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_bias_matmul_kernel[grid](
        a, b, c, d, e, bias1,
        M, N, K, H,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        d.stride(0), d.stride(1),
        e.stride(0), e.stride(1),
        ACTIVATION=activation,
    )
    return c

def torch_matmul_bias_matmul(a, b, d, bias):
    # activation_leakyrelu = torch.nn.LeakyReLU(0.01)
    # result = activation_leakyrelu(torch.matmul(a, b) + bias) 
    c = torch.matmul(a, b) + bias
    result = c * d
    return result


if __name__ == '__main__':
    torch_cuda_identify()
    DATA_TYPE = torch.float16

    para_m = 1024
    para_n = 1024
    para_k = 1024
    para_h = 1024
    torch.manual_seed(0)
    a = torch.randn((para_m, para_k), device='cuda', dtype=DATA_TYPE)
    b = torch.randn((para_k, para_n), device='cuda', dtype=DATA_TYPE)
    d = torch.randn((para_m, para_h), device='cuda', dtype=DATA_TYPE)
    bias1 =  torch.randn((para_m, para_n), device='cuda', dtype=DATA_TYPE)
    
    triton_output = triton_matmul_bias_matmul(a, b, d, bias1) 
    torch_output = torch_matmul_bias_matmul(a, b, d, bias1)
    # print(f"triton_output_with_fp16_inputs={triton_output}")
    # print(f"torch_output_with_fp16_inputs={torch_output}")

    print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(torch_output - triton_output))))