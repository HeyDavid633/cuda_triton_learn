# 2024.12.09 周一
# 第一个关于Matmul的triton 例子，期望理解其中的划分

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
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'], # 这个值的变化会带来 调优配置变化
)

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,  # 矩阵指针 a(M, K) * b(K, N) = c(M, N)
    M, N, K,              # 矩阵的维度信息
    # 跨步信息，在移动一个元素时，相对于ptr应该移动多少
    stride_am, stride_ak, # 对于矩阵a来说，stride_am 表示为了访问下一行，需要在a_ptr上相对增加多少
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # 元参数 Meta-perameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,  
    ACTIVATION: tl.constexpr
):    
    pid = tl.program_id(axis=0)    # 块id,而不是线程id,  其值为[0，9）最大值为8；因为启动的时候grid 3*3启动的
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 每个红框里，有多少程序pid = 9*3=27个，此时认为 GROUP_SIZE_M=9
    group_id = pid // num_pid_in_group           # 本程序所在的group的id， 第几个红框 (27为单位的)
    first_pid_m = group_id * GROUP_SIZE_M        # 小组里（9为单位），第一个程序的行id，而没有去 group_id * num_pid_in_group 
    
    # if first_pid_m != 0:
    #     print(" group_id != 0, group_id = ", group_id )
    #     print(" first_pid_m != 0, first_pid_m = ", first_pid_m )
    
    #  很怀疑最终减掉完了group_size_m是负数 --> group_id一直是0，则不存在这个问题
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 保护最后一个group，预防 num_pid_m不能被整除 
    # print("group_size_m:", group_size_m)                        # print函数用的是triton里面的，只能 单个字符串 + 值
    # print("num_pid_m:", num_pid_m)
    # print("first_pid_m:", first_pid_m)
    
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m) # 红框中的程序的行 id
    pid_n = (pid % num_pid_in_group) // group_size_m                # 红框中的程序的列 id
    
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k  = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    
    # 主要计算，沿着K的维度以（BLOCK_SIZE_M, BLOCK_SIZE_N）的计算量计算 -----------------
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32) # 一次性的计算结果为一个块
    for k in range (0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)
    
    
    # 计算结果写回到c -------------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs  = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask  = (offs_cm[:, None] < M) &  (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)
    
def triton_matmul(a, b, activation=""):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=DATA_TYPE)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
    )
    return c


if __name__ == '__main__':
    torch_cuda_identify()
    DATA_TYPE = torch.float16

    para_m = 1024
    para_n = 1024
    para_k = 1024
    torch.manual_seed(0)
    a = torch.randn((para_m, para_k), device='cuda', dtype=DATA_TYPE)
    b = torch.randn((para_k, para_n), device='cuda', dtype=DATA_TYPE)
    
    triton_output = triton_matmul(a, b)
    torch_output = torch.matmul(a, b)
    # print(f"triton_output_with_fp16_inputs={triton_output}")
    # print(f"torch_output_with_fp16_inputs={torch_output}")

    print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(torch_output - triton_output))))