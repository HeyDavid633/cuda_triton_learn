# 2024.12.10  周二
# 
# Matmul连乘: MatMul + MatMul
# 在Github上找的现成 GEMM连乘算子，解决其中需要同步的问题; 看起来解决方法是  tl.atomic_add
# https://github.com/cmullovisl/triton-fused-mlp/blob/master/triton-b2b-matmul.py
# 奇怪的是，找的这个连乘算子计算结果非常相近但还是有差异
# + fp32时 平均误差在 1.25，总体差异在 0.1 -- 1% 左右
# + fp16时 平均误差在 0.25，差异量级缩小到了 0.1%  计算结果可以认为正确
#        如果在全1阵时，则计算能正确，说明没问题；主要来源于精度误差
# 
# A(M, K) @ B(K, N) = C(M, N);  C(M, N) @ D(N, H) = E(M, H)   
# 其中 H = K 否则这个任务划分也不对


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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32,  'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64,  'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64,  'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32,  'BLOCK_SIZE_K': 64,  'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4)
    ],
    key=['M', 'N', 'K'],
    reset_to_zero=['e_ptr'],
)


@triton.jit
def matmul_matmul_kernel(
    a_ptr, b_ptr, d_ptr, e_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_dn, stride_dk,
    stride_em, stride_ek,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    accumulator = accumulator.to(tl.float16)

    offs_dn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_em = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    d_ptrs = d_ptr + (offs_dn[:, None] * stride_dn + offs_k[None, :] * stride_dk)
    e_ptrs = e_ptr + (offs_em[:, None] * stride_em + offs_k[None, :] * stride_ek)
    
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        d = tl.load(d_ptrs)
        # M x N @ N x K
        accumulator2 = tl.dot(accumulator, d)
        # 不太理解这里的原子加法，应该同步了一下，之后误差较大，以及在这里其实没有写回
        tl.atomic_add(e_ptrs, accumulator2) 
        d_ptrs += BLOCK_SIZE_K * stride_dk
        e_ptrs += BLOCK_SIZE_K * stride_ek
    


def triton_matmul_matmul(A, B, D):
    M, K = A.shape
    K, N = B.shape
    # N, K = D.shape
    # result as E
    E = torch.zeros((M, K), device=A.device, dtype=A.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    matmul_matmul_kernel[grid](
        A, B, D, E,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        D.stride(0), D.stride(1),
        E.stride(0), E.stride(1),
        ACTIVATION=None,
    )

    return E


def torch_matmul_matmul(A, B, D):
    result = torch.matmul(torch.matmul(A, B), D)
    return result

@torch.compile
def compiled_torch_matmul_matmul(A, B, D):
    result = torch.matmul(torch.matmul(A, B), D)
    return result


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  #  用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  #  `x_name` `x_name` 的不同可能值
        line_arg='provider',  #  参数名，其值对应于图表中不同线条
        line_vals=['triton', 'torch', 'torch.compile'],  #  `line_arg`` `line_arg` 的可能值
        line_names=[
            "Triton",
            "Torch",
            "Torch.compile",
        ],  # label name for the lines 线条的标签名称
        styles=[('blue', '-'), ('green', '-'), ('orange', ':')],  # line styles 线条的样式
        ylabel="TFLOPS",  # label name for the y-axis y 轴的标签名称
        plot_name="matmul-matmul",  # 图表的名称，也用作保存图表的文件名
        args={'M': 1024, 'K': 1024},  # `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, K, provider):
    A = torch.randn((M, K), device='cuda', dtype=DATA_TYPE)
    B = torch.randn((K, N), device='cuda', dtype=DATA_TYPE)
    D = torch.randn((N, K), device='cuda', dtype=DATA_TYPE)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_matmul_matmul(A, B, D), quantiles=quantiles)
    if provider == 'torch.compile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled_torch_matmul_matmul(A, B, D), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul_matmul(A, B, D), quantiles=quantiles)
    
    perf = lambda ms: 4 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    torch_cuda_identify()
    torch.manual_seed(0)
    seq_len = 1024
    DATA_TYPE = torch.float16
    
    A = torch.randn((seq_len, seq_len*2), device='cuda', dtype=DATA_TYPE)
    B = torch.randn((seq_len*2, seq_len), device='cuda', dtype=DATA_TYPE)
    D = torch.randn((seq_len, seq_len*2), device='cuda', dtype=DATA_TYPE)
    
    # A = torch.ones((seq_len, seq_len), device='cuda', dtype=DATA_TYPE)
    # B = torch.ones((seq_len, seq_len), device='cuda', dtype=DATA_TYPE)
    # D = torch.ones((seq_len, seq_len), device='cuda', dtype=DATA_TYPE)
    
    # torch_output  = torch_matmul_matmul(A, B, D)
    # triton_output = triton_matmul_matmul(A, B, D)
   
    # print(f"torch_output={torch_output}")
    # print(f"triton_output={triton_output}") 
    # print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(torch_output - triton_output))))
    
    benchmark.run(show_plots=True, print_data=True, save_path='./04-matmul-matmul')

