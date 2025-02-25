# 2024.12.09 周一
# 保证块大小能被整除的前提下，在这里上面加bias --- 此处计算正确
# 

import torch
import triton
import triton.language as tl
from triton_activtion import *

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
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'], # 这个值的变化会带来 调优配置变化
)

@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,  # 矩阵指针 a(M, K) * b(K, N) = c(M, N)
    M, N, K,              # 矩阵的维度信息
    stride_am, stride_ak, # 对于矩阵a来说，stride_am 表示为了访问下一行，需要在a_ptr上相对增加多少
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,  
    ACTIVATION: tl.constexpr, 
    BIAS: tl.constexpr
):    
    pid = tl.program_id(axis=0)    # 块id,而不是线程id, 
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 每个红框里，有多少程序pid; GROUP_SIZE_M是方框在M维度上的尺寸
    group_id = pid // num_pid_in_group           # 本程序所在的group的id， 第几个红框 
    first_pid_m = group_id * GROUP_SIZE_M        # 在这个group中，第一个程序的行id
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  #  预防 num_pid_m 不是 GROUP_SIZE_M的倍数
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # 红框中的程序的行 id
    pid_n = (pid % num_pid_in_group) // group_size_m                 # 红框中的程序的列 id
    
    
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
        accumulator = tl.dot(a, b, accumulator) # dot在计算的时候还是float16，但是存的时候转float32
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    bias_ptrs = bias_ptr + (offs_am[:, None] * stride_am + offs_bn[None, :] * stride_bn)
    # bias_ptrs = bias_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    bias = tl.load(bias_ptrs)
    accumulator += bias
    
    if ACTIVATION == "gelu":
        accumulator = gelu(accumulator)
    c = accumulator.to(tl.float16)
        
    # 计算结果写回到c -------------------------------------------------------------
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs  = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c)


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)
    
def triton_matmul_bias(a, b, bias, activation=""):
    M, N, K = a.shape[0], b.shape[1], a.shape[1]
    c = torch.empty((M, N), device=a.device, dtype=DATA_TYPE)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_bias_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=activation,
        BIAS=True
    )
    return c

def torch_matmul_bias_act(a, b, bias):
    result = torch.matmul(a, b) + bias
    result_gelu = torch.nn.functional.gelu(result)
    return result_gelu

@torch.compile
def compiled_torch_matmul_bias_act(a, b, bias):
    result = torch.matmul(a, b) + bias
    result_gelu = torch.nn.functional.gelu(result)
    return result_gelu

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
        styles=[('blue', '-'), ('green', '-'), ('green', ':')],  # line styles 线条的样式
        ylabel="TFLOPS",  # label name for the y-axis y 轴的标签名称
        plot_name="matmul-bias-act",  # 图表的名称，也用作保存图表的文件名
        args={'M': 1024, 'K': 64},  # `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=DATA_TYPE)
    b = torch.randn((K, N), device='cuda', dtype=DATA_TYPE)
    bias =  torch.randn((M, N), device='cuda', dtype=DATA_TYPE)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_matmul_bias_act(a, b, bias), quantiles=quantiles)
    if provider == 'torch.compile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled_torch_matmul_bias_act(a, b, bias), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_matmul_bias(a, b, bias, "gelu"), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == '__main__':
    torch_cuda_identify()
    DATA_TYPE = torch.float16

    seq_len = 2048
    para_m = seq_len
    para_n = seq_len
    para_k = seq_len
    torch.manual_seed(0)
    # a = torch.randn((para_m, para_k), device='cuda', dtype=DATA_TYPE)
    # b = torch.randn((para_k, para_n), device='cuda', dtype=DATA_TYPE)
    # bias =  torch.randn((para_m, para_n), device='cuda', dtype=DATA_TYPE)
    
    # triton_output = triton_matmul_bias(a, b, bias, "gelu") 
    # torch_output = torch_matmul_bias_act(a, b, bias)
    # print(f"triton_output_with_fp16_inputs={triton_output}")
    # print(f"torch_output_with_fp16_inputs={torch_output}")

    # print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(torch_output - triton_output))))
    
    benchmark.run(show_plots=True, print_data=True, save_path='./03-matmul-bias-act')