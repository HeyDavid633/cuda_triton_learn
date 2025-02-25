# 2024.12.08  周日
# 融合的 softmax代码，以展示融合kernel对访存密集型应用带来的好处
# 现在的实现为了防止数值溢出，减去了最大值

import torch

import triton
import triton.language as tl
from triton.runtime import driver

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


@torch.jit.script
def naive_softmax(x):
    # 此处对于每行剪去了最大值，以保证数据不溢出 
    # softmax(x + a) = softmax(x)  证明可见 https://blog.csdn.net/benben044/article/details/126662800
    
    # x.max(dim=1)：这个操作沿着指定的维度（这里是 dim=1，即列方向）对张量进行最大值计算
    # 返回的是一个包含两个元素的元组：[0]第一个元素是最大值的张量. [1]第二个元素是这些最大值对应的索引张量。
    x_max = x.max(dim=1)[0]   # (M,)
    z = x - x_max[:, None]    # (M, N) - (M, 1) -> (M, N) 广播
    exp_z = torch.exp(z)      # (M, N) -> (M, N)
    epsilon_exp_z = exp_z.sum(dim=1)  # (M, N) -> (M, 1)
    result = exp_z / epsilon_exp_z[:, None]    # (M, N) / (M, 1) -> (M, N) 广播
    
    return result

@torch.compile
def torch_compiled_softmax(x):
    x_max = x.max(dim=1)[0]   # (M,)
    z = x - x_max[:, None]    # (M, N) - (M, 1) -> (M, N) 广播
    exp_z = torch.exp(z)      # (M, N) -> (M, N)
    epsilon_exp_z = exp_z.sum(dim=1)  # (M, N) -> (M, 1)
    result = exp_z / epsilon_exp_z[:, None]    # (M, N) / (M, 1) -> (M, N) 广播
    
    return result


@triton.jit
def softmax_kernel(input_ptr,
                   output_ptr,
                   input_row_stride,
                   output_row_stride,
                   n_rows, 
                   n_cols,
                   BLOCK_SIZE: tl.constexpr,
                   num_stages: tl.constexpr # 软件流水线阶段数，循环流水线
                   ):
    row_start = tl.program_id(0) # 块的id
    row_step  = tl.num_programs(0) # 多少个块
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        
        # 块 的来决定列上做多少加和
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        
        # print("tl.max(row, axis=0) = ", tl.max(row, axis=0))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)
        
    

def triton_softmax(x):
    n_rows, n_cols = x.shape
    # 每次循环迭代的块大小是大于等于 x 列数的最小二的幂
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    # 通过增加每行分配的线程数 使编译器使用更多的warp
    num_warps = 8
    # 软件流水线阶段的数量
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    
    # 为结果分配空间
    y = torch.empty_like(x)
    
    # 预编译内核以获取寄存器使用情况并计算线程占用情况 --- warmup
    kernel, num_programs = kernels.get(BLOCK_SIZE, (None, 0))
    if kernel is None:
        kernel = softmax_kernel.warmup(x, y, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                       num_stages=num_stages, num_warps=num_warps, grid=(1, ))
        kernel._init_handles()
        n_regs = kernel.n_regs
        size_smem = kernel.metadata.shared
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
        occupancy = min(occupancy, SIZE_SMEM // size_smem)
        num_programs = NUM_SM * occupancy
        kernels[BLOCK_SIZE] = (kernel, num_programs)
        
    num_programs = min(num_programs, n_rows)
    
    kernel[(num_programs, 1, 1)](
        x, y,
        x.stride(0), y.stride(0),
        n_rows, n_cols)  # 实际调度triton kernel
    
    return y
    

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  #  用作图表 x 轴的参数名
        x_vals=[128 * i for i in range(2, 100)],  #  `x_name` `x_name` 的不同可能值
        line_arg='provider',  #  参数名，其值对应于图表中不同线条
        line_vals=['triton', 'torch', 'torch.jit', 'torch.compile'],  #  `line_arg`` `line_arg` 的可能值
        line_names=[
            "Triton",
            "Torch",
            "Torch.jit",
            "Torch.compile",
        ],  # label name for the lines 线条的标签名称
        styles=[('blue', '-'), ('green', '-'), ('green', '-.'), ('green', ':')],  # line styles 线条的样式
        ylabel="GB/s",  # label name for the y-axis y 轴的标签名称
        plot_name="softmax-performance",  # 图表的名称，也用作保存图表的文件名
        args={'M': 4096},  # `x_names` 和 `y_name` 中未包含的函数参数的值
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1), quantiles=quantiles)
    if provider == 'torch.jit':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_softmax(x), quantiles=quantiles)
    if provider == 'torch.compile':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch_compiled_softmax(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: triton_softmax(x), quantiles=quantiles)
    gbps = lambda ms: 2 * x.nelement() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    torch_cuda_identify()
    torch.manual_seed(0)
    x = torch.randn(1024, 1024, device='cuda')
    
    device = torch.cuda.current_device()
    properties = driver.active.utils.get_device_properties(device)
    NUM_SM = properties["multiprocessor_count"]
    NUM_REGS = properties["max_num_regs"]
    SIZE_SMEM = properties["max_shared_mem"]
    WARP_SIZE = properties["warpSize"]
    target = triton.runtime.driver.active.get_current_target()
    kernels = {}
    
    print("NUM_SM {}, NUM_REGS {}, SIZE_SMEM {}, WARP_SIZE {},".format(NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE))
    
    torch_output = torch.softmax(x, axis = 1)
    triton_output = triton_softmax(x)
    
    print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(torch_output - triton_output))))
    # assert torch.allclose(torch_output, triton_output), (torch_output, triton_output)
    
    
    benchmark.run(show_plots=True, print_data=True, save_path='./02-fused-softmax')