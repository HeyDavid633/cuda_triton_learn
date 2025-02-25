# 2024.12.08  周日
# 初步从头来写triton代码，以向量add为例
# https://triton.hyper.ai/docs/getting-started/tutorials/vector-addition 
#
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

# 在这个函数里 没有torch的痕迹
@triton.jit
def add_kernel(x_ptr, # x 进来，直接以指针来操作
               y_ptr, 
               output_ptr,
               n_elements, 
               BLOCK_SIZE: tl.constexpr, 
               ):
    pid = tl.program_id(axis = 0)   #获取块的id 
    block_start = pid * BLOCK_SIZE  #获取当前块的要操作的元素的首地址, 但是最后还是分到了thread里来执行
    # 索引一个block中每个元素，例如BLOCK_SIZE = 64， 则生成 [0, 63]的数组 --- 数组里的元素去到了每个块里作索引
    # 所以返回值并不是 一个数组 element_offsets[], 而是在每个block里执行的 索引值（int）
    # tl.arange 返回的 和 torch.arange返回的并不一样； 而且tl.arange只能在@triton.jit修饰的函数里出现
    element_offsets = block_start + tl.arange(0, BLOCK_SIZE)   
    
    # print("element_offsets: ", element_offsets)
    
    # 预防 n_elements 并不恰好是 BLOCK_SIZE的整数倍，搞一个mask
    # 这里作的比较，实际上是int值的比较；
    mask = element_offsets < n_elements   
    
    x = tl.load(x_ptr + element_offsets, mask = mask)
    y = tl.load(y_ptr + element_offsets, mask = mask)
    output = x + y
    
    # print("x_ptr + element_offsets: ", x_ptr + element_offsets)
    
    tl.store(output_ptr + element_offsets, output, mask=mask)
    
    
    

# 对于triton包装函数编程而言，最好写一个类型注解，例如 x:torch.Tensor, y:torch.Tensor
def tirton_add(x, y):
    # 利用assert断言来检验输入的正确性
    assert x.is_cuda and y.is_cuda, "[ERROR] Input x or y are not on CUDA"
    # 预先分配结果的空间
    output = torch.empty_like(x)
    n_elements = output.numel()
    # print(output.size(), output.numel(), output.shape)
    # print(type(output.size()), type(output.numel()), type(output.shape))
    
    # meta 参数包含了配置信息和动态参数
    # meta['BLOCK_SIZE']：从 meta 字典中提取的一个特定键 'BLOCK_SIZE' 的值
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # @triton.jit所修饰的函数，可以通过 启动网格grid的索引，生成GPU kernel
    # 其实就是将 @triton.jit所修饰的函数 看作CUDA kernel，这个grid看成GridSize
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE = 1024)
    # 此时立刻返回 output的handle，但是kenrel还在异步执行，所以如果计时需要 torch.cuda.synchronize()
    return output
    
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot. 用作绘图 x 轴的参数名称。
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`. `x_name` 的不同可能值。
        x_log=True,  # x axis is logarithmic. x 轴为对数。
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot. 参数名称，其值对应于绘图中的不同线条。
        
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`. `line_arg` 的可能值。
        line_names=['Triton', 'Torch'],  # Label name for the lines. 线条的标签名称。
        styles=[('blue', '-'), ('green', '-')],  # Line styles. 线条样式。
        
        ylabel='GB/s',  # Label name for the y-axis. y 轴标签名称。
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot. 绘图名称。也用作保存绘图的文件名。
        args={},  # Values for function arguments not in `x_names` and `y_name`. 不在 `x_names` 和 `y_name` 中的函数参数值。
    ))    
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: tirton_add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

    
if __name__ == "__main__":
    torch_cuda_identify()
    
    torch.manual_seed(0)
    size = 8192
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    torch_output = x + y
    triton_output = tirton_add(x, y)
    print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(torch_output - triton_output))))
     
    # benchmark.run(print_data=True, show_plots=True, save_path='./01-vectoradd/results/')
    

   
    