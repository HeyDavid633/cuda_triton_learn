import time

import torch
import triton
import triton.language as tl

@triton.jit
def gemm_chain_kernel(
    Q, K, V, Out,
    stride_qm, stride_qk,
    stride_kn, stride_kk,
    stride_vk, stride_vn,
    stride_om, stride_on,
    N_CTX,
    BLOCK_M: tl.constexpr, 
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    start_m = tl.program_id(0)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    off_v = offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    # loop over k, v and update accumulator
    for start_n in range(0, (start_m + 1) * BLOCK_M, BLOCK_N):
        # -- compute qk ----
        k = tl.load(k_ptrs)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        # update acc
        p = qk.to(Q.dtype.element_ty)
        v = tl.load(v_ptrs)
        acc += tl.dot(p, v)
        # update pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = offs_m[:, None] * stride_om + offs_n[None, :] * stride_on

    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc)

# wrapper函数
# q, k, v分别代表query, key, value三个输入Tensor。
def gemm_chain(q, k, v):
    # 初始化一个与q相同形状和类型的空Tensor，用于存储输出结果。
    o = torch.empty_like(q)
    # 这几行设置了几个关键的性能调优参数，包括处理块的大小（BLOCK_M, BLOCK_N）和
    # 计算阶段的数量（num_stages）。num_warps指的是每个CUDA block中warp的数量。
    BLOCK_M = 16
    BLOCK_N = 16
    num_stages = 2

    # 决定了一个线程块的线程数量
    num_warps = 4
    # 计算Triton kernel的网格尺寸
    grid = (triton.cdiv(q.shape[0], BLOCK_M), 1)

    # for debug purpose
    # print(f"block_M={BLOCK_M}")
    # print("q.shape[0]=", q.shape[0])
    # print("q.shape[1]=", q.shape[1])
    # print(o.shape)
    # print("tensor o:",o)


    gemm_chain_kernel[grid](
        q, k, v, o,  #
        q.stride(0), q.stride(1),  #
        k.stride(0), k.stride(1),  #
        v.stride(0), v.stride(1),  #
        o.stride(0), o.stride(1),  #
        N_CTX=q.shape[0],  #
        BLOCK_M=BLOCK_M,  #
        BLOCK_N=BLOCK_N,  #
        BLOCK_DMODEL=q.shape[1],  #
        num_warps=num_warps,  #
        num_stages=num_stages  #
    )
    return o

# torch标准的 gemm chain 实现
def torch_matmul_matmul(A, B, D):
    result = torch.matmul(torch.matmul(A, B), D)
    return result

@torch.compile
def compiled_torch_matmul_matmul(A, B, D):
    result = torch.matmul(torch.matmul(A, B), D)
    return result

if __name__ == "__main__":

    torch.manual_seed(0)
    seq_len = 16
    head_dim = 32
    DATA_TYPE = torch.float32
    
    A = torch.ones((seq_len, head_dim), device='cuda', dtype=DATA_TYPE)
    B = torch.ones((head_dim, seq_len), device='cuda', dtype=DATA_TYPE)
    D = torch.ones((seq_len, head_dim), device='cuda', dtype=DATA_TYPE)

    # 测试 torch_matmul_matmul 函数的执行时间
    start_time = time.time()
    compiled_torch_output  = compiled_torch_matmul_matmul(A, B, D)
    end_time = time.time()
    compiled_torch_matmul_matmul_time = end_time - start_time

    # 测试 gemm_chain 函数的执行时间
    start_time = time.time()
    triton_output = gemm_chain(A, B, D)
    end_time = time.time()
    gemm_chain_time = end_time - start_time

    # 输出结果  
    print(f"compiled_torch_matmul_matmul execution time: {compiled_torch_matmul_matmul_time:.6f} seconds")
    print(f"gemm_chain execution time: {gemm_chain_time:.6f} seconds")


    print(f"torch_output={compiled_torch_output}")
    print(f"torch_output_shape={compiled_torch_output.shape}")
    print(f"triton_output={triton_output}") 
    print(f"triton_output_shape={triton_output.shape}") 
    print('Mean difference :{:.4f}'.format(torch.mean(torch.abs(compiled_torch_output - triton_output))))

    # 性能测试
    # benchmark.run(show_plots=True, print_data=True, save_path='./04-matmul-matmul')