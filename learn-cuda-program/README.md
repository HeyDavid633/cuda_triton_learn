# GEMM优化 熟悉TensorCore

1. [Github | cuda-samples TC-GEMM](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features/cudaTensorCoreGemm)：给了两版GEMM，简单的版本入手TC，复杂的版本看优化思路
2. 实例1：[知乎 ｜ Nvidia Tensor Core-CUDA HGEMM优化](https://zhuanlan.zhihu.com/p/639297098)  配套为萌哥推荐[Github-cuda-hgemm](https://github.com/Bruce-Lee-LY/cuda_hgemm)⭐️
3. 实例2：[知乎 ｜ 一步步优化GEMM](https://zhuanlan.zhihu.com/p/638522893) ；每步链接到Github有代码
4. 实例3：[B站 ｜ tensor core实现矩阵乘法](https://www.bilibili.com/video/BV1jwHMecESd/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243)；[B站 ｜ cuda实现matmul的重新解读](https://www.bilibili.com/video/BV16fykYzEem/?share_source=copy_web&vd_source=fc58db99551d5dde52430792ddbb9243)；，附带的代码 [Github | hpc_project](https://github.com/xgqdut2016/hpc_project/tree/main/cuda/matrix) ；[知乎 | CUDA实现矩阵乘的优化](https://zhuanlan.zhihu.com/p/708583794) --- 解读写的很烂，胜在**逐步优化够细**
   1. [知乎 ｜ 深入浅出GPU优化系列：GEMM优化（一）](https://zhuanlan.zhihu.com/p/435908830)： 最易懂
   2. [知乎 ｜ CUDA SGEMM矩阵乘法优化笔记](https://zhuanlan.zhihu.com/p/518857175)；重点看，性能高
   3. [知乎 ｜ 旷视 CUDA 矩阵乘法终极优化指南](https://zhuanlan.zhihu.com/p/410278370)

- CUDA Program Guide循环展开 [CUDA | pragma-unroll](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=unrolling#pragma-unroll)
- 关于共享内存的用法 [谭升 | CUDA共享内存概述](https://face2ai.com/CUDA-F-5-1-CUDA共享内存概述/)

> 数据类型均为 Float32；测试平台为4080-laptop（SM 58； Shared Mem Per SM 48KB； L2 48MB）
>
> 主要参考的实现为 [Github | hpc_project](https://github.com/xgqdut2016/hpc_project/tree/main/cuda/matrix) 

| 版本            | 1          | 2                             | 3                 | V4                             | V5                                   | 6                   | 7                       | 8                           | 9              | 10        | 11     |
| --------------- | ---------- | ----------------------------- | ----------------- | ------------------------------ | ------------------------------------ | ------------------- | ----------------------- | --------------------------- | -------------- | --------- | ------ |
| 说明            | 最朴素实现 | 一个线程处理多个元素设置TN TM | 引入了 Shared Mem | 一个线程处理多个元素Shared Mem | 数据重排优化加载到  Shared Mem的过程 | 引入float4 加速访存 | 解决bankconflict        | 降低了 Shared Mem的重复读取 | 引入流水线并行 | 循环展开  | CuBLAS |
| MNK1024时间(ms) | 8.0683     | 14.206 (不展开) 4.1514 (展开) | 7.9364            | 13.7001                        | 4.3980 (行主) 1.2975 (列主)          | 0.5880              | 0.6427 0.2260(填充展开) | 0.2315                      | 0.2280         | 0.2232    | 0.1134 |
| grid dim        | 32, 32, 1  | 8, 8, 1 (TN=4)                | 32, 32, 1         | 8, 8, 1 (TN=4)                 | 8, 8, 1                              | 8, 8, 1             | 8, 8, 1                 | 8, 8, 1                     | 8, 8, 1        | 8, 8, 1   |        |
| block dim       | 32, 32, 1  | 32, 32, 1                     | 32, 32, 1         | 32, 32, 1                      | 32, 32, 1                            | 16, 16, 1           | 16, 16, 1               | 16, 16, 1                   | 16, 16, 1      | 16, 16, 1 |        |

- V3 在 grid dim: 8, 8, 1（TN=8）  block dim: 16, 16, 1 时；7.5188ms

进一步的，对TensorCore的matmul继续，一个递进的优化方式：

| 版本    | 1                     | 2                                     | 3                                                         | V4               | V5                                   |
| ------- | --------------------- | ------------------------------------- | --------------------------------------------------------- | ---------------- | ------------------------------------ |
| 说明    | 最朴素实现的TC matmul | 继承朴素实现的思路，但从shred Mem来取 | blockIdx.x处理B的列，blockIdx.y处理A的行 ，没用shared mem | block版TC matmul | 利用了shared Mem 的 block版TC matmul |
| MNK512  | 0.0359                | 0.1046                                | 0.0355                                                    | 0.0354           | 0.0378                               |
| MNK1024 | 0.2040                | 0.6111                                | 0.2037                                                    | 0.2037           | 0.2263                               |
| MNK2048 | 1.5797                | 4.4083                                | 1.5850                                                    | 1.5815           | 1.6687                               |
| MNK4096 | 16.8488               | 34.9168                               | 11.8984                                                   | 11.8407          | 13.9711                              |
| MNK8192 | 311.6743              | 324.3732                              | 116.8367                                                  | 117.0343         | 139.8621                             |

| 矩阵尺寸MNK | 基准 cublas | 优化V10 循环展开 | TC V3    |
| ----------- | ----------- | ---------------- | -------- |
| 512         | 0.0599      | 0.2196           | 0.0355   |
| 1024        | 0.1134      | 0.2232           | 0.2037   |
| 2048        | 0.7502      | 1.1457           | 1.5850   |
| 4096        | 12.7228     | 15.7630          | 11.8984  |
| 8192        | 45.8742     | 97.1034          | 116.8367 |



# ncu与nsys 分析kerenl性能

- `nsys：Nvidia Nsight Systems`粗粒度分析 / `ncu：Nvidia Nsight Compute`细粒度分析 

  - 前者是Sys级的，不仅对GPU，还对CPU/IP以及OS都有分析到 --- 就是数据指标看球不懂～

- 两者都是对kenrel层面的分析（哪怕你编译成了`.so`），比如我的任务中是`python XXX.py`其中调用的cuda都能看到

  - 因为是直接对GPU做的监测，无论以什么方式运行的程序，只要用到的CUDA kernel被提交给了GPU，那么就都能看到
    - 所以不仅仅是`nvcc sample.cu -o sample.out `生成执行的文件能用命令`ncu sample.out`监测，像我的python脚本也可以用`ncu python sample.py`来监测

- 使用：Linux监测 和 查看 分离

  - 本地需要在[Nvidia Developer / CUDA compute](https://developer.nvidia.com/tools-overview)中按照你的本地系统来下载查看软件

  - 通过先在命令行中生成对应的`.ncu-rep`或`.nsys-rep`；然后拉到本地-用查看软件打开

  - ```Python
    #nsy的分析方法 - 生成的报告文件在当前路径下，名为 XXX.nsys-rep
    nsys profile python syncfree1.py 8 1 256 12 64
    
    #ncu的分析 - 可以在命令行就查看 也可以保存为报告 
    #在命令行中直接查看；方便，但kernel一多输出就很难接受了；不需要对应的查看软件
    ncu -o python syncfree1.py 8 1 256 12 64  
    #生成报告，名为 XXX.ncu-rep
    ncu -o rep1 python syncfree1.py 8 1 256 12 64
    
    # 简简单单分析一下子
    ncu ./shared_read_test 
    # 开启详细模式，并且结果输出到txt文件; 有没有多很多
    ncu --set full ./shared_read_test > ncu_report.txt
    ```

关于Occupancy占用率的解释 [知乎 - GPU基础：Occupancy、wave and tail effect](https://zhuanlan.zhihu.com/p/657005697)

- **高占用率**不总是代表高性能；没那么高的时候，就是有限的任务，但资源很丰富能干好 
- 但是（过）**低占用率**总是会干扰隐藏内存延迟的能力，性能会下降
- 所以存在一个最佳点，超过这个点以后 ，提高占用率不会提高性能



# Appdix 在CUDA之外

## Appdix1 - 给GPU函数计时

- Reference：[谭升博客 - 2.2 给核函数计时](https://face2ai.com/CUDA-F-2-2-核函数计时/)
- 那么正确的计时方式 gettimeofday - 使用绝对时间
  - `#include <sys/time.h>  gettimeofday() `
  - 错误的计时方式：对于Clock() 记录的是CPU滴答数
  - `cudaDeviceSynchronize();`同步函数
    - 必须要加一个**同步函数**等待kernel执行完毕
    - 如果不加这个同步函数，那么测试的时间是：从调用kernel，到kernel返回给主机线程的时间段(这个时间可能会非常短，远远短于实际的kernel执行时间)；**不是kernel的执行完成时间**
- nvprof的工具，在我们目前常用的CUDA10、11、12已经弃用了，不消关注；
  - 当前实验室用的主流GPU架构也不支持 使用这个

## Appdix2 - reduction

- Reference:
  - 快速学习上手 - [知乎- HPC方向学生笔记](https://zhuanlan.zhihu.com/p/365581043)
    - 辅助理解[谭升博客 - 3.4-避免分支分化](https://face2ai.com/CUDA-F-3-4-避免分支分化/)--- 和上面说的是一回事，但更讲人话
    - 关于bank冲突 [共享内存与Thread同步](https://blog.csdn.net/sunmc1204953974/article/details/51078818)
  - 根本的来源 - [Nvidia官方 - reduciton 7级优化](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
- 所谓7级优化：
  - Interleaved addressing with divergent branching ；具有不同分支的交错寻址
  - Interleaved addressing with bank conflicts ；bank冲突的交错寻址
  - Sequential addressing 顺序寻址
  - First add during global load 首先在全局加载期间添加
  - Unroll last warp 展开最后一个warp
  - Completely unroll 完全展开
  - Mltiple elements per thread 每个线程多个元素