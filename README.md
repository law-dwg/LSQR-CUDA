# LSQR-CUDA
## Overview
LSQR-CUDA is written by Lawrence Ayers under the supervision of Stefan Guthe of the [GRIS](https://www.informatik.tu-darmstadt.de/gris/startseite_1/team/index.de.jsp) institute at the Technische Universität Darmstadt. It is a CUDA port of the LSQR algorithm of Chris Paige and Michael Saunders

The goal of this work was to accelerate the computation time of the well-known [LSQR](https://web.stanford.edu/group/SOL/software/lsqr/) algorithm using a CUDA capable GPGPU.

The LSQR algorithm is an iterative method used to find the solution x for either of the following problems:
* Ax=b
* min(||Ax-b||)

where A is a large, often sparse, square or rectangular matrix, and b is a vector of size #A-rows.

LSQR was first authored by Chris Paige and Michael Saunders in their publication [here](https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf), and has since been widely used for various applications.

## Requirements
LSQR-CUDA has the following requirements:
* *nix system or WSL for windows machines
* CUDA Capable GPGPU
* CUDA (nvcc) v11 or higher 
* g++ v11 or higher
* make

## Execution
To run the software, enter the [source](source/) directory and type the following command into your terminal
```
make run
```
### Inputs
You will then be asked if you would like automatic test inputs generated for you and what sparsity you would like to have Matrix A set to. The range of auto-generated inputs is set in the start, end, and iteration variables of [main.cu](source/gpu/main.cu#107), but can be manually changed if desired. If you have your own inputs available, you will need to save them as files with .mat (dense and sparse matricies) and .vec (vectors) extensions in the [input](source/input/) directory. These must use a white space delimiter: " ", and have a number of values such that Ax=b can be satisfied.

Inputs should have the following notation:
* ```#Arows_#Acols_A_#sparsity.mat```
* ```#Arows_1_b.vec```

As an example, a sparse matrix A with 1500 rows, 2000 columns, and a sparsity of 0.75% would have the following input files:
* ```1500_2000_A_75.mat```
* ```1500_1_b.vec```

### Outputs
The solution, x, will be written to [output](source/output/) in a directory corresponding to the time of execution (Year-Month-DayTHourMinute) in the format:
* ```YYYY-MM-DDTHHMM/#Acols_1_x_implementation.vec```

for the above example, the x output file would look like this:
* ```YYYY-MM-DDTHHMM/2000_1_x_CUDA-SPARSE.vec```

The 5 different implementations created for this work will then run on each set of inputs located in the [input](source/input/) directory, with the runtime of each saved to a csv called ```YYYY-MM-DDTHHMM/YYYY-MM-DDTHHMM_LSQR-CUDA.csv```
___

<details open>
<summary><b>Table of Contents</b></summary>

* [1. Introduction](#Introduction)
* [2. Background](#Background)
* [3. Methods](#Methods)
    + [3.1. Cpp-DENSE](#Cpp-DENSE)
    + [3.2. CUDA-DENSE](#CUDA-DENSE)
    + [3.3. CUDA-SPARSE](#CUDA-SPARSE)
    + [3.4. cuBLAS-DENSE](#cuBLAS-DENSE)
    + [3.5. cuSPARSE-SPARSE](#cuSPARSE-SPARSE)
* [4. Results](#Results)
    + [4.1. Speedup](#Speedup)
    + [4.2. Accuracy](#Accuracy)
* [5. Conclusion](#Conclusion)

</details>

___

<a id="Introduction"></a>

## 1. Introduction
The purpose of this work was to implement the LSQR algorithm on a CUDA-capabale GPU to analyze any potential runtime speedups in comparison to a standard, sequential CPU implementation. When run in CUDA, many matrix operations (e.g. multiplication, euclidean norm, addition, subtraction, etc.) can be run in parallel, and can, therefore, decrease computation time.

This work has both sequential and parallel implementations of LSQR that are intended for both sparse and dense inputs. The 5 implementations are, correspondingly listed as follows:

1.  Cpp-DENSE (CPU)
1.  CUDA-DENSE (GPU)
1.  CUDA-SPARSE (GPU)
1.  cuBLAS-DENSE (GPU)
1.  cuSPARSE-SPARSE (GPU)

A sparse sequential algorithm was not explicitly created for this work, rather, the robust [scipy-lsqr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html) algorithm was used instead as the baseline for verifying results and comparison of runtimes for sparse inputs.
___
<a id="Background"></a>

## 2. Background
___
<a id="Methods"></a>

## 3. Methods
The LSQR algorithm in this work is largely based off the scipy-lsqr [source code](https://github.com/scipy/scipy/blob/v1.6.1/scipy/sparse/linalg/isolve/lsqr.py#L96-L568) as well as the [C++ port](https://github.com/tvercaut/LSQR-cpp) provided by Luis Ibanez. In this work, the general LSQR algorithm is located in [lsqr.hpp](source/cpu/lsqr.hpp), whereby each implementation (listed above) is passed to the function as a combination of different class types (via a template).

<a id="Cpp-DENSE"></a>

## CPU Implementations
## [3.1. Cpp-DENSE](source/cpu/vectorCPU.hpp)
The Cpp-Dense implementation is written in C++ and runs the sequentially on the CPU. This implementation uses Naive operations for add, subtract, multiply, Dnrm2, etc. It is the slowest of the implementations and used as a baseline to compare to Dense GPU implementations.
Corresponding source files are [vectorCPU.cpp](source/cpu/vectorCPU.cpp) and [vectorCPU.hpp](vectorCPU.hpp)

## [scipy-lsqr](https://github.com/scipy/scipy/blob/v1.6.1/scipy/sparse/linalg/isolve/lsqr.py#L96-L568)
Scipy's lsqr solver runs on either sparse or dense inputs and is used as a baseline to compare to the sparse LSQR-CUDA implementations created here. Related information can be found on scipy's [website](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html), and its use in this work can be found in [lsqr.py](python/lsqr.py)

<a id="CUDA-DENSE"></a>

## GPU Implementations
All source files pertaining to GPU implementations can be found in in the [gpu](source/gpu/) directory.

For all kernels designed in this work, the blocksize (i.e. the number of threads in a block) is set to a constant value found in [utils.cuh](source/gpu/utils.cu#L3). This value can be changed if desired. For best results using the GPU in this work, a blocksize of 16*16 (256) threads was used.

The kernels used for these implementations are where most development time for LSQR-CUDA were spent. 

## [3.2. CUDA-DENSE](source/gpu/vectorCUDA.cuh)
The CUDA-DENSE implementation is written with the standard CUDA library, and executes many of its own [kernels](source/gpu/kernels.cuh) for various vector operations. This implementation has two dense inputs of type VectorCUDA and runs them through lsqr with accelerated multiplication, addition/subtraction, euclidean norm, and transpose operations. All operations used for this implementation are defined within the [VectorCUDA](source/gpu/vectorCUDA.cu) class.

An output of nvprof for test-run (2500_2500_A_0.mat) of this implementation can be seen here:

<details close>
<summary><b>nvprof output of CUDA-DENSE</b></summary>

```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.00%  66.9277s      9941  6.7325ms  6.6156ms  7.4214ms  multiplyTiled(double*, unsigned int*, unsigned int*, double*, unsigned int*, unsigned int*, double*)
                    0.51%  351.46ms     34792  10.101us  8.8320us  34.528us  scale(double*, double, double*, unsigned int*, unsigned int*, bool)
                    0.33%  225.62ms     19880  11.349us  8.8960us  32.864us  add(double*, double*, unsigned int*, unsigned int*, double*, bool)
                    0.33%  223.22ms     64615  3.4540us  2.7840us  3.6601ms  [CUDA memset]
                    0.29%  197.12ms    159068  1.2390us     864ns  30.217ms  [CUDA memcpy HtoD]
                    0.20%  137.70ms     14912  9.2340us  8.8000us  32.513us  dnrm2(double*, unsigned int, unsigned int, double*, double*)
                    0.14%  96.548ms     34802  2.7740us  1.7600us  23.009us  [CUDA memcpy DtoD]
                    0.13%  87.822ms     14912  5.8890us  5.0230us  26.016us  maxVal(double*, unsigned int, unsigned int, double*)
                    0.06%  37.655ms     29827  1.2620us  1.1200us  25.344us  [CUDA memcpy DtoH]
                    0.01%  7.9884ms         1  7.9884ms  7.9884ms  7.9884ms  transposeTiled(double*, double*, unsigned int*, unsigned int*)
```

</details>

### Multiplication
From the nvprof output above it is clear to see that the most time intensive operation of LSQR is the matrix-vector and vector-vector multiplication operations. Since CUDA-DENSE works only with dense inputs, this operation is treated the same for both matrix-vector and vector-vector multiplication (i.e. neither matrix nor vector are in a compressed format). 

A naive approach to parallel multiplication is to have a each thread solve for one entry in the solution matrix, i.e. a thread accesses one row of the first input and one column of the second input from global memory to perform the dot product of these two arrays in a loop. Since the latency of global memory accesses can be quite high, a cached, "tiled", memory solution is used instead, [multiplyTiled](source/gpu/kernels.cu#L125). A [multiplyNaive](source/gpu/kernels.cu#L114) kernel is available for reference.

In the [multiplyTiled](source/gpu/kernels.cu#L125) approach to parallel matrix multiplication, inputs are first loaded into GPU-cached (["shared"](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)) memory, or "tiles", that iteratively "sweep" across inputs,continuously summing up the dot product result with the running total in each iteration. Each thread works in parallel towards calculating one value in the resultant matrix. An excellent visual representation of this can be found in Penny Xu's work, [Tiled Matrix Multiplication](https://penny-xu.github.io/blog/tiled-matrix-multiplication).

In multiplyTiled, the use of cache memory halves the number of global memory accesses required for each thread in comparison to the naive approach. For a dense input of 2500x2500, this implementation of lsqr has a speedup of about 1.5x when switching from multiplyNaive to multiplyTiled.

### Scale, Addition, and Subtraction
Due to their already low computation time within the LSQR algorithm, the [scale](source/gpu/kernels.cu#L168), [add, and subtract](source/gpu/kernels.cu#L186) operations use naive approaches. No further development for these operations was deemed necessary.

### Euclidean Norm
The euclidean norm, or Dnrm2 operation, is split into two different kernels. The first, [maxVal](source/gpu/kernels.cu#61), finds the max value within the matrix or vector, and the second, [dnrm2](source/gpu/kernels.cu#86), then divides all values by this max value whilst performing the necessary multiplication and addition operations, e.g. a[0]/maxVal ** 2 + a[1]/maxVal ** 2 + ... The solution is then found by taking the square root of the kernel result, before multiplying it by the max-value found in the previous kernel. This is the same method used by Ibanez in [LSQR-cpp](https://github.com/tvercaut/LSQR-cpp) and ensures numerical stability.

Standard, parallel reduction techniques are used for both of these kernels, whereby the number of working threads in a block is halved in each iteration, and memory accesses are coalesced. Much of the development here was inspired by Mark Harris' webinar, "[Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)" and the topics covered in the PMPP course at TUD. Both these kernels also utilize cache memory as to decrease memory access latency.

### Matrix Transpose
Like the multiplcation operation, the matrix transpose operation, [transposeTiled](gpu/source/kernels.cu#201), also utilizes a "tiled" approach, where a cached "tile" is swept across the matrix iteratively transposing it section by section. While the multiplyTiled kernel requires two seperate tiles (one for each input), transposeTiled requires only one that temporarily stores a section of the matrix before loading it to global memory with swapped indices, e.g. ```output[3][2]=input[2][3]```. This method outlined in Nvidias blog post, "[An Efficient Matrix Transpose in CUDA C/++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)", authored by Mark Harris.

<a id="CUDA-SPARSE"></a>

## [3.3. CUDA-SPARSE](source/gpu/matrixCUDA.cuh)
The CUDA-SPARSE implementation is written with the standard CUDA library, and has inputs of type MatrixCUDA (matrix A) and VectorCUDA (vector b). When loading A into MatrixCUDA, it is converted into compressed sparse row (CSR) form, reducing its size and, depending on its sparsity, required computation effort.

All operations used here are the same as the CUDA-DENSE implementation besides matrix-vector multiplication and matrix transpose operations. These sparse matrix operations can all be found within the MatrixCUDA [source code](source/gpu/matrixCUDA.cu).

<details close>
<summary><b>nvprof output of CUDA-SPARSE</b></summary>

```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   97.49%  56.2308s      9947  5.6530ms  5.6171ms  5.6886ms  spmvCSRVector(unsigned int*, int*, int*, double*, double*, double*)
                    0.60%  346.90ms     34813  9.9640us  8.8630us  31.328us  scale(double*, double, double*, unsigned int*, unsigned int*, bool)
                    0.39%  225.51ms     19892  11.336us  8.8000us  34.208us  add(double*, double*, unsigned int*, unsigned int*, double*, bool)
                    0.39%  225.08ms     64657  3.4810us  2.6880us  3.6540ms  [CUDA memset]
                    0.37%  211.50ms    159166  1.3280us     863ns  30.498ms  [CUDA memcpy HtoD]
                    0.24%  137.06ms     14921  9.1850us  8.7670us  30.495us  dnrm2(double*, unsigned int, unsigned int, double*, double*)
                    0.17%  95.903ms     34823  2.7540us  2.1440us  23.008us  [CUDA memcpy DtoD]
                    0.15%  87.760ms     14921  5.8810us  5.0230us  28.544us  maxVal(double*, unsigned int, unsigned int, double*)
                    0.07%  37.599ms     29846  1.2590us  1.1200us  14.752us  [CUDA memcpy DtoH]
                    0.04%  22.277ms         3  7.4258ms  7.3831ms  7.4588ms  void cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=1, bool=0, int, int, int>(cub::DeviceRadixSortPolicy<int, int, int>::Policy700 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=1, bool=0, int, int, int>*, bool=1 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=1, bool=0, int, int, int>**, bool=0*, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=1, bool=0, int, int, int>**, int, int, cub::GridEvenShare<cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=1, bool=0, int, int, int>**>)
                    0.03%  18.866ms         1  18.866ms  18.866ms  18.866ms  void cusparse::gather_kernel<unsigned int=128, double, double>(double const *, int const *, int, double*)
                    0.03%  14.797ms         2  7.3985ms  7.3967ms  7.4004ms  void cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=0, bool=0, int, int, int>(cub::DeviceRadixSortPolicy<int, int, int>::Policy700 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=0, bool=0, int, int, int>*, bool=0 const *, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=0, bool=0, int, int, int>**, bool=0*, cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=0, bool=0, int, int, int>**, int, int, cub::GridEvenShare<cub::DeviceRadixSortDownsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=0, bool=0, int, int, int>**>)
                    0.02%  9.9703ms         1  9.9703ms  9.9703ms  9.9703ms  void cusparse::gather_kernel<unsigned int=128, int, int>(int const *, int const *, int, int*)
                    0.01%  5.6034ms         3  1.8678ms  1.8501ms  1.8894ms  void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=1, bool=0, int, int>(cub::DeviceRadixSortPolicy<int, int, int>::Policy700 const *, bool=1*, cub::DeviceRadixSortPolicy<int, int, int>::Policy700 const *, int, int, cub::GridEvenShare<cub::DeviceRadixSortPolicy<int, int, int>::Policy700 const *>)
                    0.01%  3.7753ms         2  1.8876ms  1.8756ms  1.8996ms  void cub::DeviceRadixSortUpsweepKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, bool=0, bool=0, int, int>(cub::DeviceRadixSortPolicy<int, int, int>::Policy700 const *, bool=0*, cub::DeviceRadixSortPolicy<int, int, int>::Policy700 const *, int, int, cub::GridEvenShare<cub::DeviceRadixSortPolicy<int, int, int>::Policy700 const *>)
                    0.00%  2.3064ms         1  2.3064ms  2.3064ms  2.3064ms  void cub::DeviceReduceByKeyKernel<cub::DispatchReduceByKey<int*, int*, cub::ConstantInputIterator<int, int>, int*, int*, cub::Equality, cub::Sum, int>::PtxReduceByKeyPolicy, int*, int*, cub::ConstantInputIterator<int, int>, int*, int*, cub::ReduceByKeyScanTileState<int, int, bool=1>, cub::Equality, cub::Sum, int>(int*, int, int, cub::ConstantInputIterator<int, int>, int*, int*, int, cub::Equality, cub::Sum, int)
                    0.00%  2.1231ms         1  2.1231ms  2.1231ms  2.1231ms  void cusparse::sequence_kernel<unsigned int=128, int>(int*, int, cusparse::sequence_kernel<unsigned int=128, int>)
                    0.00%  1.8211ms         1  1.8211ms  1.8211ms  1.8211ms  void cusparse::csr2csr_kernel_v1<unsigned int=128, unsigned int=896, int=0, int>(int const *, int const *, int, int const **)
                    0.00%  54.239us         5  10.847us  10.176us  11.456us  void cub::RadixSortScanBinsKernel<cub::DeviceRadixSortPolicy<int, int, int>::Policy700, int>(int*, int)
                    0.00%  13.664us         1  13.664us  13.664us  13.664us  void cusparse::merge_path_partition_kernel<unsigned int=128, unsigned int=896, int=0, int>(int const *, int, cusparse::merge_path_partition_kernel<unsigned int=128, unsigned int=896, int=0, int>, int, int*, int)
                    0.00%  10.880us         1  10.880us  10.880us  10.880us  void cub::DeviceScanKernel<cub::DispatchScan<int*, int*, cub::Sum, int, int>::PtxAgentScanPolicy, int*, int*, cub::ScanTileState<int, bool=1>, cub::Sum, int, int>(int*, cub::Sum, int, int, int, cub::DispatchScan<int*, int*, cub::Sum, int, int>::PtxAgentScanPolicy, int*)
                    0.00%  5.3120us         1  5.3120us  5.3120us  5.3120us  void cusparse::scatter_kernel<unsigned int=128, int=0, int, int>(int const *, int const *, int, int*)
                    0.00%  4.9280us         1  4.9280us  4.9280us  4.9280us  void cub::DeviceCompactInitKernel<cub::ReduceByKeyScanTileState<int, int, bool=1>, int*>(int, int, int)
                    0.00%  3.1360us         1  3.1360us  3.1360us  3.1360us  void cub::DeviceScanInitKernel<cub::ScanTileState<int, bool=1>>(int, int)
```
</details>

### SpMV
The nvprof output above shows that the most expensive operation used for this implementation is sparse matrix-vector multiplication operation, or SpMV, that solves for the product between a sparse matrix in compressed format (CSR) and a vector. The result is a vector of size #A-rows x 1.

Much of the work done for the SpMV operation is based off Georgi Evushenko's medium article, [Sparse Matrix-Vector Multiplication with CUDA](https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f), and Pooja Hiranandani's report, [Sparse Matrix Vector Multiplication on GPUs: Implementation and analysis of five algorithms](https://github.com/poojahira/spmv-cuda/blob/master/SpMV_Report.pdf) and corresponding github repository, [spmv-cuda](https://github.com/poojahira/spmv-cuda/blob/master/SpMV_Report.pdf).

Three different SpMV kernels were created when developing this implementation
* [spmvNaive](source/gpu/kernels.cu#226)
* [spmvCSRVector](source/gpu/kernels.cu#235)
* [spmvCSRVectorShared](source/gpu/kernels.cu#260)

While the first, spmvNaive, uses one thread per row in matrix A to perform the dot product, spmvCSRVector and spmvCSRVectorShared use a warp of threads(i.e. 32 threads) per row of matrix A. This allows for a better utilization of resources, since the naive approach can create bottlenecks if it encounters a row that is significantly more dense than others.

The biggest difference between the other two kernels is their use of shared memory; spmvCSRVectorShared uses shared, cached memory, while spmvCSRVector does not. There was no real significant speedup found between these kernels when they were used in lsqr. spmvCSRVector is set to the default in this implementation, but it can easily be switched via the "kern" variable used in [matrixCUDA.cu](source/gpu/matrixCUDA.cu#8).

The run time of lsqr when using each kernel can be seen in the table below (inputs 2500_2500_A_0.mat and 2500_1_b.vec):
|kernel             |calculation time (s)|
|-------------------|--------------------|
|spmvNaive          |198.099             |
|spmvCSRVector      |61.102              |
|spmvCSRVectorShared|60.198              |

### cuSPARSE Transpose
The transpose of a CSR matrix is its compressed sparse column, CSC, counterpart. A kernel for this operation was not explicitly developed for this work, as it was both difficult to design or find a good parallel implementation for it.

Also, as can be seen from the nvprof output, the transpose operation is only called once within the entire lsqr algorithm. It was, therefore, not seen as high priority seeing as it would have little impact on its overall speedup.

Therefore, the existing cusparseCsr2cscEx2 function within the cuSPARSE library was used. This implementation can be found in [matrixCUDA.cu](source/gpu/matrixCUDA.cu#37) More information regarding the cuSPARSE library can be found within the [CUDA toolkit documentation](https://docs.nvidia.com/cuda/cusparse/index.html).

<a id="cuBLAS-DENSE"></a>

## [3.4. cuBLAS-DENSE](source/gpu/vectorCUBLAS.cuh)
The cuBLAS-DENSE implementation is written using both the CUDA and cuBLAS. cuBLAS is a library from NVIDIA that provides "basic linear algebra" operations on a GPU. For this implementation, two inputs of type [VectorCUBLAS](source/gpu/vectorCUBLAS.cuh) are used.

Information regarding cuBLAS how to use it is documented extensively in the [CUDA toolkit documentation](https://docs.nvidia.com/cuda/cublas/index.html), and will therefore, not be further discussed here.

To see how these cuBLAS operations were used for this implementation, please refer to the [VectorCUBLAS source files](source/gpu/vectorCUBLAS.cu)

<a id="cuSPARSE-SPARSE"></a>

## [3.5. cuSPARSE-SPARSE](source/gpu/matrixCUSPARSE.cuh)
The cuSPARSE-SPARSE implementation is written using both CUDA, cuBLAS, and cuSPARSE libraries. cuSPARSE is a library from NVIDIA that provides "a set of basic linear algebra subroutines used for handling sparse matrices". For this implementation, one input of type [MatrixCUSPARSE]() (matrix A) and one input of type [VectorCUBLAS]() (vector b) are used.

This implementation uses all the same operations as the cuBLAS-DENSE implementation, besides the SpMV and matrix transform operations, which are both executed using the cuSPARSE library. Information regarding cuBLAS how to use it is documented extensively in the [CUDA toolkit documentation](https://docs.nvidia.com/cuda/cusparse/index.html), and will therefore, not be further discussed here.

To see how these cuSPARSE operations were used for this implementation, please refer to the [MatrixCUSPARSE source files](source/gpu/matrixCUSPARSE.cu)

<a id="Results"></a>

## 4. Results

<a id="Conclusion"></a>

## 5. Conclusion


