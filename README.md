# C++ and CUDA implementations of the lsqr algorithm
The following repository is split into two folders, one for the cpu implementation of lsqr, and one for the gpu implementation of lsqr.
___
# CPU Implementation
___
# GPU Implementation
The Kernels used in this implementation are all 2-Dimensional, and can all handle matricies of large various sizes depending on the capabilities of the GPU.

Here, the speed of both naive and optimizied algorithms are analyzed and compared. 

## Naive kernels


## Optimized kernels

### Transpose
The transpose kernel utilizes coalesced memory access via shared memory (block scope).

### Multiply