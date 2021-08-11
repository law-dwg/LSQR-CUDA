#C++ and CUDA implementations of the lsqr algorithm
The following repository is split into two folders, one for the cpu implementation of lsqr, and one for the gpu implementation of lsqr.
___
#CPU Implementation

#GPU Implementation
## Transpose
The transpose kernel utilizes coalesced memory access via shared memory (block scope).