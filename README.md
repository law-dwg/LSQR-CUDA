# LSQR-CUDA
## Overview
LSQR-CUDA was written by Lawrence Ayers under the supervision of Stefan Guthe of the [GRIS](https://www.informatik.tu-darmstadt.de/gris/startseite_1/team/index.de.jsp) institute at the Technische Universität Darmstadt.

The goal of this work was to accelerate the computation time of the well-known [LSQR](https://web.stanford.edu/group/SOL/software/lsqr/) algorithm using a CUDA capable GPGPU.


## Requirements
LSQR-CUDA has the following requirements:
* CUDA Capable GPGPU
* CUDA v11 or higher 
* g++ v11 or higher
* *nix system 

<details open>
<summary><b>Table of Contents</b></summary>
<!-- MarkdownTOC -->

1.  [General](#General)
1.  [Background](#Background)
1.  [Methods](#Methods)
    1.  [CPU](#CPU)
    1.  [GPU](#CPU)
1.  [Results](#Results)
    1.   [Speedup](#Speedup)
    1.   [Accuracy](#Accuracy)
1.  [Conclusion](#Conclusion)
<!-- /MarkdownTOC -->
</details>

<a id="General"></a>
## 1. General
The purpose of this work was to implement the LSQR algorithm on a CUDA capabale GPU in order to analyze a potential speedup in calculation time in comparison to a standard sequential CPU implementation. When run in CUDA, many matrix operations (e.g. multiplication, euclidean norm, addition, subtraction, etc.) can be run in parallel  

This work has both sequential and parallel implementations that works for both sparse and dense inputs.
___
<a id="Background"></a>
## 2. Background
___
<a id="Methods"></a>
## 3. Methods
___
<a id="Results"></a>
## 4. Results
___
<a id="Conclusion"></a>
## 5. Conclusion
___

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