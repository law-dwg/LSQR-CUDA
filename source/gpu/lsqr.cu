#include "matCsr_gpu.cuh"
#include "matVec_gpu.cuh"
//#include "../cpu/matVec_cpu.h"
#include "../cpu/lsqr_cpu.h"
#include "device_launch_parameters.h"
#include "lsqr_gpu.cuh"
#include <assert.h>
#include <chrono>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> //srand, rand
#include <string.h>
#include <time.h>

int main() {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int a_csr_rows, b_csr_rows, a_csr_cols, b_csr_cols;
  a_csr_rows = b_csr_rows = a_csr_cols = b_csr_cols = 3;
  double *a_csr_heap = new double[9]{0, 1, 2, 4, 0, 5, 6, 7, 0};
  double *b_csr_heap = new double[9]{0, 1, 0, 3, 4, 5, 6, 7, 8};
  Matrix_CSR_GPU(a_csr_rows, a_csr_cols, a_csr_heap);
  delete a_csr_heap, b_csr_heap;
  // double *a_heap, *b_heap;
  // int a_rows, b_rows, a_cols, b_cols;
  // a_rows = b_rows = 10;
  // a_cols = 10;
  // b_cols = 1;
  // a_heap = new double[a_rows * a_cols];
  // b_heap = new double[b_rows * b_cols];
  // for (int i = 0; i < (a_rows * a_cols); i++) {
  //   if (i < b_rows) {
  //     a_heap[i] = i;
  //     b_heap[i] = i;
  //   } else {
  //     a_heap[i] = i;
  //   }
  // }
  //
  // writeArrayToFile("A.txt", a_rows * a_cols, a_heap);
  // writeArrayToFile("b.txt", b_rows * b_cols, b_heap);
  // Vector_CPU A_c(a_rows, a_cols, a_heap);
  // Vector_CPU b_c(b_rows, b_cols, b_heap);
  // Matrix_GPU c_c(b_rows, b_cols, b_heap);
  // Matrix_GPU A = A_c;
  // Vector_GPU b = b_c;

  // printf("STARTING LSQR\n");
  // Vector_GPU x_G = lsqr_gpu(A, b);
  // Vector_CPU x_C = lsqr_cpu(A_c, b_c);
  // Vector_CPU x_G_out = x_G.matDeviceToHost();
  // bool ans = compareMat(x_G_out.getMat(), x_G_out.getRows(), x_G_out.getColumns(), x_C.getMat(), x_C.getRows(), x_C.getColumns());
  // x_G_out.print();
  // x_C.print();
  // writeArrayToFile("x_c.txt", x_G_out.getRows() * x_G_out.getColumns(), x_G_out.getMat());
  // writeArrayToFile("x_g.txt", x_C.getRows() * x_C.getColumns(), x_C.getMat());

  // delete a_heap, b_heap;
  cudaError_t err = cudaGetLastError(); // add
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  } // add
  cudaProfilerStop();

  return 0;
};