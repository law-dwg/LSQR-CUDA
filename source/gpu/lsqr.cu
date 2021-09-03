#include "matVec_gpu.cuh"
//#include "../cpu/matVec_cpu.h"
#include "device_launch_parameters.h"
#include "lsqr_gpu.cuh"
#include <assert.h>
#include <chrono>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> //srand, rand
#include <string.h>
#include <time.h>

int main() {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  checkDevice();
  unsigned int rows1 = 1024; // 167 correct 168 wrong
  unsigned int columns1 = 1;
  int array_size_1 = rows1 * columns1;
  int byte_size_1 = sizeof(double) * array_size_1;
  double *h_in1 = new double[array_size_1];

  for (int i = 0; i < array_size_1; i++) {
    // h_in1[i] = double(rand() % 10 + 1);
    h_in1[i] = double(1);
  }

  Vector_GPU h1(rows1, columns1, h_in1);
  double out = h1.dDnrm2();

  Vector_CPU h2 = h1.matDeviceToHost();
  double out2 = h2.Dnrm2();

  printf("GPU: %f\n", out);
  printf("CPU: %f\n", out2);

  delete h_in1;                         //, matCpu, matGpu;  //, h_in3, h_out;
  cudaError_t err = cudaGetLastError(); // add
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  } // add
  cudaProfilerStop();

  return 0;
};