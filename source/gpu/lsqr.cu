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

  unsigned int rows1 = 7;
  unsigned int columns1 = 7;
  int array_size_1 = rows1 * columns1;
  int byte_size_1 = sizeof(double) * array_size_1;
  double *h_in1 = new double[array_size_1];

  srand(time(NULL));
  for (int i = 0; i < array_size_1; i++) {
    // h_in1[i] = double(rand() % 10 + 1);
    h_in1[i] = double(i);
  }
  unsigned int rows2 = 7;
  unsigned int columns2 = 3;
  int array_size_2 = rows2 * columns2;
  int byte_size_2 = sizeof(double) * array_size_2;
  double *h_in2 = new double[array_size_2];
  for (int i = 0; i < array_size_2; i++) {
    // h_in2[i] = double(rand() %10 + 1);
    h_in2[i] = double(i);
  }
  Vector_GPU h1(rows1, columns1, h_in1);
  Vector_GPU h2(rows2, columns2, h_in2);
  Vector_GPU h3 = h2;
  // hc3.print();
  // double *matCpu = new double[h3.rows * h3.columns];
  // matCpu = &h3.mat[0];
  // Vector_GPU d1 = h1;
  // Vector_GPU d2 = h2;
  // cudaEventRecord(start);
  // Vector_GPU d3 = d1 * d2;
  // cudaEventRecord(stop);
  // Vector_CPU hd3 = d3.matDeviceToHost();
  // cudaEventSynchronize(stop);
  // float milisecs = 0;
  // cudaEventElapsedTime(&milisecs, start, stop);
  // printf("GPU %f[ms] elapsed\n", milisecs);
  // double *matGpu = new double[hd3.rows * hd3.columns];
  // matGpu = &hd3.mat[0];
  // compareMat(matCpu, h3.getRows(), h3.getColumns(), matGpu, d3.getRows(), d3.getColumns());
  delete h_in1, h_in2;                  //, matCpu, matGpu;  //, h_in3, h_out;
  cudaError_t err = cudaGetLastError(); // add
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  } // add
  cudaProfilerStop();

  return 0;
};