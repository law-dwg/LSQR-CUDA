#include "matVec_gpu.cuh"
//#include "../cpu/matVec_cpu.h"
#include "device_launch_parameters.h"
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

void checker(double *MC, int rowC, int colC, double *MG, int rowG, int colG) {
  bool same = true;
  double epsilon = 0.001;
  if (rowC != rowG || colC != colG || !same) {
    printf("MATRICIES SIZE  DO NOT MATCH matCPU(%d x %d) != matGPU(%d, %d)\n", rowC, colC, rowG, rowC);
    same = false;
  }
  while (same) {
    for (int i = 0; i < rowC * colC; i++) {
      // printf("MG[%d] = %f, MC[%d] = %f\n", i, MG[i], i, MC[i]);
      // printf("DIFF = %f, %f == %f\n", std::abs(MG[i] - MC[i]), MG[i], MC[i]);
      if (!(std::abs(MG[i] - MC[i]) < epsilon)) {
        printf("MATRICIES SIZE (%d x %d) DO NOT MATCH DISCREPANCY AT INDEX %d; DIFF = %f, %f "
               "== %f\n",
               rowC, colC, i, std::abs(MG[i] - MC[i]), MG[i], MC[i]);
        printf("MG[%d]=%f, MC[%d]=%f\nMG[%d]=%f, MC[%d]=%f\nMG[%d]=%f, "
               "MC[%d]=%f\nMG[%d]=%f, MC[%d]=%f\n",
               i - 1, MG[i - 1], i - 1, MC[i - 1], i, MG[i], i, MC[i], i + 1, MG[i + 1], i + 1, MC[i + 1], i + 2, MG[i + 2], i + 2, MC[i + 2]);
        printf("LAST ELEMENTS MG[%d]=%f, MC[%d]=%f\n", (rowC * colC) - 1, MG[(rowC * colC) - 1], (rowC * colC) - 1, MC[(rowC * colC) - 1]);
        same = false;
        break;
      }
    }
    if (same) {
      printf("MATRICES MATCH FOR (%d x %d)\n", rowC, colC);
      same = false;
    };
  }
}
int main() {
  // Check Cuda Capabale Device
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;

  if (deviceCount > 0) {
    for (device = 0; device < deviceCount; ++device) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      printf("Device %s has compute capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);
      printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
      printf("Clock rate: %d Hz\n", deviceProp.clockRate);
      printf("Total amount of global memory: %d KB\n", deviceProp.totalGlobalMem / 1024);
      printf("Total amount of constant memory: %d KB\n", deviceProp.totalConstMem / 1024);
      printf("Total amount of shared memory per block: %d KB\n", deviceProp.sharedMemPerBlock / 1024);
      printf("Total amount of shared memory per SM: %d KB\n", 64);
      printf("Warp size: %d\n", deviceProp.warpSize);
      printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
      printf("Maximum number of blocks per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock);
      printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
      printf("Maximum number of warps per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / 32);
      printf("Maximum Grid size: (%d,%d,%d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
      printf("Maximum block dimension: (%d,%d,%d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    unsigned int rows1 = 267;
    unsigned int columns1 = 2000;
    int array_size_1 = rows1 * columns1;
    int byte_size_1 = sizeof(double) * array_size_1;
    double *h_in1 = new double[array_size_1];

    srand(time(NULL));
    for (int i = 0; i < array_size_1; i++) {
      // h_in1[i] = double(rand() % 10 + 1);
      h_in1[i] = double(i);
    }
    unsigned int rows2 = 2000;
    unsigned int columns2 = 983;
    int array_size_2 = rows2 * columns2;
    int byte_size_2 = sizeof(double) * array_size_2;
    double *h_in2 = new double[array_size_2];
    for (int i = 0; i < array_size_2; i++) {
      // h_in2[i] = double(rand() %10 + 1);
      h_in2[i] = double(i);
    }
    Vector_GPU d1(rows1, columns1, h_in1);
    Vector_GPU d2(rows2, columns2, h_in2);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    Vector_GPU d3 = d1 * d2;
    cudaEventRecord(stop);
    Vector_CPU hd3 = d3.matDeviceToHost();
    cudaEventSynchronize(stop);
    float milisecs = 0;
    cudaEventElapsedTime(&milisecs, start, stop);
    printf("GPU %f milliseconds elapsed\n", milisecs);
    double *matGpu = new double[hd3.rows * hd3.columns];
    matGpu = &hd3.mat[0];
    Vector_CPU h1(rows1, columns1, h_in1);
    Vector_CPU h2(rows2, columns2, h_in2);
    Vector_CPU h3 = h1 * h2;
    double *matCpu = new double[h3.rows * h3.columns];
    matCpu = &h3.mat[0];
    printf("CHECKING\n");
    checker(matCpu, h3.getRows(), h3.getColumns(), matGpu, d3.getRows(), d3.getColumns());
    delete h_in1, h_in2, matCpu, matGpu; //, h_in3, h_out;

    cudaError_t err = cudaGetLastError(); // add
    if (err != cudaSuccess) {
      std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    } // add
    cudaProfilerStop();
  } else {
    printf("NO CUDA DEVICE AVAILABLE");
  }
  return 0;
}