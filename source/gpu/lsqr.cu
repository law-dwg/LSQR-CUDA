#include "matVec_gpu.cuh"
//#include "../cpu/matVec_cpu.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> //srand, rand
#include <string.h>
#include <time.h>

#include <iostream>
#include <sstream>

#include "device_launch_parameters.h"
void __global__ print() {}

int main() {
  // Check Cuda Capabale Device
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;

  if (deviceCount > 0) {
    for (device = 0; device < deviceCount; ++device) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      printf("Device %s has compute capability %d.%d.\n", deviceProp.name, deviceProp.major,
             deviceProp.minor);
      printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
      printf("Clock rate: %d Hz\n", deviceProp.clockRate);
      printf("Total amount of global memory: %d KB\n", deviceProp.totalGlobalMem / 1024);
      printf("Total amount of constant memory: %d KB\n", deviceProp.totalConstMem / 1024);
      printf("Total amount of shared memory per block: %d KB\n",
             deviceProp.sharedMemPerBlock / 1024);
      printf("Total amount of shared memory per SM: %d KB\n", 64);
      printf("Warp size: %d\n", deviceProp.warpSize);
      printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
      printf("Maximum number of blocks per multiprocessor: %d\n",
             deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock);
      printf("Maximum number of threads per multiprocessor: %d\n",
             deviceProp.maxThreadsPerMultiProcessor);
      printf("Maximum number of warps per multiprocessor: %d\n",
             deviceProp.maxThreadsPerMultiProcessor / 32);
      printf("Maximum Grid size: (%d,%d,%d)\n", deviceProp.maxGridSize[0],
             deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
      printf("Maximum block dimension: (%d,%d,%d)\n", deviceProp.maxThreadsDim[0],
             deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    }
    for (int e = 11200; e < (11201); e += 100) {
      unsigned int rows = e;
      unsigned int columns = e;
      int array_size = rows * columns;
      int byte_size = sizeof(double) * array_size;
      double *h_in1 = new double[array_size];
      // double *h_in2 = new double[array_size];
      // double *h_in3 = new double[array_size / 2];
      // double *h_out = new double[array_size * array_size];

      for (int i = 0; i < array_size; i++) {
        h_in1[i] = i;
        // h_in2[i] = 5 * i;
      }
      // for (int i = 0; i < array_size / 2; i++) {
      //   h_in3[i] = 3 * i;
      // }
      Vector_GPU d_i1(rows, columns, h_in1);
      // Vector_GPU d_i2(columns, rows, h_in2);
      cudaDeviceSynchronize();
      d_i1 = d_i1.transpose();
      Vector_CPU hd_i1 = d_i1.matDeviceToHost();
      Vector_CPU comparator(rows, columns, h_in1);
      comparator = comparator.transpose();
      double *matGpu = new double[hd_i1.rows * hd_i1.columns];
      double *matCpu = new double[hd_i1.rows * hd_i1.columns];
      matGpu = &hd_i1.mat[0];
      matCpu = &comparator.mat[0];
      bool same = true;
      double epsilon = 0.001;
      do {
        for (int i = 0; i < rows * columns; i++) {
          // printf("matGpu[%d] = %f, matCpu[%d] = %f\n", i, matGpu[i], i, matCpu[i]);
          // printf("DIFF = %f, %f == %f\n", std::abs(matGpu[i] - matCpu[i]), matGpu[i], matCpu[i]);
          if (!(std::abs(matGpu[i] - matCpu[i]) < epsilon)) {
            printf("MATRICIES SIZE (%d x %d) DO NOT MATCH DISCREPANCY AT INDEX %d; DIFF = %f, %f "
                   "== %f\n",
                   rows, columns, i, std::abs(matGpu[i] - matCpu[i]), matGpu[i], matCpu[i]);
            printf("matGpu[%d]=%f, matCpu[%d]=%f\nmatGpu[%d]=%f, matCpu[%d]=%f\nmatGpu[%d]=%f, "
                   "matCpu[%d]=%f\nmatGpu[%d]=%f, matCpu[%d]=%f\n",
                   i - 1, matGpu[i - 1], i - 1, matCpu[i - 1], i, matGpu[i], i, matCpu[i], i + 1,
                   matGpu[i + 1], i + 1, matCpu[i + 1], i + 2, matGpu[i + 2], i + 2, matCpu[i + 2]);
            printf("LAST ELEMENTS matGpu[%d]=%f, matCpu[%d]=%f\n", (rows * columns) - 1,
                   matGpu[(rows * columns) - 1], (rows * columns) - 1,
                   matCpu[(rows * columns) - 1]);
            same = false;
            break;
          }
        }
        if (same) {
          printf("MATRICES MATCH FOR (%d x %d)\n", rows, columns);
          same = false;
        };
      } while (same);
      // comparator.~Vector_CPU();
      // hd_i1.~Vector_CPU();
      delete h_in1, matCpu, matGpu; //, h_in3, h_out;
    }
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