#include "matVec_gpu.cuh"
//#include "../cpu/matVec_cpu.h"
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>   //NULL, printf
#include <stdlib.h>  //srand, rand
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

    unsigned int rows = 2;
    unsigned int columns = 5;
    int array_size = rows * columns;
    int byte_size = sizeof(double) * array_size;
    double *h_in1 = new double[array_size];
    double *h_in2 = new double[array_size];
    double *h_in3 = new double[array_size / 2];
    double *h_out = new double[array_size * array_size];

    for (int i = 0; i < array_size; i++) {
      h_in1[i] = i;
      h_in2[i] = 5 * i;
    }
    for (int i = 0; i < array_size / 2; i++) {
      h_in3[i] = 3 * i;
    }
    Vector_GPU d_i1(rows, columns, h_in1);
    Vector_GPU d_i2(columns, rows, h_in2);
    d_i2 = d_i1.transpose();
    // d_i2.transpose();
    Vector_CPU h_i1 = d_i1.matDeviceToHost();
    Vector_CPU h_i2 = d_i2.matDeviceToHost();
    h_i1.print();
    h_i2.print();
    d_i2 = (d_i1 * d_i2);
    d_i2.printmat();
    h_i2 = d_i2.matDeviceToHost();
    h_i2.print();
    // Vector_GPU d_i2(rows,columns,h_in1);
    // d_i2 = d_i1.transpose();
    // d_i2.printmat();
    /*
    Vector_GPU d_i3(rows,columns/2,h_in3);
    d_i3.printmat();

    Vector_GPU d_out = d_i1 * d_i2;
    d_i2 = d_i2 - d_i1;
    d_i1 = d_i2 + d_i1 + d_i2;
    Vector_GPU copy = d_out;
    d_i3 = d_i1;
    d_i1.printmat();
    cudaDeviceSynchronize();

    Vector_CPU h = d_i3.matDeviceToHost();
    */
    delete h_in1, h_in2, h_in3, h_out;
  } else {
    printf("NO CUDA DEVICE AVAILABLE");
  }
  return 0;
}