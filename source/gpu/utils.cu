#include "utils.cuh"
#include <iostream>
const unsigned BLOCK_SIZE_X = 32;
const unsigned BLOCK_SIZE_Y = 32;
const double ONE = 1.0;
const double ZERO = 0.0;
const double NEGONE = -1.0;
const double TOL = 1.e-12;

/** CUDA */
int checkDevice() {
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
  } else {
    printf("NO CUDA DEVICE AVAILABLE");
  }
  return deviceCount;
};

void cudaLastErrCheck() {
  cudaError_t err = cudaGetLastError(); // add
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  }
}

/** cuBLAS */
#define MODE CUBLAS_POINTER_MODE_HOST
cublasStatus_t stat;
cudaError_t cudaStat;
cublasHandle_t handle;
// cudaStream_t stream;

void cublasStart() {
  cublasErrCheck(cublasCreate(&handle));
  cublasErrCheck(cublasSetPointerMode(handle, MODE));
  // cudaStatCreateStream = cudaStreamCreate(&stream);
  // if (cudaStatCreateStream != cudaSuccess) {
  //   printf("CUBLAS create stream failed\n");
  //   // return EXIT_FAILURE;
  // }
  // statSetStream = cublasSetStream(handle, stream);
  // if (statSetStream != CUBLAS_STATUS_SUCCESS) {
  //   printf("CUBLAS set pointer failed\n");
  //   // return EXIT_FAILURE;
  // }
};

void cublasStop() { cublasDestroy(handle); };

/** cuSPARSE */
#define SPMODE CUSPARSE_POINTER_MODE_HOST
cusparseStatus_t spStat;
cusparseHandle_t spHandle;

void cusparseStart() {
  cusparseErrCheck(cusparseCreate(&spHandle));
  cusparseErrCheck(cusparseSetPointerMode(spHandle, SPMODE));
};

void cusparseStop() { cusparseErrCheck(cusparseDestroy(spHandle)); };

cusolverStatus_t solStat;
cusolverSpHandle_t solHandle;
void cusolverStart() { cusolverErrCheck(cusolverSpCreate(&solHandle)); };

void cusolverStop() { cusolverErrCheck(cusolverSpDestroy(solHandle)); };