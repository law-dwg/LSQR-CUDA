#pragma once
#include <limits>

bool compareMat(double *MC, int rowC, int colC, double *MG, int rowG, int colG) {
  bool same = true;
  double epsilon = 0.001;
  if (rowC != rowG || colC != colG || !same) {
    printf("MATRICIES SIZE  DO NOT MATCH matCPU(%d x %d) != matGPU(%d, %d)\n", rowC, colC, rowG, rowC);
    same = false;
  }
  if (same) {
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
  };
  if (same) {
    printf("MATRICES MATCH FOR (%d x %d)\n", rowC, colC);
  };
  return same;
};
bool compareVal(double *VC, double *VG) {
  typedef std::numeric_limits<double> dbl;
  bool same = false;
  printf("GPU: %20f\n", *VG);
  printf("CPU: %20f\n", *VC);
  std::cout.precision(dbl::max_digits10);
  std::cout << *VC << std::endl;
  std::cout << *VG << std::endl;
  std::cout << std::abs(*VC - *VG) << std::endl;
  if (std::abs(*VC - *VG) < 1e-15) {
    printf("THEY ARE SAME\n");
    same = true;
  } else {
    printf("THEY ARE NOT SAME\n");
  }
  return same;
}

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
