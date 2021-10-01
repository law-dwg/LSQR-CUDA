#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_
#include <cuda.h>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val == 0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#endif
#endif

static __inline__ __device__ double atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> //srand, rand
#include <string.h>
#include <time.h>

#include <iostream>
#include <sstream>

#include "../cpu/matVec_cpu.hpp"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "matVec_gpu.cuh"

cudaError_t cudaStat;
cublasHandle_t handle;
cudaStream_t stream;
cublasStatus_t stat1 = cublasCreate(&handle);
cudaError_t cudaStat1 = cudaStreamCreate(&stream);
cublasStatus_t stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
cublasStatus_t stat2 = cublasSetStream(handle, stream);
void cublasReset() {
  stat1 = cublasCreate(&handle);
  cudaStat1 = cudaStreamCreate(&stream);
  stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  stat2 = cublasSetStream(handle, stream);
}

#define gpuErrchk(ans)                                                                                                                               \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

using namespace cooperative_groups;
// OUR TILE SIZE SHOULD MATCH THAT OF OUR BLOCK
#define TILE_DIM_X 16
#define TILE_DIM_Y 16
// nvcc -arch=sm_37 --std=c++17
// gridDim.x - # of blocks in a grid, in x
// gridDim.y - # of blocks in a grid, in y
// blockDim.x - # of threads in a block, in x
// blockDim.y - # of threads in a block, in y

// CUDA kernels
void __global__ multiplyNaive(double *in1, unsigned int *rows1, unsigned int *cols1, double *in2, unsigned int *rows2, unsigned int *cols2,
                              double *output) {
  const unsigned int x = blockIdx.y * blockDim.y + threadIdx.y; // row
  const unsigned int y = blockIdx.x * blockDim.x + threadIdx.x; // column
  double sum = 0;
  if ((x < *rows1) && (y < *cols2)) {
    for (int i = 0; i < *cols1; ++i) {
      sum += in1[x * *cols1 + i] * in2[i * *cols2 + y];
      // printf("sum = %f += in1[%d] * in2[%d] = %f * %f\n",sum,r * *cols1 + i,
      // i * *cols2 + c,in1[r * *cols1 + i],in2[i * *cols2 + c]);
    }
    output[x * *cols2 + y] = sum;
    // printf("output[%d] = %f\n",r * *cols2 + c, output[r * *cols2 + c]);
  }
}

void __global__ scale(double *input, double *scalar, double *output, unsigned *r, unsigned *c, bool inverse) {
  const unsigned int bid = blockIdx.x                               // 1D
                           + blockIdx.y * gridDim.x                 // 2D
                           + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned int threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                       * blockDim.z;                // 3D
  const unsigned int tid = threadIdx.x                              // 1D
                           + threadIdx.y * blockDim.x               // 2D
                           + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned int gid = bid * threadsPerBlock + tid;
  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d,
  // value=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,input[gid]);
  if (gid < *r * *c) {
    if (inverse) {
      output[gid] = input[gid] * (1.0 / *scalar);
      // printf("%f = %f / %f\n", output[gid], input[gid], *scalar);
    } else {
      output[gid] = input[gid] * *scalar;
      // printf("out[%d] = %f = %f * %f\n", gid, output[gid], input[gid], *scalar);
    }
  }
}

void __global__ print(double *input, unsigned *r, unsigned *c) {
  const unsigned int bid = blockIdx.x                               // 1D
                           + blockIdx.y * gridDim.x                 // 2D
                           + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned int threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                       * blockDim.z;                // 3D
  const unsigned int tid = threadIdx.x                              // 1D
                           + threadIdx.y * blockDim.x               // 2D
                           + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned int gid = bid * threadsPerBlock + tid;
  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d,
  // value=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,input[gid]);
  __syncthreads();
  if (gid < *r * *c) {
    printf("%f\n", input[gid]);
  }
}

void __global__ assignment(double *in1, double *in2) {
  const unsigned int bid = blockIdx.x                               // 1D
                           + blockIdx.y * gridDim.x                 // 2D
                           + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned int threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                       * blockDim.z;                // 3D
  const unsigned int tid = threadIdx.x                              // 1D
                           + threadIdx.y * blockDim.x               // 2D
                           + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned int gid = bid * threadsPerBlock + tid;
  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, in1=%f, in2=%f\n", threadIdx.x,
  //        threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, bid, gid, in1[gid],
  //        in2[gid]);
  in1[gid] = in2[gid];
}

void __global__ subtract(double *in1, double *in2, unsigned *rows, unsigned *cols, double *output) {
  const unsigned int bid = blockIdx.x                               // 1D
                           + blockIdx.y * gridDim.x                 // 2D
                           + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned int threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                       * blockDim.z;                // 3D
  const unsigned int tid = threadIdx.x                              // 1D
                           + threadIdx.y * blockDim.x               // 2D
                           + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned int gid = bid * threadsPerBlock + tid;
  const unsigned int r = blockIdx.y * blockDim.y + threadIdx.y; // the row of M1
  const unsigned int c = blockIdx.x * blockDim.x + threadIdx.x; // the col of M2
  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, %f *
  // %f\n",threadIdx.x,threadIdx.y,threadIdx.z,
  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
  if (gid < *rows * *cols)
    output[gid] = in1[gid] - in2[gid];
  // printf("%f = %f - %f\n", output[gid], in1[gid], in2[gid]);
}

void __global__ add(double *in1, double *in2, unsigned *rows, unsigned *cols, double *out) {
  const unsigned int bid = blockIdx.x                               // 1D
                           + blockIdx.y * gridDim.x                 // 2D
                           + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned int threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                       * blockDim.z;                // 3D
  const unsigned int tid = threadIdx.x                              // 1D
                           + threadIdx.y * blockDim.x               // 2D
                           + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned int gid = bid * threadsPerBlock + tid;
  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, in1=%f,
  // in2=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
  if (gid < *rows * *cols)
    out[gid] = in1[gid] + in2[gid];
  // printf("%f = %f + %f\n", out[gid], in1[gid], in2[gid]);
}

__device__ double reduce_sum(thread_group g, double *temp, double val) {
  int lane = g.thread_rank();

  // Each iteration halves the number of active threads
  // Each thread adds its partial sum[i] to sum[lane+i]
  for (int i = g.size() / 2; i > 0; i /= 2) {
    temp[lane] = (val);
    g.sync(); // wait for all threads to store
    if (lane < i)
      val += (temp[lane + i]);
    g.sync(); // wait for all threads to load
  }
  return val; // note: only thread 0 will return full sum
}

__device__ double thread_sum(double *input, int n) {
  double sum = 0.0;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int gid2 = gid + blockDim.x * gridDim.x;
  // printf("i=%d; i1=%d\n", gid, gid2);
  // if (n < TILE_DIM_X) {
  //  n = TILE_DIM_X;
  //}
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n / 4; i += blockDim.x * gridDim.x) {
    double4 in = ((double4 *)input)[i];
    // printf("in.x=%f + in.y=%f + in.z=%f + in.w=%f\n", in.x, in.y, in.z, in.w * in.w);
    sum += in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w;
    // printf("sum = %f\n", sum);
  }
  return sum;
}

__device__ void warpReduce(volatile double *sum, int thread) {
  // printf("COMBINEDCALLED threadIdx.x = %d\n", thread);
  // if (blockD >= 64)
  sum[thread] += sum[thread + 32];
  // if (blockD >= 32)
  sum[thread] += sum[thread + 16];
  // if (blockD >= 16)
  sum[thread] += sum[thread + 8];
  // if (blockD >= 8)
  sum[thread] += sum[thread + 4];
  // if (blockD >= 4)
  sum[thread] += sum[thread + 2];
  // if (blockD >= 2)
  sum[thread] += sum[thread + 1];
}

void __global__ dnrm2Coop(double *in1, unsigned int *r, unsigned int *c, double *out) {
  // using cooperative groups as described here: https://developer.nvidia.com/blog/cooperative-groups/
  double my_sum = thread_sum(in1, *r * *c);
  __shared__ double temp[TILE_DIM_X];
  auto g = this_thread_block();
  double block_sum = reduce_sum(g, temp, my_sum);
  if (g.thread_rank() == 0) {
    atomicAdd(out, block_sum);
  }
}

void __global__ dnrm2(double *in1, unsigned int *r, unsigned int *c, double *out) {
  __shared__ double sum[TILE_DIM_X];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  int x2 = (blockIdx.x * (blockDim.x * 2) + threadIdx.x) + blockDim.x;
  // load into shared
  // sum[threadIdx.x] = in1[x] * in1[x] + in1[x2] * in1[x2];

  if (x < (*r * *c) && x2 < (*r * *c)) {
    sum[threadIdx.x] = in1[x] * in1[x] + in1[x2] * in1[x2];
    // printf("INIT: block(%d, %d) thread(%d, %d) sum[%d] =  in1[%d] + in2[%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.x,
    // x, x2, sum[threadIdx.x]);
  } else if ((x < (*r * *c)) && ((*r * *c) <= x2)) {
    sum[threadIdx.x] = in1[x] * in1[x];
    // printf("INIT: block(%d, %d) thread(%d, %d) sum[%d] =  in1[%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.x, x,
    // sum[threadIdx.x]);
  } else {
    // printf("BLOCKED: block(%d, %d) thread(%d, %d) gid = %d  x = %d x2 = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, gid, x, x2);
    sum[threadIdx.x] = 0.0;
  }
  __syncthreads();
  // printf("block(%d, %d) thread(%d, %d) PS[%d] = v[%d]+v[%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.x, x, x2,
  //       sum[threadIdx.x]);
  int s_stop = 0;
  // if (blockDim.x / 2 > 32) {
  //  s_stop = 32;
  //}
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sum[threadIdx.x] += sum[threadIdx.x + s];
      sum[threadIdx.x + s] = 0;
      // printf("gid+s = %d block(%d, %d) thread(%d, %d) s=%d PS[%d] += PS[%d] =%f\n", gid + s, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, s,
      //       threadIdx.x, threadIdx.x + s, sum[threadIdx.x + s]);
    }
    __syncthreads();
  }
  // if (threadIdx.x < 32 && s_stop != 0) {
  //   warpReduce(sum, threadIdx.x);
  // }
  // __syncthreads();
  if (threadIdx.x == 0) {
    // printf("sum[0]=%f\n",sum[0]);
    atomicAdd(out, sum[0]);
  }
}

void __global__ maxVal(double *in1, unsigned int *r, unsigned int *c, double *out) {
  __shared__ double maxV[TILE_DIM_X];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  int x2 = (blockIdx.x * (blockDim.x * 2) + threadIdx.x) + blockDim.x;
  // load into shared
  // sum[threadIdx.x] = in1[x] * in1[x] + in1[x2] * in1[x2];

  if (x < (*r * *c) && x2 < (*r * *c)) {
    maxV[threadIdx.x] = fmax(in1[x], in1[x2]);
    // printf("INIT: block(%d, %d) thread(%d, %d) sum[%d] =  in1[%d] + in2[%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.x,
    // x, x2, sum[threadIdx.x]);
  } else if ((x < (*r * *c)) && ((*r * *c) <= x2)) {
    maxV[threadIdx.x] = in1[x];
    // printf("INIT: block(%d, %d) thread(%d, %d) sum[%d] =  in1[%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.x, x,
    // sum[threadIdx.x]);
  } else {
    // printf("BLOCKED: block(%d, %d) thread(%d, %d) gid = %d  x = %d x2 = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, gid, x, x2);
    maxV[threadIdx.x] = 0.0;
  }
  __syncthreads();
  // printf("block(%d, %d) thread(%d, %d) PS[%d] = v[%d]+v[%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.x, x, x2,
  //       sum[threadIdx.x]);
  int s_stop = TILE_DIM_X / 2;
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      maxV[threadIdx.x] = fmax(std::abs(maxV[threadIdx.x]), maxV[threadIdx.x + s]);
      // printf("gid+s = %d block(%d, %d) thread(%d, %d) s=%d PS[%d] += PS[%d] =%f\n", gid + s, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, s,
      //         threadIdx.x, threadIdx.x + s, maxV[threadIdx.x + s]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // printf("%f\n", maxV[0]);
    atomicMax(out, std::abs(maxV[0]));
  }
  // if (lastBlock(lastBlockCounter)) {
  //   shArr[thIdx] = thIdx<gridSize ? gOut[thIdx] : 0;
  //   __syncthreads();
}

// BLOCK SWEEPS ACROSS TILE (TILE SIZE > BLOCK SIZE)
// source: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/
void __global__ transposeTiled(double *in1, double *output, unsigned int *rows, unsigned int *cols) {
  __shared__ double A[(TILE_DIM_X)][TILE_DIM_Y + 1]; // Add +1 to prevent bank-conflicts

  int x = blockIdx.x * TILE_DIM_X + threadIdx.x; // col
  int y = blockIdx.y * TILE_DIM_Y + threadIdx.y; // row

  // Load the matrix into shared memory
  for (int i = 0; i < TILE_DIM_Y; i += blockDim.y) {
    if ((x < *cols) && (y < *rows)) {
      // A[(row + i) * height + col] = in1[(row + i) * width + col];
      A[threadIdx.y + i][threadIdx.x] = in1[(y + i) * *cols + x];
      // printf(
      //     "block(%d, %d), thread(%d,% d), row = %d, col = %d, ,i=%d, A[%d][%d] "
      //     "= in1[%d] = %f\n",
      //     blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, y, x, i, threadIdx.y + i,
      //     threadIdx.x, (y + i) * *cols + x, in1[(y + i) * *cols + x]);
    }
  };

  __syncthreads();
  x = blockIdx.y * TILE_DIM_X + threadIdx.x; // x-dimension col
  y = blockIdx.x * TILE_DIM_Y + threadIdx.y; // y-dimension row

  for (int i = 0; i < TILE_DIM_Y; i += blockDim.y) {
    // printf("block(%d, %d), thread(%d, %d), i=%d, A[%d][%d] = %f\n",
    //       blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i,
    //       threadIdx.y+i,threadIdx.x, A[threadIdx.y+i][threadIdx.x]);
    if ((y + i < *cols) && (x < *rows)) {
      // output[col * width + (row+i)] = A[(row + i) * width + col];
      output[(y + i) * *rows + x] = A[threadIdx.x][threadIdx.y + i];
      // printf(
      //     "block(%d, %d), thread(%d, %d), row = %d, col = %d, i=%d, output[%d] "
      //     "= A[%d][%d] = %f\n",
      //     blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, y, x, i, (y + i) * *rows + x,
      //     threadIdx.x, threadIdx.y + i, A[threadIdx.x][threadIdx.y + i]);
    }
  }
}

// BLOCK AND TILE SWEEP TOGETHER (BLOCK_SIZE = TILE_SIZE)
void __global__ multiplyTiled(double *in1, unsigned int *rows1, unsigned int *cols1, double *in2, unsigned int *rows2, unsigned int *cols2,
                              double *output) {

  __shared__ double A[TILE_DIM_X][TILE_DIM_Y + 1], B[TILE_DIM_X][TILE_DIM_Y + 1];

  int y = blockIdx.y * blockDim.y + threadIdx.y; // row
  int x = blockIdx.x * blockDim.x + threadIdx.x; // col
  double sum = 0;                                // sum in block

  for (int i = 0; i < *cols1; i += blockDim.x) {
    int id1, id2;
    if (i + threadIdx.x < *cols1 && y < *rows1) {
      id1 = y * *cols1 + i + threadIdx.x;
      A[threadIdx.x][threadIdx.y] = in1[id1];
    } else {
      A[threadIdx.x][threadIdx.y] = 0.0;
    }
    if (i + threadIdx.y < *rows2 && x < *cols2) {
      id2 = (i * *cols2 + threadIdx.y * *cols2) + x;
      B[threadIdx.x][threadIdx.y] = in2[id2];
    } else {
      B[threadIdx.x][threadIdx.y] = 0.0;
    }

    __syncthreads();
    for (int j = 0; j < blockDim.x; j++) {
      // if (x + j < *cols2 && y + j < *rows1) {
      int Ax, Ay, Bx, By;
      Ax = j;
      Ay = threadIdx.y;
      Bx = threadIdx.x;
      By = j;
      // if (blockIdx.x == 1 && blockIdx.y == 2 && threadIdx.x == 0 && threadIdx.y == 0) {
      //   printf("OUT block(%d, %d), thread(%d,% d), y = %d, x = %d, i=%d, j=%d, A[%d][%d]=%f * "
      //          "B[%d][%d]=%f\n",
      //          blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, x, y, i, j, Ax, Ay, A[Ax][Ay], Bx, By, B[Bx][By]);
      // }
      sum += A[Ax][Ay] * B[Bx][By];
      //}
    };
    __syncthreads();
  };
  if (x < *cols2 && y < *rows1) {
    output[y * *cols2 + x] = sum;
  }
}

// Operator overloads

// Vector_GPU Vector_GPU::operator*(Vector_GPU &v) {
//   cublasLtMatmul(cublasLtHandle_t               lightHandle,
//     cublasLtMatmulDesc_t           computeDesc,
//     const void                    *alpha,
//     const void                    *A,
//     cublasLtMatrixLayout_t         Adesc,
//     const void                    *B,
//     cublasLtMatrixLayout_t         Bdesc,
//     const void                    *beta,
//     const void                    *C,
//     cublasLtMatrixLayout_t         Cdesc,
//     void                          *D,
//     cublasLtMatrixLayout_t         Ddesc,
//     const cublasLtMatmulAlgo_t    *algo,
//     void                          *workspace,
//     size_t                         workspaceSizeInBytes,
//     cudaStream_t                   stream);
// }

Vector_GPU Vector_GPU::operator*(double h_i) {
  // printf("scale %f\n", h_i);
  Vector_GPU out(this->h_rows, this->h_columns);
  unsigned int blocksX = (this->h_rows / TILE_DIM_X) + 1;
  unsigned int blocksY = (this->h_columns / TILE_DIM_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(TILE_DIM_X, TILE_DIM_Y, 1);
  double *d_i;
  cudaMalloc((void **)&d_i, sizeof(double));
  cudaMemcpy(d_i, &h_i, sizeof(double), cudaMemcpyHostToDevice);

  scale<<<grid, block>>>(this->d_mat, d_i, out.d_mat, this->d_rows, this->d_columns, false);
  // cudaDeviceSynchronize();

  return out;
}

Vector_GPU Vector_GPU::operator-(const Vector_GPU &v) {
  // printf("SUBTRACT CALLED\n");
  Vector_GPU out(this->h_rows, this->h_columns);
  unsigned int blocksX = (this->h_rows / TILE_DIM_X) + 1;
  unsigned int blocksY = (this->h_columns / TILE_DIM_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(TILE_DIM_X, TILE_DIM_Y, 1);
  if (this->h_rows == v.h_rows && this->h_columns == v.h_columns) {
    subtract<<<grid, block>>>(this->d_mat, v.d_mat, this->d_rows, this->d_columns, out.d_mat);
    // cudaDeviceSynchronize();
  } else {
    printf("SUBTRACT: ARRAYS ARE NOT THE SAME SIZE, canot perform operation %d!=%d\n", this->h_rows, v.h_rows);
    // assert(h_rows == v.h_rows);
  }
  return out;
}

void Vector_GPU::operator=(Vector_CPU &v) { // Copy assignment Vector_CPU -> this Vector_GPU
  printf("ASSIGNMENT #2 called\n");
  cudaFree(d_mat);
  h_rows = v.getRows();
  h_columns = v.getColumns();
  cudaMalloc((void **)&d_mat, sizeof(double) * v.rows * v.columns);
  cudaMemcpy(d_rows, &h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_columns, &h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_mat, &v.mat[0], sizeof(double) * h_rows * h_columns, cudaMemcpyHostToDevice);
}

Vector_GPU Vector_GPU::operator+(const Vector_GPU &v) {
  Vector_GPU out(this->h_rows, this->h_columns);
  unsigned int blocksX = (this->h_rows / TILE_DIM_X) + 1;
  unsigned int blocksY = (this->h_columns / TILE_DIM_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(TILE_DIM_X, TILE_DIM_Y, 1);
  if (this->h_rows == v.h_rows && this->h_columns == v.h_columns) {
    add<<<grid, block>>>(this->d_mat, v.d_mat, this->d_rows, this->d_columns, out.d_mat);
    // cudaDeviceSynchronize();
  } else {
    printf("ADDITION: ARRAYS ARE NOT THE SAME SIZE, canot perform operation %d!=%d\n", this->h_rows, v.h_rows);
    // assert(this->h_rows == v.h_rows);
  }
  return out;
}

void Vector_GPU::printmat() {
  unsigned int blocksX = (this->h_rows / TILE_DIM_X) + 1;
  unsigned int blocksY = (this->h_columns / TILE_DIM_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(TILE_DIM_X, TILE_DIM_Y, 1);

  print<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns);
  // gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
}

Vector_CPU Vector_GPU::matDeviceToHost() {
  double *out = new double[this->h_columns * this->h_rows]; // heap to prevent a stack overflow
  unsigned int rows;
  unsigned int cols;
  cudaMemcpy(out, this->d_mat, sizeof(double) * this->h_columns * this->h_rows, cudaMemcpyDeviceToHost);
  // cudaMemcpy(out, this, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&rows, this->d_rows, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&cols, this->d_columns, sizeof(unsigned int), cudaMemcpyDeviceToHost);
  // std::cout << "d_rows=" << rows << "=h_rows=" << this->h_rows << std::endl;
  // std::cout << "d_columns=" << cols << "=h_columns=" << this->h_columns << std::endl;
  if (rows != this->h_rows || cols != this->h_columns) {
    printf("INCONSISTENT ROWS AND COLS BETWEEN HOST AND DEVICE\n");
  }
  Vector_CPU v_cpu(this->h_rows, this->h_columns, out);
  return v_cpu;
}

Vector_GPU Vector_GPU::transpose() {
  Vector_GPU out(this->h_columns, this->h_rows);

  dim3 numOfThreadsInBlock(TILE_DIM_X, TILE_DIM_Y, 1);
  unsigned int blocksX = (this->h_rows / TILE_DIM_X) + 1;
  unsigned int blocksY = (this->h_columns / TILE_DIM_Y) + 1;
  dim3 numOfBlocksInGrid(blocksX, blocksY, 1);
  // printf("threadsinblock(%d x %d)=%d, blocksingrid(%d, %d)=%d\n", numOfThreadsInBlock.x, numOfThreadsInBlock.y,
  //       numOfThreadsInBlock.x * numOfThreadsInBlock.y, numOfBlocksInGrid.x, numOfBlocksInGrid.y, numOfBlocksInGrid.x * numOfBlocksInGrid.y);
  transposeTiled<<<numOfBlocksInGrid, numOfThreadsInBlock>>>(this->d_mat, out.d_mat, this->d_rows, this->d_columns);
  // cudaDeviceSynchronize();
  return out;
};

double Vector_GPU::Dnrm2Leg() {
  dim3 threads(TILE_DIM_X, 1);
  int blockX = ((this->h_rows * this->h_columns + TILE_DIM_X - 1) / TILE_DIM_X);
  dim3 blocks(blockX, 1);
  double *d_out, *d_max, *tempMat;
  double zero = 0.0;
  double h_max;
  double h_out;
  gpuErrchk(cudaMalloc(&d_out, sizeof(double)));
  gpuErrchk(cudaMalloc(&d_max, sizeof(double)));
  gpuErrchk(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_max, &zero, sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMalloc(&tempMat, sizeof(double) * this->h_rows * this->h_columns));

  // printf("dnrm2 threads(%d x %d)=%d, blocks(%d, %d)=%d\n", threads.x, threads.y, threads.x * threads.y, blocks.x, blocks.y, blocks.x * blocks.y);
  // unsigned s_mem = sizeof(double) * TILE_DIM_X;

  // ERROR HERE
  maxVal<<<blocks, threads>>>(this->d_mat, this->d_rows, this->d_columns, d_max);
  // gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  // unsigned s_mem = sizeof(double) * TILE_DIM_X;
  scale<<<blocks, threads>>>(this->d_mat, d_max, tempMat, this->d_rows, this->d_columns, true);
  gpuErrchk(cudaDeviceSynchronize());
  dnrm2<<<blocks, threads>>>(tempMat, this->d_rows, this->d_columns, d_out);
  gpuErrchk(cudaDeviceSynchronize());
  cudaFree(tempMat);
  gpuErrchk(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost));

  assert(!(h_out != h_out));
  assert(h_out > 0);
  return std::abs(h_max) * sqrt(h_out);
};

double Vector_GPU::Dnrm2() {
  // int blockX = ((this->h_rows * this->h_columns + TILE_DIM_X - 1) / TILE_DIM_X);
  // dim3 blocks(blockX, 1);
  double h_out;
  int *version;
  double *d_out;
  double zero = 0.0;
  // double c_d_mat[h_rows*h_columns];
  int incre = 1;
  gpuErrchk(cudaMalloc(&d_out, sizeof(double)));
  // gpuErrchk(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  // gpuErrchk(cudaMemcpy(&c_d_mat,d_mat, sizeof(double)*h_rows*h_columns,cudaMemcpyDeviceToHost));

  // for (int i = 0; i<100;++i){
  //  printf("%f\n",c_d_mat[i]);
  //}
  // gpuErrchk(cudaMalloc(&c_d_mat, sizeof(double)*h_rows*h_columns));
  // gpuErrchk(cudaMemcpy(&c_d_mat, d_mat, sizeof(double) * h_columns * h_rows, cudaMemcpyDeviceToDevice));
  int size = (this->h_rows * this->h_columns);

  // stat = cublasSetVector(size, sizeof(double), (void **)&c_d_mat, incre, d_d_mat, incre);
  // if (stat != CUBLAS_STATUS_SUCCESS) {
  //   printf ("CUBLAS Version failed\n");
  //   return EXIT_FAILURE;
  // }

  cublasStatus_t stat3 = cublasDnrm2(handle, size, this->d_mat, incre, d_out);
  cudaDeviceSynchronize();
  if (stat3 != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS DNRM2 failed\n");
    return EXIT_FAILURE;
  }

  // stat = cublasDestroy(handle);
  // if (stat != CUBLAS_STATUS_SUCCESS) {
  //   printf ("CUBLAS destroy failed\n");
  //   return EXIT_FAILURE;
  // }
  // cublasDestroy(handle);
  gpuErrchk(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  assert(!(h_out != h_out));
  return h_out;
};

// Vector_GPU Vector_GPU::operator*(Vector_GPU &v) {
Vector_GPU Vector_GPU::multNai(Vector_GPU &v) {
  // printf("MATMULT\n");
  Vector_GPU out(this->h_rows, v.h_columns);
  dim3 threads(TILE_DIM_X, TILE_DIM_Y, 1);
  unsigned int blocksX = (out.h_columns - 1) / threads.x + 1;
  unsigned int blocksY = (out.h_rows - 1) / threads.y + 1;
  dim3 blocks(blocksX, blocksY, 1);
  // if (out.h_columns % TILE_DIM_X > 0 || blocksX == 0) {
  //   blocksX += 2;
  // };
  // if (out.h_rows % TILE_DIM_Y > 0 || blocksY == 0) {
  //   blocksY += 2;
  // };

  // printf("threadsinblock(%d x %d)=%d, blocksingrid(%d, %d)=%d\n", numOfThreadsInBlock.x, numOfThreadsInBlock.y,
  //       numOfThreadsInBlock.x * numOfThreadsInBlock.y, numOfBlocksInGrid.x, numOfBlocksInGrid.y, numOfBlocksInGrid.x * numOfBlocksInGrid.y);
  if (h_columns == v.h_rows) {
    multiplyNaive<<<blocks, threads>>>(this->d_mat, this->d_rows, this->d_columns, v.d_mat, v.d_rows, v.d_columns, out.d_mat);
  } else {
    printf("MATRICIES CANNOT BE MULTIPLED, INVALID SIZES");
  }
  // cudaDeviceSynchronize();
  // dim3 grid(1, 1, 1);
  // dim3 block(out.h_rows, out.h_columns, 1);
  // multiplyNaive<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns, v.d_mat, v.d_rows,
  //                                v.d_columns, out.d_mat);
  return out;
};

// Vector_GPU Vector_GPU::multNai(Vector_GPU &v) {
Vector_GPU Vector_GPU::operator*(Vector_GPU &v) {
  // printf("MATMULT\n");
  Vector_GPU out(this->h_rows, v.h_columns);
  dim3 numOfThreadsInBlock(TILE_DIM_X, TILE_DIM_Y, 1);
  unsigned int blocksY = (out.h_rows / TILE_DIM_X) + 1;
  unsigned int blocksX = (out.h_columns / TILE_DIM_Y) + 1;
  dim3 numOfBlocksInGrid(blocksX, blocksY, 1);
  // printf("threadsinblock(%d x %d)=%d, blocksingrid(%d, %d)=%d\n", numOfThreadsInBlock.x, numOfThreadsInBlock.y,
  //       numOfThreadsInBlock.x * numOfThreadsInBlock.y, numOfBlocksInGrid.x, numOfBlocksInGrid.y, numOfBlocksInGrid.x * numOfBlocksInGrid.y);
  multiplyTiled<<<numOfBlocksInGrid, numOfThreadsInBlock>>>(this->d_mat, this->d_rows, this->d_columns, v.d_mat, v.d_rows, v.d_columns, out.d_mat);
  // cudaDeviceSynchronize();
  // dim3 grid(1, 1, 1);
  // dim3 block(out.h_rows, out.h_columns, 1);
  // multiplyNaive<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns, v.d_mat, v.d_rows,
  //                                v.d_columns, out.d_mat);
  return out;
}