#include "Kernels.cuh"
#include <cooperative_groups.h>
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

static __inline__ __device__ double atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

using namespace cooperative_groups;
// OUR TILE SIZE SHOULD MATCH THAT OF OUR BLOCK
#define TILE_DIM_X 16
#define TILE_DIM_Y 16

void __global__ print(double *input, unsigned *r, unsigned *c) {
  const unsigned bid = blockIdx.x                               // 1D
                       + blockIdx.y * gridDim.x                 // 2D
                       + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                   * blockDim.z;                // 3D
  const unsigned tid = threadIdx.x                              // 1D
                       + threadIdx.y * blockDim.x               // 2D
                       + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned gid = bid * threadsPerBlock + tid;
  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d,
  // value=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,input[gid]);
  __syncthreads();
  if (gid < *r * *c) {
    printf("%f\n", input[gid]);
  }
}

void __global__ dnrm2(double *in1, unsigned r, unsigned c, double *max, double *out) {
  extern __shared__ double sum[];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  int x2 = (blockIdx.x * (blockDim.x * 2) + threadIdx.x) + blockDim.x;
  // load into shared
  // sum[threadIdx.x] = in1[x] * in1[x] + in1[x2] * in1[x2];

  if (x < (r * c) && x2 < (r * c)) {
    sum[threadIdx.x] = in1[x] / *max * in1[x] / *max + in1[x2] / *max * in1[x2] / *max;
    // printf("INIT: block(%d, %d) thread(%d, %d) sum[%d] =  in1[%d] + in2[%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.x,
    // x, x2, sum[threadIdx.x]);
  } else if ((x < (r * c)) && ((r * c) <= x2)) {
    sum[threadIdx.x] = in1[x] / *max * in1[x] / *max;
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
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
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

void __global__ maxVal(double *in1, unsigned r, unsigned c, double *out) {
  extern __shared__ double maxV[];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  int x2 = (blockIdx.x * (blockDim.x * 2) + threadIdx.x) + blockDim.x;
  // load into shared
  // sum[threadIdx.x] = in1[x] * in1[x] + in1[x2] * in1[x2];

  if (x < (r * c) && x2 < (r * c)) {
    maxV[threadIdx.x] = fmax(in1[x], in1[x2]);
    // printf("INIT: block(%d, %d) thread(%d, %d) sum[%d] =  in1[%d] + in2[%d] = %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.x,
    // x, x2, sum[threadIdx.x]);
  } else if ((x < (r * c)) && ((r * c) <= x2)) {
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
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
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
