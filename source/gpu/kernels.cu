/*=========================================================================
 *
 *  CUDA KERNELS
 *
 *=========================================================================*/

#include "kernels.cuh"
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
/** atomicAdd for doubles (if not supported) */
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
};
#endif
/** atomicMax for doubles */
static __inline__ __device__ double atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
};

/** VectorCUDA and MatrixCUDA Kernels */
void __global__ print(double *input, unsigned *r, unsigned *c) {
  // used for debugging
  const unsigned bid = blockIdx.x                               // 1D
                       + blockIdx.y * gridDim.x                 // 2D
                       + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                   * blockDim.z;                // 3D
  const unsigned tid = threadIdx.x                              // 1D
                       + threadIdx.y * blockDim.x               // 2D
                       + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned gid = bid * threadsPerBlock + tid;
  __syncthreads();
  if (gid < *r * *c) {
    printf("%f\n", input[gid]);
  }
};
void __global__ maxVal(double *in1, unsigned r, unsigned c, double *out) {
  extern __shared__ double maxV[];
  int x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  int x2 = (blockIdx.x * (blockDim.x * 2) + threadIdx.x) + blockDim.x;
  // load into shared
  if (x < (r * c) && x2 < (r * c)) {
    maxV[threadIdx.x] = fmax(in1[x], in1[x2]);
  } else if ((x < (r * c)) && ((r * c) <= x2)) {
    maxV[threadIdx.x] = in1[x];
  } else {
    maxV[threadIdx.x] = 0.0;
  }
  __syncthreads();
  // standard reduction
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      maxV[threadIdx.x] = fmax(std::abs(maxV[threadIdx.x]), maxV[threadIdx.x + s]);
    }
    __syncthreads();
  }
  // save to out
  if (threadIdx.x == 0) {
    atomicMax(out, std::abs(maxV[0]));
  }
};
void __global__ dnrm2(double *in1, unsigned r, unsigned c, double *max, double *out) {
  extern __shared__ double sum[];
  int x = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
  int x2 = (blockIdx.x * (blockDim.x * 2) + threadIdx.x) + blockDim.x;
  // load into shared
  if (x < (r * c) && x2 < (r * c)) {
    sum[threadIdx.x] = in1[x] / *max * in1[x] / *max + in1[x2] / *max * in1[x2] / *max;
  } else if ((x < (r * c)) && ((r * c) <= x2)) {
    sum[threadIdx.x] = in1[x] / *max * in1[x] / *max;
  } else {
    sum[threadIdx.x] = 0.0;
  }
  __syncthreads();
  // standard reduction
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sum[threadIdx.x] += sum[threadIdx.x + s];
      sum[threadIdx.x + s] = 0;
    }
    __syncthreads();
  }
  // save to out
  if (threadIdx.x == 0) {
    atomicAdd(out, sum[0]);
  }
};

/** VectorCUDA Kernels */
void __global__ multiplyNaive(double *in1, unsigned *rows1, unsigned *cols1, double *in2, unsigned *rows2, unsigned *cols2, double *output) {
  const unsigned x = blockIdx.y * blockDim.y + threadIdx.y; // row
  const unsigned y = blockIdx.x * blockDim.x + threadIdx.x; // column
  double sum = 0;
  if ((x < *rows1) && (y < *cols2)) {
    for (int i = 0; i < *cols1; ++i) {
      sum += in1[x * *cols1 + i] * in2[i * *cols2 + y];
    }
    output[x * *cols2 + y] = sum;
  }
};
void __global__ multiplyTiled(double *in1, unsigned *rows1, unsigned *cols1, double *in2, unsigned *rows2, unsigned *cols2, double *output) {
  // BLOCK AND TILE SWEEP TOGETHER (BLOCK_SIZE = TILE_SIZE)
  extern __shared__ double array[];
  double *A = (double *)array;
  double *B = (double *)&A[blockDim.x * (blockDim.y + 1)];
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row
  int x = blockIdx.x * blockDim.x + threadIdx.x; // col
  double sum = 0;                                // sum in block

  for (int i = 0; i < *cols1; i += blockDim.x) {
    int id1, id2;
    // load into shared
    if (i + threadIdx.x < *cols1 && y < *rows1) {
      id1 = y * *cols1 + i + threadIdx.x;
      A[(blockDim.y + 1) * threadIdx.y + threadIdx.x] = in1[id1];
    } else {
      A[(blockDim.y + 1) * threadIdx.y + threadIdx.x] = 0.0;
    }
    if (i + threadIdx.y < *rows2 && x < *cols2) {
      id2 = (i * *cols2 + threadIdx.y * *cols2) + x;
      B[(blockDim.y + 1) * threadIdx.y + threadIdx.x] = in2[id2];
    } else {
      B[(blockDim.y + 1) * threadIdx.y + threadIdx.x] = 0.0;
    }
    __syncthreads();
    // perform multiplication
    for (int j = 0; j < blockDim.x; j++) {
      int Ax, Ay, Bx, By;
      Ax = j;
      Ay = threadIdx.y;
      Bx = threadIdx.x;
      By = j;
      sum += A[(blockDim.y + 1) * Ay + Ax] * B[(blockDim.y + 1) * By + Bx];
    };
    __syncthreads();
    // save to out
  };
  if (x < *cols2 && y < *rows1) {
    output[y * *cols2 + x] = sum;
  }
};
void __global__ scale(double *input, double scalar, double *output, unsigned *r, unsigned *c, bool inverse) {
  const unsigned bid = blockIdx.x                               // 1D
                       + blockIdx.y * gridDim.x                 // 2D
                       + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                   * blockDim.z;                // 3D
  const unsigned tid = threadIdx.x                              // 1D
                       + threadIdx.y * blockDim.x               // 2D
                       + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned gid = bid * threadsPerBlock + tid;
  if (gid < *r * *c) {
    if (inverse) {
      output[gid] = input[gid] * (1.0 / scalar);
    } else {
      output[gid] = input[gid] * scalar;
    }
  }
};
void __global__ add(double *in1, double *in2, unsigned *rows, unsigned *cols, double *out, bool add) {
  const unsigned bid = blockIdx.x                               // 1D
                       + blockIdx.y * gridDim.x                 // 2D
                       + gridDim.x * gridDim.y * blockIdx.z;    // 3D
  const unsigned threadsPerBlock = blockDim.x * blockDim.y      // 2D
                                   * blockDim.z;                // 3D
  const unsigned tid = threadIdx.x                              // 1D
                       + threadIdx.y * blockDim.x               // 2D
                       + blockDim.x * blockDim.x * threadIdx.z; // 3D
  const unsigned gid = bid * threadsPerBlock + tid;
  // switch for add or subtract
  int s = add ? 1 : -1;
  if (gid < *rows * *cols)
    out[gid] = in1[gid] + (s * in2[gid]);
};
void __global__ transposeTiled(double *in1, double *output, unsigned *rows, unsigned *cols) {
  // BLOCK SWEEPS ACROSS TILE (TILE SIZE > BLOCK SIZE)
  // source: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

  extern __shared__ double A[];                  // Add +1 to prevent bank-conflicts
  int x = blockIdx.x * blockDim.x + threadIdx.x; // col
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row

  // load into shared
  for (int i = 0; i < blockDim.y; i += blockDim.y) {
    if ((x < *cols) && (y < *rows)) {
      A[threadIdx.y + i + blockDim.x * threadIdx.x] = in1[(y + i) * *cols + x];
    }
  };
  __syncthreads();
  // perform transform
  x = blockIdx.y * blockDim.x + threadIdx.x; // x-dimension col
  y = blockIdx.x * blockDim.y + threadIdx.y; // y-dimension row
  for (int i = 0; i < blockDim.y; i += blockDim.y) {
    if ((y + i < *cols) && (x < *rows)) {
      // save to out
      output[(y + i) * *rows + x] = A[threadIdx.x + blockDim.x * (threadIdx.y + i)];
    }
  }
};

/** MatrixCUDA Kernels */
void __global__ spmvNaive(unsigned *rows, unsigned *col, int *rowPtr, int *colIdx, double *val, double *rhs, double *out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid;
  if (gid < *rows) {
    out[gid] = double(0.0);
    for (bid = rowPtr[gid]; bid < rowPtr[gid + 1]; ++bid) {
      out[gid] += val[bid] * rhs[colIdx[bid]];
    }
  }
};