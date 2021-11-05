/*=========================================================================
 *
 *  CUDA KERNELS
 *
 *=========================================================================*/

#include "kernels.cuh"
#define FULL_MASK 0xffffffff
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
double __device__ warpReduce(double sdata) {
  // Reference:
  // https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
  //
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    sdata += __shfl_down_sync(FULL_MASK, sdata, offset, warpSize);
  return sdata;
};

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
  int tileWidth = blockDim.y;
  int tileHeight = blockDim.x;
  double *A = (double *)array;
  double *B = (double *)&A[tileWidth * tileHeight];
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row
  int x = blockIdx.x * blockDim.x + threadIdx.x; // col
  double sum = 0;                                // sum in block

  for (int i = 0; i < *cols1; i += blockDim.x) {
    int id1, id2;
    // load into shared
    if (i + threadIdx.x < *cols1 && y < *rows1) {
      id1 = y * *cols1 + i + threadIdx.x;
      A[tileWidth * threadIdx.y + threadIdx.x] = in1[id1];
    } else {
      A[tileWidth * threadIdx.y + threadIdx.x] = 0.0;
    }
    if (i + threadIdx.y < *rows2 && x < *cols2) {
      id2 = (i * *cols2 + threadIdx.y * *cols2) + x;
      B[tileWidth * threadIdx.y + threadIdx.x] = in2[id2];
    } else {
      B[tileWidth * threadIdx.y + threadIdx.x] = 0.0;
    }
    __syncthreads();
    // perform multiplication
    for (int j = 0; j < blockDim.x; j++) {
      int Ax, Ay, Bx, By;
      Ax = j;
      Ay = threadIdx.y;
      Bx = threadIdx.x;
      By = j;
      sum += A[tileWidth * Ay + Ax] * B[tileWidth * By + Bx];
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
  // reference: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

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
void __global__ spmvNaive(unsigned *rows, int *csrRowPtr, int *csrColIdn, double *csrVal, double *rhs, double *out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x; // global scope tid
  if (gid < *rows) {
    out[gid] = 0.0;
    for (int i = csrRowPtr[gid]; i < csrRowPtr[gid + 1]; ++i) {
      out[gid] += csrVal[i] * rhs[csrColIdn[i]];
    }
  }
};

void __global__ spmvCSRVector(unsigned *rows, int *csrRowPtr, int *csrColInd, double *csrVal, double *rhs, double *out) {
  // Reference:
  // https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f
  //
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; // global id
  unsigned wid = gid / warpSize;                        // warp id
  unsigned lane = threadIdx.x % warpSize;               // id within warp (0-31)
  unsigned row = wid;                                   // one row == one warp
  // printf("grid%d %d %d",gridDim.x,gridDim.y,gridDim.z);
  if (row < *rows) {
    int rowStart = csrRowPtr[row];
    int rowEnd = csrRowPtr[row + 1];
    double sum = 0;
    for (int i = rowStart + lane; i < rowEnd; i += warpSize) {
      sum += csrVal[i] * rhs[csrColInd[i]];
    };

    // __shfl_down_sync
    double temp = warpReduce(sum);

    if (lane == 0) {
      out[row] = temp;
    }
  }
};

void __global__ spmvCSRVectorShared(unsigned *rows, int *csrRowPtr, int *csrColInd, double *csrVal, double *rhs, double *out) {
  // Reference:
  // https://medium.com/analytics-vidhya/sparse-matrix-vector-multiplication-with-cuda-42d191878e8f
  // and
  // https://github.com/poojahira/spmv-cuda/blob/master/code/src/spmv_csr_vector.cu
  // use of shared memory rather than __shfl_down_sync
  // no speedup between with and without shared memory
  //
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x; // global id
  unsigned wid = gid / warpSize;                        // warp id
  unsigned lane = threadIdx.x % warpSize;               // id within warp (0-31)
  unsigned row = wid;                                   // one row == one warp
  extern __shared__ volatile double sumSpmv[];          // shared + volatile
  // printf("grid%d %d %d",gridDim.x,gridDim.y,gridDim.z);
  if (row < *rows) {
    int rowStart = csrRowPtr[row];
    int rowEnd = csrRowPtr[row + 1];
    sumSpmv[threadIdx.x] = 0;
    for (int i = rowStart + lane; i < rowEnd; i += warpSize) {
      sumSpmv[threadIdx.x] += csrVal[i] * rhs[csrColInd[i]];
    };

    // reduce 32 threads down to one -> add up sums between 32 threads
    __syncthreads();
    if (lane < 16)
      sumSpmv[threadIdx.x] += sumSpmv[threadIdx.x + 16]; // e.g. sumSpmv[15] = sumSpmv[15] + sumSpmv[31]
    if (lane < 8)
      sumSpmv[threadIdx.x] += sumSpmv[threadIdx.x + 8];
    if (lane < 4)
      sumSpmv[threadIdx.x] += sumSpmv[threadIdx.x + 4];
    if (lane < 2)
      sumSpmv[threadIdx.x] += sumSpmv[threadIdx.x + 2];
    if (lane < 1)
      sumSpmv[threadIdx.x] += sumSpmv[threadIdx.x + 1]; // final reduction
    __syncthreads();

    if (lane == 0) {
      out[row] = sumSpmv[threadIdx.x];
    }
  }
};