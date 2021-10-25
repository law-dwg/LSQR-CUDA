#include "Kernels.cuh"
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
// atomicAdd for doubles (if not supported)
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
// atomicMax for doubles
static __inline__ __device__ double atomicMax(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = ::atomicCAS(address_as_ull, assumed, __double_as_longlong(::fmaxf(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
};

/** CUDA kernels */
/** VectorCUDA and MatrixCUDA Kernels */
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
};
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
};
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
};

/** VectorCUDA Kernels */
void __global__ multiplyNaive(double *in1, unsigned *rows1, unsigned *cols1, double *in2, unsigned *rows2, unsigned *cols2, double *output) {
  const unsigned x = blockIdx.y * blockDim.y + threadIdx.y; // row
  const unsigned y = blockIdx.x * blockDim.x + threadIdx.x; // column
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
};
void __global__ multiplyTiled(double *in1, unsigned *rows1, unsigned *cols1, double *in2, unsigned *rows2, unsigned *cols2, double *output) {
  // BLOCK AND TILE SWEEP TOGETHER (BLOCK_SIZE = TILE_SIZE)
  extern __shared__ double array[];
  double *A = (double *)array;
  double *B = (double *)&A[blockDim.x * blockDim.y];
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row
  int x = blockIdx.x * blockDim.x + threadIdx.x; // col
  double sum = 0;                                // sum in block

  for (int i = 0; i < *cols1; i += blockDim.x) {
    int id1, id2;
    if (i + threadIdx.x < *cols1 && y < *rows1) {
      id1 = y * *cols1 + i + threadIdx.x;
      A[threadIdx.x + blockDim.x * threadIdx.y] = in1[id1];
    } else {
      A[threadIdx.x + blockDim.x * threadIdx.y] = 0.0;
    }
    if (i + threadIdx.y < *rows2 && x < *cols2) {
      id2 = (i * *cols2 + threadIdx.y * *cols2) + x;
      B[threadIdx.x + blockDim.x * threadIdx.y] = in2[id2];
    } else {
      B[threadIdx.x * blockDim.x * threadIdx.y] = 0.0;
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
      sum += A[Ax + blockDim.x * Ay] * B[Bx + blockDim.x * By];
      //}
    };
    __syncthreads();
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
  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d,
  // value=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,input[gid]);
  if (gid < *r * *c) {
    if (inverse) {
      output[gid] = input[gid] * (1.0 / scalar);
      // printf("%f = %f / %f\n", output[gid], input[gid], *scalar);
    } else {
      output[gid] = input[gid] * scalar;
      // printf("out[%d] = %f = %f * %f\n", gid, output[gid], input[gid], *scalar);
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
  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d, in1=%f,
  // in2=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,in1[gid],in2[gid]);
  int s = add ? 1 : -1;
  if (gid < *rows * *cols)
    out[gid] = in1[gid] + (s * in2[gid]);
  // printf("%f = %f + %f\n", out[gid], in1[gid], in2[gid]);
};
void __global__ transposeTiled(double *in1, double *output, unsigned *rows, unsigned *cols) {
  // BLOCK SWEEPS ACROSS TILE (TILE SIZE > BLOCK SIZE)
  // source: https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/

  extern __shared__ double A[];                  // Add +1 to prevent bank-conflicts
  int x = blockIdx.x * blockDim.x + threadIdx.x; // col
  int y = blockIdx.y * blockDim.y + threadIdx.y; // row

  // Load the matrix into shared memory
  for (int i = 0; i < blockDim.y; i += blockDim.y) {
    if ((x < *cols) && (y < *rows)) {
      // A[(row + i) * height + col] = in1[(row + i) * width + col];
      A[threadIdx.y + i + blockDim.x * threadIdx.x] = in1[(y + i) * *cols + x];
      // printf(
      //     "block(%d, %d), thread(%d,% d), row = %d, col = %d, ,i=%d, A[%d][%d] "
      //     "= in1[%d] = %f\n",
      //     blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, y, x, i, threadIdx.y + i,
      //     threadIdx.x, (y + i) * *cols + x, in1[(y + i) * *cols + x]);
    }
  };

  __syncthreads();
  x = blockIdx.y * blockDim.x + threadIdx.x; // x-dimension col
  y = blockIdx.x * blockDim.y + threadIdx.y; // y-dimension row

  for (int i = 0; i < blockDim.y; i += blockDim.y) {
    // printf("block(%d, %d), thread(%d, %d), i=%d, A[%d][%d] = %f\n",
    //       blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, i,
    //       threadIdx.y+i,threadIdx.x, A[threadIdx.y+i][threadIdx.x]);
    if ((y + i < *cols) && (x < *rows)) {
      // output[col * width + (row+i)] = A[(row + i) * width + col];
      output[(y + i) * *rows + x] = A[threadIdx.x + blockDim.x * (threadIdx.y + i)];
      // printf(
      //     "block(%d, %d), thread(%d, %d), row = %d, col = %d, i=%d, output[%d] "
      //     "= A[%d][%d] = %f\n",
      //     blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, y, x, i, (y + i) * *rows + x,
      //     threadIdx.x, threadIdx.y + i, A[threadIdx.x][threadIdx.y + i]);
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