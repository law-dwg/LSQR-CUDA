#include "cublas_v2.h"
#include "matVec_cublas.cuh"
#include "utils.cuh"
#include <assert.h>
#include <stdio.h>
const double ONE = 1.0;
const double ZERO = 0.0;
const double NEGONE = -1.0;
// void __global__ print(double *input, unsigned *r, unsigned *c) {
//  const unsigned int bid = blockIdx.x                               // 1D
//                           + blockIdx.y * gridDim.x                 // 2D
//                           + gridDim.x * gridDim.y * blockIdx.z;    // 3D
//  const unsigned int threadsPerBlock = blockDim.x * blockDim.y      // 2D
//                                       * blockDim.z;                // 3D
//  const unsigned int tid = threadIdx.x                              // 1D
//                           + threadIdx.y * blockDim.x               // 2D
//                           + blockDim.x * blockDim.x * threadIdx.z; // 3D
//  const unsigned int gid = bid * threadsPerBlock + tid;
//  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d,
//  // value=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
//  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,input[gid]);
//  __syncthreads();
//  if (gid < *r * *c) {
//    printf("%f\n", input[gid]);
//  }
//}
/** Operator overloads */
Vector_CUBLAS Vector_CUBLAS::operator*(Vector_CUBLAS &v) {
  Vector_CUBLAS out(this->h_rows, v.h_columns);
  int m = out.h_rows;
  int n = out.h_columns;
  int k = this->h_columns;
  int lda = m, ldb = k, ldc = m;
  // printf("m=%d,n=%d,k=%d,lda=%d, ldb=%d, ldc=%d, d_rows=%d, d_cols=%d,\n", m, n, k, lda, ldb, ldc, rows, cols);
  stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &ONE, this->d_mat, lda, v.d_mat, ldb, &ZERO, out.d_mat, ldc);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << cublasGetErrorString(stat) << std::endl;
    printf("CUBLAS matrix mult failed\n");
  }

  return out;
};
Vector_CUBLAS Vector_CUBLAS::operator*(double i) {
  Vector_CUBLAS out(this->h_rows, this->h_columns);
  stat = cublasDcopy(handle, this->h_rows * this->h_columns, this->d_mat, 1, out.d_mat, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << cublasGetErrorString(stat) << std::endl;
    printf("CUBLAS scale failed\n");
  }
  stat = cublasDscal(handle, this->h_rows * this->h_columns, &i, out.d_mat, 1);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << cublasGetErrorString(stat) << std::endl;
    printf("CUBLAS scale failed\n");
  }
  return out;
};
void Vector_CUBLAS::operator=(Vector_CPU &v){};

Vector_CUBLAS Vector_CUBLAS::operator-(const Vector_CUBLAS &v) {
  Vector_CUBLAS out(this->h_rows, this->h_columns);
  if (this->h_rows * this->h_columns == v.h_rows * v.h_columns) {
    int m = out.h_rows;
    int n = out.h_columns;
    int k = this->h_columns;
    int lda = m, ldb = k, ldc = m;
    cublasOperation_t OP = (m == 1 || n == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    // printf("h_rows = %d, h_cols = %d, v.h_rows = %d, v.h_cols = %d\n",this->h_rows,this->h_columns,v.h_rows,v.h_columns);
    stat = cublasDgeam(handle, CUBLAS_OP_N, OP, m, n, &ONE, this->d_mat, lda, &NEGONE, v.d_mat, ldb, out.d_mat, ldc); // ADD-SUB-TRANSPOSE
    cudaDeviceSynchronize();
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cout << cublasGetErrorString(stat) << std::endl;
      printf("CUBLAS matrix subract failed\n");
    }
  } else
    printf("CUBLAS SUBTRACT ERROR, MATRICES ARENT SAME SIZE\n");
  return out;
};
Vector_CUBLAS Vector_CUBLAS::operator+(const Vector_CUBLAS &v) {
  Vector_CUBLAS out(this->h_rows, this->h_columns);
  // printf("h_rows = %d, h_cols = %d, v.h_rows = %d, v.h_cols = %d\n",this->h_rows,this->h_columns,v.h_rows,v.h_columns);
  if (this->h_rows * this->h_columns == v.h_rows * v.h_columns) {
    int m = out.h_rows;
    int n = out.h_columns;
    int k = this->h_columns;
    int lda = m, ldb = k, ldc = m;
    cublasOperation_t OP = (m == 1 || n == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    stat = cublasDgeam(handle, CUBLAS_OP_N, OP, m, n, &ONE, this->d_mat, lda, &ONE, v.d_mat, ldb, out.d_mat, ldc); // ADD-SUB-TRANSPOSE
    cudaDeviceSynchronize();
    if (stat != CUBLAS_STATUS_SUCCESS) {
      std::cout << cublasGetErrorString(stat) << std::endl;
      printf("CUBLAS matrix addition failed\n");
    }
  } else {
    printf("CUBLAS ADDITION ERROR, MATRICES ARENT SAME SIZE\n");
  }
  return out;
};

/** Member functions */
// void printmat();
Vector_CPU Vector_CUBLAS::matDeviceToHost() {
  double *out = new double[this->h_columns * this->h_rows]; // heap to prevent a stack overflow
  unsigned int rows;
  unsigned int cols;
  gpuErrchk(cudaMemcpy(out, this->d_mat, sizeof(double) * this->h_columns * this->h_rows, cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(&rows, this->d_rows, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(&cols, this->d_columns, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  // printf("d_rows = %d, h_rows = %d, d_cols = %d, h_cols = %d\n", rows, h_rows, cols, h_columns);
  if (rows != this->h_rows || cols != this->h_columns) {
    printf("INCONSISTENT ROWS AND COLS BETWEEN HOST AND DEVICE\n");
  }
  Vector_CPU v_cpu(this->h_rows, this->h_columns, out);
  return v_cpu;
};

double Vector_CUBLAS::Dnrm2() {
  double h_out;
  int *version;
  double *d_out;
  double zero = 0.0;
  gpuErrchk(cudaMalloc(&d_out, sizeof(double)));
  int size = (this->h_rows * this->h_columns);

  int incre = 1;
  // stat = cublasSetVector(size, sizeof(double), (void **)&c_d_mat, incre, d_mat, incre);
  // if (stat != CUBLAS_STATUS_SUCCESS) {
  //   printf ("CUBLAS Version failed\n");
  //   return EXIT_FAILURE;
  // }

  cublasStatus_t stat3 = cublasDnrm2(handle, size, this->d_mat, incre, &h_out);
  gpuErrchk(cudaDeviceSynchronize());
  if (stat3 != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS DNRM2 failed\n");
    // return EXIT_FAILURE;
  }

  // stat = cublasDestroy(handle);
  // if (stat != CUBLAS_STATUS_SUCCESS) {
  //   printf ("CUBLAS destroy failed\n");
  //   return EXIT_FAILURE;
  // }
  // cublasDestroy(handle);
  // gpuErrchk(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  assert(!(h_out != h_out));
  return h_out;
};

void Vector_CUBLAS::printmat() {
  unsigned int blocksX = (this->h_rows / 16) + 1;
  unsigned int blocksY = (this->h_columns / 16) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(16, 16, 1);

  // print<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns);
  // gpuErrchk(cudaPeekAtLastError());
  cudaDeviceSynchronize();
}

Vector_CUBLAS Vector_CUBLAS::transpose() {
  Vector_CUBLAS out(this->h_columns, this->h_rows);
  int m = this->h_rows, n = this->h_columns, k = this->h_columns;
  int lda = m, ldb = k, ldc = m;
  stat = cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &ZERO, this->d_mat, lda, &ONE, this->d_mat, ldb, out.d_mat, ldc); // ADD-SUB-TRANSPOSE
  if (stat != CUBLAS_STATUS_SUCCESS) {
    std::cout << cublasGetErrorString(stat) << std::endl;
    printf("CUBLAS matrix transpose failed\n");
  }
  return out;
};