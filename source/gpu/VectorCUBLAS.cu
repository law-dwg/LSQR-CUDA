#include "VectorCUBLAS.cuh"
#include <assert.h>
// void __global__ print(double *input, unsigned *r, unsigned *c) {
//  const unsigned bid = blockIdx.x                               // 1D
//                           + blockIdx.y * gridDim.x                 // 2D
//                           + gridDim.x * gridDim.y * blockIdx.z;    // 3D
//  const unsigned threadsPerBlock = blockDim.x * blockDim.y      // 2D
//                                       * blockDim.z;                // 3D
//  const unsigned tid = threadIdx.x                              // 1D
//                           + threadIdx.y * blockDim.x               // 2D
//                           + blockDim.x * blockDim.x * threadIdx.z; // 3D
//  const unsigned gid = bid * threadsPerBlock + tid;
//  // printf("thread(%d,%d,%d), block(%d,%d,%d), bid=%d, gid=%d,
//  // value=%f\n",threadIdx.x,threadIdx.y,threadIdx.z,
//  //    blockIdx.x,blockIdx.y,blockIdx.z,bid,gid,input[gid]);
//  __syncthreads();
//  if (gid < *r * *c) {
//    printf("%f\n", input[gid]);
//  }
//}
/** Operator overloads */
VectorCUBLAS VectorCUBLAS::operator*(VectorCUBLAS &v) {

  // a(m x k) * b(k * n) = c(m * n)
  VectorCUBLAS out(this->getRows(), v.getColumns());
  if (this->getColumns() == v.getRows()) {
    if (v.h_columns == 1) {
      cublasDgemv(handle, CUBLAS_OP_T, this->h_columns, this->h_rows, &ONE, this->d_mat, this->h_columns, v.d_mat, 1, &ZERO, out.d_mat, 1);
    } else {
      cublasErrCheck(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, out.h_columns, out.h_rows, this->h_columns, &ONE, v.d_mat, out.h_columns,
                                 this->d_mat, this->h_columns, &ZERO, out.d_mat, out.h_columns));
    }
  } else {
    printf("Cannot perform multiplication, dimension mismatch\n");
  }
  return out;
};

VectorCUBLAS VectorCUBLAS::operator*(double i) {
  VectorCUBLAS out(this->h_rows, this->h_columns);
  cublasErrCheck(cublasDcopy(handle, this->h_rows * this->h_columns, this->d_mat, 1, out.d_mat, 1));
  cublasErrCheck(cublasDscal(handle, this->h_rows * this->h_columns, &i, out.d_mat, 1));
  return out;
};

void VectorCUBLAS::operator=(Vector_CPU &v){};

VectorCUBLAS VectorCUBLAS::operator-(const VectorCUBLAS &v) {
  VectorCUBLAS out(this->h_rows, this->h_columns);
  if (this->h_rows * this->h_columns == v.h_rows * v.h_columns) {
    int m = out.h_rows;
    int n = out.h_columns;
    int k = this->h_columns;
    int lda = m, ldb = k, ldc = m;
    cublasOperation_t OP = (m == 1 || n == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    // printf("h_rows = %d, h_cols = %d, v.h_rows = %d, v.h_cols = %d\n",this->h_rows,this->h_columns,v.h_rows,v.h_columns);
    cublasErrCheck(cublasDgeam(handle, CUBLAS_OP_N, OP, m, n, &ONE, this->d_mat, lda, &NEGONE, v.d_mat, ldb, out.d_mat, ldc));
  } else
    printf("CUBLAS SUBTRACT ERROR, MATRICES ARENT SAME SIZE\n");
  return out;
};

VectorCUBLAS VectorCUBLAS::operator+(const VectorCUBLAS &v) {
  VectorCUBLAS out(this->h_rows, this->h_columns);
  // printf("h_rows = %d, h_cols = %d, v.h_rows = %d, v.h_cols = %d\n",this->h_rows,this->h_columns,v.h_rows,v.h_columns);
  if (this->h_rows * this->h_columns == v.h_rows * v.h_columns) {
    int m = out.h_rows;
    int n = out.h_columns;
    int k = this->h_columns;
    int lda = m, ldb = k, ldc = m;
    cublasOperation_t OP = (m == 1 || n == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasErrCheck(cublasDgeam(handle, CUBLAS_OP_N, OP, m, n, &ONE, this->d_mat, lda, &ONE, v.d_mat, ldb, out.d_mat, ldc));
  } else {
    printf("CUBLAS ADDITION ERROR, MATRICES ARENT SAME SIZE\n");
  }
  return out;
};

/** Member functions */
Vector_CPU VectorCUBLAS::matDeviceToHost() {
  double *out = new double[this->h_columns * this->h_rows]; // heap to prevent a stack overflow
  unsigned rows;
  unsigned cols;
  cudaErrCheck(cudaMemcpy(out, this->d_mat, sizeof(double) * this->h_columns * this->h_rows, cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&rows, this->d_rows, sizeof(unsigned), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&cols, this->d_columns, sizeof(unsigned), cudaMemcpyDeviceToHost));
  // printf("d_rows = %d, h_rows = %d, d_cols = %d, h_cols = %d\n", rows, h_rows, cols, h_columns);
  if (rows != this->h_rows || cols != this->h_columns) {
    printf("INCONSISTENT ROWS AND COLS BETWEEN HOST AND DEVICE\n");
  }
  Vector_CPU v_cpu(this->h_rows, this->h_columns, out);
  return v_cpu;
};

double VectorCUBLAS::Dnrm2() {
  double h_out;
  int *version;
  double *d_out;
  double zero = 0.0;
  cudaErrCheck(cudaMalloc(&d_out, sizeof(double)));
  int size = (this->h_rows * this->h_columns);
  int incre = 1;
  cublasErrCheck(cublasDnrm2(handle, size, this->d_mat, incre, &h_out));
  assert(!(h_out != h_out));
  return h_out;
};

void VectorCUBLAS::printmat() {
  unsigned blocksX = (this->h_rows / 16) + 1;
  unsigned blocksY = (this->h_columns / 16) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(16, 16, 1);

  // print<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns);
  // cudaErrCheck(cudaPeekAtLastError());
  cudaDeviceSynchronize();
}

VectorCUBLAS VectorCUBLAS::transpose() {
  VectorCUBLAS out(this->h_columns, this->h_rows);
  cublasErrCheck(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, this->h_rows, this->h_columns, &ONE, this->d_mat, this->h_columns, &ZERO, this->d_mat,
                             this->h_columns, out.d_mat, this->h_rows));
  return out;
};