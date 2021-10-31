#include "vectorCUBLAS.cuh"
#include <assert.h>
#include <stdio.h>

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
    printf("Cannot perform multiplication, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  return out;
};

VectorCUBLAS VectorCUBLAS::operator*(double i) {
  VectorCUBLAS out(this->h_rows, this->h_columns);
  cublasErrCheck(cublasDcopy(handle, this->h_rows * this->h_columns, this->d_mat, 1, out.d_mat, 1));
  cublasErrCheck(cublasDscal(handle, this->h_rows * this->h_columns, &i, out.d_mat, 1));
  return out;
};

void VectorCUBLAS::operator=(VectorCPU &v){
  cudaErrCheck(cudaFree(d_mat));
  h_rows = v.getRows();
  h_columns = v.getColumns();
  cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.getRows() * v.getColumns()));
  cudaErrCheck(cudaMemcpy(d_rows, &h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_columns, &h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_mat, v.getMat(), sizeof(double) * h_rows * h_columns, cudaMemcpyHostToDevice));
};

VectorCUBLAS VectorCUBLAS::operator-(const VectorCUBLAS &v) {
  VectorCUBLAS out(this->h_rows, this->h_columns);
  if (this->h_rows * this->h_columns == v.h_rows * v.h_columns) {
    int m = out.h_rows;
    int n = out.h_columns;
    int k = this->h_columns;
    int lda = m, ldb = k, ldc = m;
    cublasOperation_t OP = (m == 1 || n == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasErrCheck(cublasDgeam(handle, CUBLAS_OP_N, OP, m, n, &ONE, this->d_mat, lda, &NEGONE, v.d_mat, ldb, out.d_mat, ldc));
  } else {
    printf("Cannot perform subtraction, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  return out;
};

VectorCUBLAS VectorCUBLAS::operator+(const VectorCUBLAS &v) {
  VectorCUBLAS out(this->h_rows, this->h_columns);
  if (this->h_rows * this->h_columns == v.h_rows * v.h_columns) {
    int m = out.h_rows;
    int n = out.h_columns;
    int k = this->h_columns;
    int lda = m, ldb = k, ldc = m;
    cublasOperation_t OP = (m == 1 || n == 1) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasErrCheck(cublasDgeam(handle, CUBLAS_OP_N, OP, m, n, &ONE, this->d_mat, lda, &ONE, v.d_mat, ldb, out.d_mat, ldc));
  } else {
    printf("Cannot perform addition, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  return out;
};

/** Member functions */
VectorCPU VectorCUBLAS::matDeviceToHost() {
  double *out = new double[this->h_columns * this->h_rows]; // heap to prevent a stack overflow
  unsigned rows;
  unsigned cols;
  cudaErrCheck(cudaMemcpy(out, this->d_mat, sizeof(double) * this->h_columns * this->h_rows, cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&rows, this->d_rows, sizeof(unsigned), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&cols, this->d_columns, sizeof(unsigned), cudaMemcpyDeviceToHost));
  // printf("d_rows = %d, h_rows = %d, d_cols = %d, h_cols = %d\n", rows, h_rows, cols, h_columns);
  if (rows != this->h_rows || cols != this->h_columns) {
    printf("Cannot perform move to host, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  VectorCPU v_cpu(this->h_rows, this->h_columns, out);
  return v_cpu;
};

double VectorCUBLAS::Dnrm2() {
  double h_out;
  double *d_out;
  cudaErrCheck(cudaMalloc(&d_out, sizeof(double)));
  int size = (this->h_rows * this->h_columns);
  int incre = 1;
  cublasErrCheck(cublasDnrm2(handle, size, this->d_mat, incre, &h_out));
  // sanity checks
  assert(!(h_out != h_out));
  assert(h_out > 0);
  return h_out;
};

void VectorCUBLAS::printMat() {
  unsigned blocksX = (this->h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (this->h_columns / BLOCK_SIZE_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  print<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns);
}

VectorCUBLAS VectorCUBLAS::transpose() {
  VectorCUBLAS out(this->h_columns, this->h_rows);
  cublasErrCheck(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, this->h_rows, this->h_columns, &ONE, this->d_mat, this->h_columns, &ZERO, this->d_mat,
                             this->h_columns, out.d_mat, this->h_rows));
  return out;
};