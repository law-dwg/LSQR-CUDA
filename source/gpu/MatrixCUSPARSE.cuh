#pragma once
#include "../cpu/matVec_cpu.hpp"
#include "matVec_cublas.cuh"
#include "matVec_gpu.cuh"
#include "utils.cuh"
#include <algorithm>
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

class MatrixCUSPARSE {
protected:
public:
  unsigned h_rows, h_columns, *d_rows, *d_columns;
  int *d_csr_offsets, *d_csr_columns;
  int64_t h_nnz;
  double *d_csr_values;
  cusparseSpMatDescr_t d_matDescr_sparse = NULL;
  cusparseMatDescr_t descr = NULL;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  MatrixCUSPARSE() : MatrixCUSPARSE(0, 0, 0) { // Default constructor
    printf("MatrixCUSPARSE Default Constructor called\n");
    cudaErrCheck(cudaMemset(d_csr_values, ZERO, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMemset(d_csr_columns, ZERO, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMemset(d_csr_offsets, ZERO, (h_rows + 1) * sizeof(int)));
  };
  MatrixCUSPARSE(unsigned r, unsigned c, int n) : h_rows(r), h_columns(c), h_nnz((int64_t)n) { // Helper constructor #1
    printf("MatrixCUSPARSE Helper Constructor #1 called\n");
    // memory allocation
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_csr_values, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMalloc((void **)&d_csr_columns, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&d_csr_offsets, (h_rows + 1) * sizeof(int)));
    cudaErrCheck(cudaMalloc(&dBuffer, bufferSize));
    // copy to device
    cudaErrCheck(cudaMemcpy(d_rows, &h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
    // cusparse
    cusparseErrCheck(cusparseCreateMatDescr(&descr));
    cusparseErrCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  };
  MatrixCUSPARSE(unsigned r, unsigned c, long long int n, double *values, int *columns, int *offsets)
      : MatrixCUSPARSE(r, c, n) { // Helper constructor #2
    printf("MatrixCUSPARSE Helper Constructor #2 called\n");
    // copy to device
    cudaErrCheck(cudaMemcpy(d_csr_values, values, h_nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_columns, columns, h_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_offsets, offsets, (h_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    // cusparse
    cusparseErrCheck(cusparseCreateCsr(&d_matDescr_sparse, h_rows, h_columns, h_nnz, d_csr_offsets, d_csr_columns, d_csr_values, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  }
  MatrixCUSPARSE(unsigned r, unsigned c, double *m) : h_rows(r), h_columns(c) { // Constructor (entry point)
    printf("MatrixCUSPARSE entry point called\n");
    // convert dense to sparse
    cusparseDnMatDescr_t d_matDescr_dense;
    double *d_mat;
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, r * c * sizeof(double)));
    cudaErrCheck(cudaMemcpy(d_rows, &r, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &c, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, m, r * c * sizeof(double), cudaMemcpyHostToDevice));
    cusparseErrCheck(cusparseCreateMatDescr(&descr));
    cusparseErrCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    // csr_offset ptr
    cudaErrCheck(cudaMalloc((void **)&d_csr_offsets, (r + 1) * sizeof(int)));
    // cusparse dense and sparse matricies
    cusparseErrCheck(cusparseCreateDnMat(&d_matDescr_dense, r, c, c, d_mat, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    cusparseErrCheck(cusparseCreateCsr(&d_matDescr_sparse, r, c, 0, d_csr_offsets, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseErrCheck(
        cusparseDenseToSparse_bufferSize(spHandle, d_matDescr_dense, d_matDescr_sparse, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
    cudaErrCheck(cudaMalloc(&dBuffer, bufferSize));
    cusparseErrCheck(cusparseDenseToSparse_analysis(spHandle, d_matDescr_dense, d_matDescr_sparse, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

    int64_t rows_tmp, cols_tmp;
    cusparseErrCheck(cusparseSpMatGetSize(d_matDescr_sparse, &rows_tmp, &cols_tmp, &h_nnz)); // determine # of non-zeros
    // printf("Sparse matrix created, rows=%d, cols=%d, nnz=%d\n", rows_tmp, cols_tmp, h_nnz);
    cudaErrCheck(cudaMalloc((void **)&d_csr_columns, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&d_csr_values, h_nnz * sizeof(double)));

    cusparseErrCheck(cusparseCsrSetPointers(d_matDescr_sparse, d_csr_offsets, d_csr_columns, d_csr_values));
    cusparseErrCheck(cusparseDenseToSparse_convert(spHandle, d_matDescr_dense, d_matDescr_sparse, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));
    cusparseDestroyDnMat(d_matDescr_dense);
    cudaFree(d_mat);
  };
  MatrixCUSPARSE(const MatrixCUSPARSE &m) : MatrixCUSPARSE(m.h_rows, m.h_columns, m.h_nnz, m.d_csr_values, m.d_csr_columns, m.d_csr_offsets) {
    printf("MatrixCUSPARSE Copy Constructor\n");
  };                                                   // Copy constructor
  MatrixCUSPARSE &operator=(const MatrixCUSPARSE &m) { // Copy assignment operator
    printf("MatrixCUSPARSE Copy Assignment Operator called\n");
    h_rows = m.h_rows;
    h_columns = m.h_columns;
    h_nnz = m.h_nnz;
    // destroy old allocation
    cudaErrCheck(cudaFree(d_csr_values));
    cudaErrCheck(cudaFree(d_csr_columns));
    cudaErrCheck(cudaFree(d_csr_offsets));
    cusparseErrCheck(cusparseDestroySpMat(d_matDescr_sparse));
    // memory allocation
    cudaErrCheck(cudaMalloc((void **)&d_csr_values, sizeof(double) * h_nnz));
    cudaErrCheck(cudaMalloc((void **)&d_csr_columns, sizeof(int) * h_nnz));
    cudaErrCheck(cudaMalloc((void **)&d_csr_offsets, sizeof(int) * (h_rows + 1)));
    // copy to device
    cudaErrCheck(cudaMemcpy(d_rows, m.d_rows, sizeof(unsigned), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, m.d_columns, sizeof(unsigned), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_values, m.d_csr_values, h_nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_columns, m.d_csr_columns, h_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_offsets, m.d_csr_offsets, (m.h_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    // cusparse
    cusparseErrCheck(cusparseCreateCsr(&d_matDescr_sparse, h_rows, h_columns, h_nnz, d_csr_offsets, d_csr_columns, d_csr_values, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    return *this;
  };
  MatrixCUSPARSE(MatrixCUSPARSE &&m) noexcept : MatrixCUSPARSE(m.h_rows, m.h_columns, m.h_nnz, m.d_csr_values, m.d_csr_columns, m.d_csr_offsets) {
    printf("MatrixCUSPARSE Move Constructor called\n");
    // free old resources
    cudaErrCheck(cudaFree(m.d_csr_values));
    cudaErrCheck(cudaFree(m.d_csr_offsets));
    cudaErrCheck(cudaFree(m.d_csr_columns));
    cusparseErrCheck(cusparseDestroySpMat(m.d_matDescr_sparse));
    m.h_rows = ZERO;
    m.h_nnz = ZERO;
    m.h_columns = ZERO;
    cudaErrCheck(cudaMalloc((void **)&m.d_csr_values, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMalloc((void **)&m.d_csr_columns, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&m.d_csr_offsets, (h_rows + 1) * sizeof(int)));
    cudaErrCheck(cudaMemcpy(m.d_rows, &m.h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(m.d_columns, &m.h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemset(m.d_csr_values, ZERO, m.h_nnz * sizeof(double)));
    cudaErrCheck(cudaMemset(m.d_csr_columns, ZERO, m.h_nnz * sizeof(int)));
    cudaErrCheck(cudaMemset(m.d_csr_offsets, ZERO, (m.h_rows + 1) * sizeof(int)));
    cusparseErrCheck(cusparseCreateCsr(&m.d_matDescr_sparse, m.h_rows, m.h_columns, m.h_nnz, m.d_csr_offsets, m.d_csr_columns, m.d_csr_values,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  };
  MatrixCUSPARSE &operator=(MatrixCUSPARSE &&m) noexcept { // Move assignment operator
    printf("MatrixCUSPARSE Copy Assignment called\n");
    // call copy assignment
    *this = m;
    // free old resources
    cudaErrCheck(cudaFree(m.d_csr_values));
    cudaErrCheck(cudaFree(m.d_csr_offsets));
    cudaErrCheck(cudaFree(m.d_csr_columns));
    cusparseErrCheck(cusparseDestroySpMat(m.d_matDescr_sparse));
    m.h_rows = ZERO;
    m.h_nnz = ZERO;
    m.h_columns = ZERO;
    cudaErrCheck(cudaMalloc((void **)&m.d_csr_values, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMalloc((void **)&m.d_csr_columns, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&m.d_csr_offsets, (h_rows + 1) * sizeof(int)));
    cudaErrCheck(cudaMemcpy(m.d_rows, &m.h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(m.d_columns, &m.h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemset(m.d_csr_values, ZERO, m.h_nnz * sizeof(double)));
    cudaErrCheck(cudaMemset(m.d_csr_columns, ZERO, m.h_nnz * sizeof(int)));
    cudaErrCheck(cudaMemset(m.d_csr_offsets, ZERO, (m.h_rows + 1) * sizeof(int)));
    cusparseErrCheck(cusparseCreateCsr(&m.d_matDescr_sparse, m.h_rows, m.h_columns, m.h_nnz, m.d_csr_offsets, m.d_csr_columns, m.d_csr_values,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    return *this;
  };
  ~MatrixCUSPARSE() {
    cusparseErrCheck(cusparseDestroySpMat(d_matDescr_sparse));
    cusparseErrCheck(cusparseDestroyMatDescr(descr));
    cudaErrCheck(cudaFree(d_rows));
    cudaErrCheck(cudaFree(d_columns));
    cudaErrCheck(cudaFree(d_csr_offsets));
    cudaErrCheck(cudaFree(d_csr_columns));
    cudaErrCheck(cudaFree(d_csr_values));
    cudaErrCheck(cudaFree(dBuffer));
  };
  Vector_GPU operator*(Vector_GPU &v); // Multiplication
  Vector_CUBLAS operator*(Vector_CUBLAS &v);
  MatrixCUSPARSE transpose();
  int getRows() { return h_rows; };
  int getColumns() { return h_columns; };
  double Dnrm2();
};