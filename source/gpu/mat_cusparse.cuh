#pragma once
#include "../cpu/matVec_cpu.hpp"
#include "matVec_cublas.cuh"
#include "matVec_gpu.cuh"
#include "utils.cuh"
#include <algorithm>
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

class Matrix_cuSPARSE {
protected:
public:
  unsigned int h_rows, h_columns, *d_rows, *d_columns;
  int *d_csr_offsets, *d_csr_columns;
  int64_t h_nnz;
  double *d_csr_values;
  cusparseSpMatDescr_t d_matDescr_sparse;
  void *dBuffer = NULL;
  size_t bufferSize = 0;
  Matrix_cuSPARSE(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) {
    printf("CALLED\n");
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    cudaErrCheck(cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc((void **)&d_csr_offsets, (r + 1) * sizeof(int)));
    cudaErrCheck(cudaMalloc(&dBuffer, bufferSize));
  };
  Matrix_cuSPARSE(unsigned int r, unsigned int c, double *m) : h_rows(r), h_columns(c) {
    cusparseDnMatDescr_t d_matDescr_dense;
    double *d_mat;
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, r * c * sizeof(double)));
    cudaErrCheck(cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, m, r * c * sizeof(double), cudaMemcpyHostToDevice));
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
    printf("Sparse matrix created, rows=%d, cols=%d, nnz=%d\n", rows_tmp, cols_tmp, h_nnz);
    cudaErrCheck(cudaMalloc((void **)&d_csr_columns, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&d_csr_values, h_nnz * sizeof(double)));

    cusparseErrCheck(cusparseCsrSetPointers(d_matDescr_sparse, d_csr_offsets, d_csr_columns, d_csr_values));
    cusparseErrCheck(cusparseDenseToSparse_convert(spHandle, d_matDescr_dense, d_matDescr_sparse, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));
    cusparseDestroyDnMat(d_matDescr_dense);
    cudaFree(d_mat);
  };
  Matrix_cuSPARSE(const Matrix_cuSPARSE &v) : Matrix_cuSPARSE(v.h_rows, v.h_columns) { // Copy constructor
    // printf("Vector_CUBLAS Copy Constructor was called\n");
    h_nnz = v.h_nnz;
    cudaErrCheck(cudaMalloc((void **)&d_csr_columns, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&d_csr_values, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMemcpy(d_csr_columns, v.d_csr_columns, h_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_values, v.d_csr_values, h_nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csr_offsets, v.d_csr_offsets, (v.h_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    cusparseErrCheck(cusparseCreateCsr(&d_matDescr_sparse, h_rows, h_columns, h_nnz, d_csr_offsets, d_csr_columns, d_csr_values, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  };
  ~Matrix_cuSPARSE() {
    cusparseErrCheck(cusparseDestroySpMat(d_matDescr_sparse));
    cudaErrCheck(cudaFree(d_rows));
    cudaErrCheck(cudaFree(d_columns));
    cudaErrCheck(cudaFree(d_csr_offsets));
    cudaErrCheck(cudaFree(d_csr_columns));
    cudaErrCheck(cudaFree(d_csr_values));
    cudaErrCheck(cudaFree(dBuffer));
  };
  Vector_GPU operator*(Vector_GPU &v); // Multiplication
  Vector_CUBLAS operator*(Vector_CUBLAS &v);
  Matrix_cuSPARSE transpose();
  int getRows() { return h_rows; };
  int getColumns() { return h_columns; };
  double Dnrm2();
};