#pragma once
#include "../cpu/matVec_cpu.hpp"
#include "Kernels.cuh"
#include "VectorCUDA.cuh"
#include <algorithm>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

class MatrixGPU {
protected:
  unsigned h_rows, h_columns, h_nnz, *d_rows, *d_columns, *d_nnz;
  int *d_csrRowPtr, *d_csrColInd;
  double *d_csrVal;
  // cusparse used for transpose
  cusparseMatDescr_t descr = NULL;
  void *dBuffer = NULL;
  size_t bufferSize = 0;

public:
  /** Constructors */
  MatrixGPU(unsigned r, unsigned c) : h_rows(r), h_columns(c) { // Constructor #1
    printf("MatrixGPU Constructor #1 was called\n");
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_nnz, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_csrRowPtr, (h_rows + 1) * sizeof(int)));
    cudaErrCheck(cudaMemcpy(d_rows, &h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMalloc(&dBuffer, bufferSize));
    // cusparse - used for transpose
    cusparseErrCheck(cusparseCreateMatDescr(&descr));
    cusparseErrCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    cusparseErrCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
  };
  MatrixGPU(unsigned r, unsigned c, unsigned n) : MatrixGPU(r, c) { // Constructor helper #2
    h_nnz = n;
    printf("MatrixGPU Constructor #2 was called\n");
    // allocate
    cudaErrCheck(cudaMalloc((void **)&d_csrVal, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMalloc((void **)&d_csrColInd, h_nnz * sizeof(int)));
    // copy to device
    cudaErrCheck(cudaMemcpy(d_nnz, &h_nnz, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_rows, &h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
  };
  MatrixGPU() : MatrixGPU(0, 0, 0u) { // Default Constructor
    printf("MatrixGPU Default constructor called\n");
    cudaErrCheck(cudaMemset(d_csrVal, ZERO, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMemset(d_csrColInd, ZERO, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMemset(d_csrRowPtr, ZERO, (h_rows + 1) * sizeof(int)));
  };
  MatrixGPU(unsigned r, unsigned c, unsigned n, double *values, int *colInd, int *rowPtr) : MatrixGPU(r, c, n) { // Constructor #3
    printf("MatrixGPU Constructor #3 called\n");
    // copy to device
    cudaErrCheck(cudaMemcpy(d_csrVal, values, h_nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csrColInd, colInd, h_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csrRowPtr, rowPtr, (h_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
  }
  MatrixGPU(unsigned r, unsigned c, double *m) : MatrixGPU(r, c) { // Constructor (entry point)
    h_nnz = 0;
    printf("MatrixGPU Entry Point was called\n");
    int row, col;
    col = row = 0;
    std::vector<int> temp_rowPtr, temp_colIdx;
    temp_rowPtr.push_back(0);
    std::vector<double> temp_vals;
    for (int i = 0; i < r * c; ++i) {
      if (((int)(i / c)) > row) {
        // printf("i=%d, h_nnz=%d, row=%d\n", i, h_nnz, row);
        temp_rowPtr.push_back(h_nnz);
        row = i / c;
      }
      col = i - (row * c);
      if (m[i] > 1e-15) {
        h_nnz += 1;
        temp_colIdx.push_back(col);
        temp_vals.push_back(m[i]);
      }
    }
    temp_rowPtr.push_back(h_nnz);

    // allocate
    cudaErrCheck(cudaMalloc((void **)&d_csrColInd, sizeof(unsigned) * h_nnz));
    cudaErrCheck(cudaMalloc((void **)&d_csrVal, sizeof(double) * h_nnz));
    // copy to device
    cudaErrCheck(cudaMemcpy(d_nnz, &h_nnz, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_csrRowPtr, temp_rowPtr.data(), sizeof(unsigned) * (r + 1), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_csrColInd, temp_colIdx.data(), sizeof(unsigned) * h_nnz, cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_csrVal, temp_vals.data(), sizeof(double) * h_nnz, cudaMemcpyHostToDevice));
    printf("nnz = %d, rowIdx.size = %d, colIdx.size = %d\n", h_nnz, temp_rowPtr.size(), temp_colIdx.size());
  };
  MatrixGPU(const MatrixGPU &m) : MatrixGPU(m.h_rows, m.h_columns, m.h_nnz, m.d_csrVal, m.d_csrColInd, m.d_csrRowPtr) {
    printf("MatrixGPU Copy Constructor called\n");
  }; // Copy constructor
  MatrixGPU(MatrixGPU &&m) noexcept
      : MatrixGPU(m.h_rows, m.h_columns, m.h_nnz, m.d_csrVal, m.d_csrColInd, m.d_csrRowPtr) { // MatrixGPU Move Constructor
    printf("MatrixGPU Move Constructor called\n");
    // free old resources
    cudaErrCheck(cudaFree(m.d_csrVal));
    cudaErrCheck(cudaFree(m.d_csrRowPtr));
    cudaErrCheck(cudaFree(m.d_csrColInd));
    m.h_rows = ZERO;
    m.h_nnz = ZERO;
    m.h_columns = ZERO;
    cudaErrCheck(cudaMalloc((void **)&m.d_csrVal, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMalloc((void **)&m.d_csrColInd, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&m.d_csrRowPtr, (h_rows + 1) * sizeof(int)));
    cudaErrCheck(cudaMemcpy(m.d_rows, &m.h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(m.d_columns, &m.h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(m.d_nnz, &m.h_nnz, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemset(m.d_csrVal, ZERO, m.h_nnz * sizeof(double)));
    cudaErrCheck(cudaMemset(m.d_csrColInd, ZERO, m.h_nnz * sizeof(int)));
    cudaErrCheck(cudaMemset(m.d_csrRowPtr, ZERO, (m.h_rows + 1) * sizeof(int)));
  };

  /** Destructor */
  ~MatrixGPU() { // Destructor
    printf("MatrixGPU DESTRUCTOR CALLED\n");
    cusparseErrCheck(cusparseDestroyMatDescr(descr));
    cudaErrCheck(cudaFree(dBuffer));
    cudaErrCheck(cudaFree(d_nnz));
    cudaErrCheck(cudaFree(d_rows));
    cudaErrCheck(cudaFree(d_columns));
    cudaErrCheck(cudaFree(d_csrColInd));
    cudaErrCheck(cudaFree(d_csrRowPtr));
    cudaErrCheck(cudaFree(d_csrVal));
  };

  /** Assignments */
  MatrixGPU &operator=(const MatrixGPU &m) { // Copy assignment operator
    printf("MatrixGPU Copy Assignment Operator called\n");
    // free + memory allocation (if needed)
    if (h_rows != m.h_rows) {
      h_rows = m.h_rows;
      cudaErrCheck(cudaMemcpy(d_rows, m.d_rows, sizeof(unsigned), cudaMemcpyDeviceToDevice));
      cudaErrCheck(cudaFree(d_csrRowPtr));
      cudaErrCheck(cudaMalloc((void **)&d_csrRowPtr, sizeof(int) * (m.h_rows + 1)));
    }
    if (h_nnz != m.h_nnz) {
      h_nnz = m.h_nnz;
      cudaErrCheck(cudaMemcpy(d_nnz, m.d_nnz, sizeof(unsigned), cudaMemcpyDeviceToDevice));
      cudaErrCheck(cudaFree(d_csrVal));
      cudaErrCheck(cudaFree(d_csrColInd));
      cudaErrCheck(cudaMalloc((void **)&d_csrVal, sizeof(double) * h_nnz));
      cudaErrCheck(cudaMalloc((void **)&d_csrColInd, sizeof(int) * h_nnz));
    }
    if (h_columns != m.h_columns) {
      h_columns = m.h_columns;
      cudaErrCheck(cudaMemcpy(d_columns, m.d_columns, sizeof(unsigned), cudaMemcpyDeviceToDevice));
    }

    // copy to device
    cudaErrCheck(cudaMemcpy(d_csrVal, m.d_csrVal, h_nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csrColInd, m.d_csrColInd, h_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csrRowPtr, m.d_csrRowPtr, (h_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    return *this;
  };

  MatrixGPU &operator=(MatrixGPU &&m) noexcept { // Move assignment operator
    printf("MatrixGPU Move Assignment called\n");
    // call copy assignment
    *this = m;
    // freeing memory handled by destructor (unless std::move is called explicitly)
    return *this;
  };

  // VectorCUDA operator*(VectorCUDA &v); // Multiplication
  // MatrixGPU transpose();
  int getRows() { return h_rows; };
  int getColumns() { return h_columns; };
  int getNnz() { return h_nnz; };
  virtual double Dnrm2() = 0;
};

class MatrixCUDA : public MatrixGPU {
public:
  /** Inherit everything */
  using MatrixGPU::MatrixGPU;

  /** Operator overloads */
  VectorCUDA operator*(VectorCUDA &v); // SpMV

  /** Member Functions */
  MatrixCUDA transpose(); // Transpose
  double Dnrm2();         // EuclideanNorm
};