#pragma once
#include "../cpu/matVec_cpu.h"
#include "matVec_gpu.cuh"
#include <algorithm>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

class Matrix_CSR_GPU {
public:
  unsigned int *d_rowIdx, *d_colIdx, *d_rows, *d_columns, *d_nnz, h_nnz, h_rows, h_columns;
  double *d_vals;
  Matrix_CSR_GPU() : h_columns(0), h_rows(0) { // Default Constructor
    printf("Matrix_CSR_GPU Default constructor called\n");
    cudaMalloc((void **)&d_rowIdx, sizeof(unsigned int));
    cudaMalloc((void **)&d_colIdx, sizeof(unsigned int));
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_nnz, sizeof(unsigned int));
    cudaMalloc((void **)&d_vals, sizeof(double));
    cudaMemcpy(d_rows, &h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice);
  };
  Matrix_CSR_GPU(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) { // Constructor #1
    printf("Matrix_CSR_GPU Constructor #1 was called\n");
    // allocate
    cudaMalloc((void **)&d_rowIdx, sizeof(unsigned int));
    cudaMalloc((void **)&d_colIdx, sizeof(unsigned int));
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_nnz, sizeof(unsigned int));
    cudaMalloc((void **)&d_vals, sizeof(double));
    // copy to device
    cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice);
  };
  Matrix_CSR_GPU(unsigned int r, unsigned int c, double *m) : h_rows(r), h_columns(c), h_nnz(0) { // Constructor #2
    printf("Matrix_CSR_GPU Constructor #2 was called\n");
    int row, col;
    col = row = 0;
    std::vector<int> temp_rowPtr, temp_colIdx;
    temp_rowPtr.push_back(0);
    std::vector<double> temp_vals;
    for (int i = 0; i < r * c; ++i) {
      if (((int)(i / c)) > row) {
        printf("i=%d, h_nnz=%d, row=%d\n", i, h_nnz, row);
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
    for (int i = 0; i < temp_rowPtr.size(); ++i) {
      printf("rowPtr[%d]=%d\n", i, temp_rowPtr[i]);
    }
    for (int i = 0; i < temp_colIdx.size(); ++i) {
      printf("temp_colIdx[%d]=%d\n", i, temp_colIdx[i]);
    }
    for (int i = 0; i < temp_vals.size(); ++i) {
      printf("temp_vals[%d]=%f\n", i, temp_vals[i]);
    }
    // allocate
    cudaMalloc((void **)&d_nnz, sizeof(unsigned int));
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_rowIdx, sizeof(unsigned int) * (r + 1));
    cudaMalloc((void **)&d_colIdx, sizeof(unsigned int) * h_nnz);
    cudaMalloc((void **)&d_vals, sizeof(double) * h_nnz);
    // copy to device
    cudaMemcpy(d_nnz, &h_nnz, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rows, &h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowIdx, temp_rowPtr.data(), sizeof(unsigned int) * (r + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, temp_colIdx.data(), sizeof(unsigned int) * h_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals, temp_vals.data(), sizeof(double) * h_nnz, cudaMemcpyHostToDevice);
    printf("nnz = %d, rowIdx.size = %d, colIdx.size = %d\n", h_nnz, temp_rowPtr.size(), temp_colIdx.size());
  };
  /*
  Matrix_CSR_GPU(const Matrix_CSR_GPU &v) : Matrix_CSR_GPU(v.h_rows, v.h_columns) { // Copy constructor
    printf("Matrix_CSR_GPU Copy Constructor was called\n");
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
  };
  Matrix_CSR_GPU(Vector_CPU &v) : Matrix_CSR_GPU(v.getRows(), v.getColumns(), &v.mat[0]) { // Copy constructor from CPU
    printf("Matrix_CSR_GPU/CPU Copy Constructor was called\n");
  };
  Matrix_CSR_GPU &operator=(const Matrix_CSR_GPU &v) { // Copy assignment operator
    printf("Matrix_CSR_GPU Copy assignment operator was called\n");
    cudaFree(this->d_mat);
    this->h_rows = v.h_rows;
    this->h_columns = v.h_columns;
    cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns);
    cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
    return *this;
  };
  Matrix_CSR_GPU(Matrix_CSR_GPU &&v) noexcept : h_rows(v.h_rows), h_columns(v.h_columns) { // Move constructor
    printf("Vector_CPU Move Constructor was called\n");
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns);
    cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
    cudaFree(v.d_mat); // free old memory
    v.h_rows = 0;
    v.h_columns = 0;
    double temp[0]; // set old to 0, it will be freed in destructor
    cudaMalloc((void **)&v.d_mat, sizeof(double));
    cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice);
  };
  Matrix_CSR_GPU &operator=(Matrix_CSR_GPU &&v) noexcept { // Move assignment operator
    printf("Matrix_CSR_GPU Move assignment operator was called\n");
    // Host
    h_rows = v.h_rows;
    h_columns = v.h_columns;
    // Device
    cudaFree(d_mat);
    cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns);
    cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
    cudaFree(v.d_mat); // free old memory
    v.h_rows = 0;
    v.h_columns = 0;
    double temp[0]; // set old to 0, it will be freed in destructor
    cudaMalloc((void **)&v.d_mat, sizeof(double));
    cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice);
    return *this;
  }
  */
  ~Matrix_CSR_GPU() { // Destructor
    printf("DESTRUCTOR CALLED\n");
    cudaFree(d_rows);
    cudaFree(d_columns);
    cudaFree(d_colIdx);
    cudaFree(d_rowIdx);
    cudaFree(d_vals);
  };
  void operator*(Vector_GPU &v); // Multiplication
};