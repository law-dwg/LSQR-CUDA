#pragma once
#include "../cpu/matVec_cpu.hpp"
#include "utils.cuh"
#include <algorithm>
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

class VectorGPU {
protected:
public:
  /** Attributes */
  unsigned int h_rows, h_columns, *d_rows, *d_columns;
  double *d_mat;
  /** Constructors/Destructors and rule of 5 */
  VectorGPU() : h_rows(0), h_columns(0) { // Default Constructor
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double)));
    cudaErrCheck(cudaMemcpy(d_rows, &ZERO, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &ZERO, sizeof(unsigned int), cudaMemcpyHostToDevice));
  };
  VectorGPU(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) { // Constructor #1
    // printf("VectorGPU Constructor #1 was called\n");
    // allocate to device
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * r * c));
    // copy to device
    cudaErrCheck(cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemset(d_mat, ZERO, r * c * sizeof(double)));
  };
  VectorGPU(unsigned int r, unsigned int c, double *m) : VectorGPU(r, c) { // Constructor #2
    // printf("VectorGPU Constructor #2 was called\n");
    cudaErrCheck(cudaMemcpy(d_mat, m, sizeof(double) * r * c, cudaMemcpyHostToDevice));
  };
  VectorGPU(const VectorGPU &v) : VectorGPU(v.h_rows, v.h_columns) { // Copy constructor
    printf("VectorGPU Copy Constructor was called\n");
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
  };
  VectorGPU(Vector_CPU &v) : VectorGPU(v.getRows(), v.getColumns(), &v.mat[0]) {
    // Copy constructor from CPU
    printf("VectorGPU/CPU Copy Constructor was called\n");
  };
  VectorGPU &operator=(const VectorGPU &v) { // Copy assignment operator
    printf("VectorGPU Copy assignment operator was called\n");
    if (this->h_rows * this->h_columns != v.h_rows * v.h_columns) {
      cudaErrCheck(cudaFree(this->d_mat));
      cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    }
    this->h_rows = v.h_rows;
    this->h_columns = v.h_columns;
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    cudaErrCheck(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    return *this;
  };
  VectorGPU(VectorGPU &&v) noexcept : h_rows(v.h_rows), h_columns(v.h_columns) { // Move constructor
    printf("Vector_CPU Move Constructor was called\n");
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    cudaErrCheck(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaFree(v.d_mat)); // free old memory
    v.h_rows = 0;
    v.h_columns = 0;
    double temp[0]; // set old to 0, it will be freed in destructor
    cudaErrCheck(cudaMalloc((void **)&v.d_mat, sizeof(double)));
    cudaErrCheck(cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice));
  };
  VectorGPU &operator=(VectorGPU &&v) noexcept { // Move assignment operator
    // printf("VectorGPU Move assignment operator was called\n");
    // Host
    if (this->h_rows * this->h_columns != v.h_rows * v.h_columns) {
      cudaErrCheck(cudaFree(this->d_mat));
      cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    }
    h_rows = v.h_rows;
    h_columns = v.h_columns;
    // Device
    cudaErrCheck(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    // HANDLED BY DESTRUCTOR
    // cudaErrCheck(cudaFree(v.d_mat)); // free old memory
    // v.h_rows = 0;
    // v.h_columns = 0;
    // double temp[0]; // set old to 0, it will be freed in destructor
    // cudaErrCheck(cudaMalloc((void **)&v.d_mat, sizeof(double)));
    // cudaErrCheck(cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice));
    // cudaErrCheck(cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice));
    // cudaErrCheck(cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice));
    return *this;
  }
  ~VectorGPU() { // Destructor
    // printf("DESTRUCTOR CALLED\n");
    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_columns);
  };

  /** Member functions */
  int getRows() { return h_rows; };
  int getColumns() { return h_columns; };

  /** Virtual members */
  virtual void operator=(Vector_CPU &v) = 0;
  virtual void printmat() = 0;
  virtual Vector_CPU matDeviceToHost() = 0;
  virtual double Dnrm2() = 0;
};

class VectorCUDA : public VectorGPU {
public:
  /** Inherit everything */
  using VectorGPU::VectorGPU;

  /** Operator overloads */
  VectorCUDA operator*(VectorCUDA &v);       // Multiplication
  VectorCUDA operator*(double i);            // Scale
  VectorCUDA operator-(const VectorCUDA &v); // Subtraction
  VectorCUDA operator+(const VectorCUDA &v); // Addittion
  void operator=(Vector_CPU &v);             // CopyToDevice

  /** Member Functions */
  VectorCUDA transpose();       // Transpose
  void printmat();              // PrintKernel
  Vector_CPU matDeviceToHost(); // CopyToHost
  double Dnrm2();               // EuclideanNorm
};