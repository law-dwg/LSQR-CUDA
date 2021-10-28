#pragma once
#include "../cpu/vectorCPU.hpp"
#include "kernels.cuh"
#include "utils.cuh"

class VectorGPU {
protected:
  /** Attributes */
  unsigned h_rows, h_columns, *d_rows, *d_columns;
  double *d_mat;

public:
  /** Constructors */
  VectorGPU() : h_rows(0), h_columns(0) { // Default Constr.
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double)));
    cudaErrCheck(cudaMemcpy(d_rows, &ZERO, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &ZERO, sizeof(unsigned), cudaMemcpyHostToDevice));
  };
  VectorGPU(unsigned r, unsigned c) : h_rows(r), h_columns(c) { // Constr. #1
    // allocate to device
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * r * c));
    // copy to device
    cudaErrCheck(cudaMemcpy(d_rows, &r, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &c, sizeof(unsigned), cudaMemcpyHostToDevice));
  };
  VectorGPU(unsigned r, unsigned c, double *m) : VectorGPU(r, c) { // Constr. #2
    cudaErrCheck(cudaMemcpy(d_mat, m, sizeof(double) * r * c, cudaMemcpyHostToDevice));
  };
  VectorGPU(const VectorGPU &v) : h_rows(v.h_rows), h_columns(v.h_columns) { // Copy Constr.
    // allocate to device
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    // copy to device
    cudaErrCheck(cudaMemcpy(d_rows, &v.h_rows, sizeof(unsigned), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &v.h_columns, sizeof(unsigned), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
  };
  VectorGPU(VectorGPU &&v) noexcept : VectorGPU(v) { // Move Constr.
    // free old memory
    cudaErrCheck(cudaFree(v.d_mat));
    v.h_rows = 0;
    v.h_columns = 0;
    double temp[0]; // set old to 0, it will be freed in destructor
    cudaErrCheck(cudaMalloc((void **)&v.d_mat, sizeof(double)));
    cudaErrCheck(cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice));
  };

  VectorGPU(VectorCPU &v) : VectorGPU(v.getRows(), v.getColumns(), v.getMat()){}; // Copy Constr.: VectorCPU -> VectorGPU

  /** Destructor */
  ~VectorGPU() {
    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_columns);
  };

  /** Assignments */
  VectorGPU &operator=(const VectorGPU &v) { // Copy Assignment
    // free + memory allocation (if needed) - checks to reduce cuda calls
    if (h_rows * h_columns != v.h_rows * v.h_columns) {
      cudaErrCheck(cudaFree(d_mat));
      cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    }
    if (h_rows != v.h_rows) {
      h_rows = v.h_rows;
      cudaErrCheck(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned), cudaMemcpyDeviceToDevice));
    }
    if (h_columns != v.h_columns) {
      h_columns = v.h_columns;
      cudaErrCheck(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned), cudaMemcpyDeviceToDevice));
    }
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    return *this;
  };
  VectorGPU &operator=(VectorGPU &&v) noexcept { // Move Assignment
    // call copy assignment
    *this = v;
    v.h_rows = ZERO;
    v.h_columns = ZERO;
    // freeing memory handled by destructor. Potential operation errors blocked via rows = cols = 0
    return *this;
  }

  /** Member functions */
  int getRows() { return h_rows; };
  int getColumns() { return h_columns; };
  double *getMat() { return d_mat; };

  /** Virtual members */
  virtual void operator=(VectorCPU &v) = 0; // Copy Assignment: VectorCPU -> VectorGPU
  virtual void printMat() = 0;
  virtual VectorCPU matDeviceToHost() = 0;
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
  void operator=(VectorCPU &v);              // CopyToDevice

  /** Member Functions */
  VectorCUDA transpose();      // Transpose
  void printMat();             // PrintKernel
  VectorCPU matDeviceToHost(); // CopyToHost
  double Dnrm2();              // EuclideanNorm
};