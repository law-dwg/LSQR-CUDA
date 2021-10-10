#pragma once
#include "../cpu/matVec_cpu.hpp"
#include "utils.cuh"
#include <algorithm>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

class Vector_CUBLAS {
protected:
public:
  /** Attributes */
  unsigned int h_rows, h_columns, *d_rows, *d_columns;
  double *d_mat;
  /** Constructors/Destructors and rule of 5 */
  Vector_CUBLAS() : h_rows(0), h_columns(0) { // Default Constructor
    gpuErrchk(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **)&d_mat, sizeof(double)));
    gpuErrchk(cudaMemcpy(d_rows, &ZERO, sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_columns, &ZERO, sizeof(unsigned int), cudaMemcpyHostToDevice));
  };
  Vector_CUBLAS(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) { // Constructor #1
    // printf("Vector_CUBLAS Constructor #1 was called\n");
    // allocate to device
    gpuErrchk(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **)&d_mat, sizeof(double) * r * c));
    // copy to device
    gpuErrchk(cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_mat, ZERO, r * c * sizeof(double)));
  };
  Vector_CUBLAS(unsigned int r, unsigned int c, double *m) : Vector_CUBLAS(r, c) { // Constructor #2
    // printf("Vector_CUBLAS Constructor #2 was called\n");
    gpuErrchk(cudaMemcpy(d_mat, m, sizeof(double) * r * c, cudaMemcpyHostToDevice));
  };
  Vector_CUBLAS(const Vector_CUBLAS &v) : Vector_CUBLAS(v.h_rows, v.h_columns) { // Copy constructor
    printf("Vector_CUBLAS Copy Constructor was called\n");
    gpuErrchk(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
  };
  Vector_CUBLAS(Vector_CPU &v) : Vector_CUBLAS(v.getRows(), v.getColumns(), &v.mat[0]) {
    // Copy constructor from CPU
    printf("Vector_CUBLAS/CPU Copy Constructor was called\n");
  };
  Vector_CUBLAS &operator=(const Vector_CUBLAS &v) { // Copy assignment operator
    printf("Vector_CUBLAS Copy assignment operator was called\n");
    gpuErrchk(cudaFree(this->d_mat));
    this->h_rows = v.h_rows;
    this->h_columns = v.h_columns;
    gpuErrchk(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    gpuErrchk(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    return *this;
  };
  Vector_CUBLAS(Vector_CUBLAS &&v) noexcept : h_rows(v.h_rows), h_columns(v.h_columns) { // Move constructor
    printf("Vector_CPU Move Constructor was called\n");
    gpuErrchk(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    gpuErrchk(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaFree(v.d_mat)); // free old memory
    v.h_rows = 0;
    v.h_columns = 0;
    double temp[0]; // set old to 0, it will be freed in destructor
    gpuErrchk(cudaMalloc((void **)&v.d_mat, sizeof(double)));
    gpuErrchk(cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice));
  };
  Vector_CUBLAS &operator=(Vector_CUBLAS &&v) noexcept { // Move assignment operator
    // printf("Vector_CUBLAS Move assignment operator was called\n");
    // Host
    h_rows = v.h_rows;
    h_columns = v.h_columns;
    // Device
    gpuErrchk(cudaFree(d_mat));
    gpuErrchk(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    gpuErrchk(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaFree(v.d_mat)); // free old memory
    v.h_rows = 0;
    v.h_columns = 0;
    double temp[0]; // set old to 0, it will be freed in destructor
    gpuErrchk(cudaMalloc((void **)&v.d_mat, sizeof(double)));
    gpuErrchk(cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice));
    return *this;
  }

  ~Vector_CUBLAS() { // Destructor
    // printf("DESTRUCTOR CALLED\n");
    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_columns);
  };

  /** Operator overloads */
  Vector_CUBLAS operator*(Vector_CUBLAS &v);       // Multiplication
  Vector_CUBLAS operator*(double i);               // Scale
  void operator=(Vector_CPU &v);                   // Assignment CPU
  Vector_CUBLAS operator-(const Vector_CUBLAS &v); // Subtraction
  Vector_CUBLAS operator+(const Vector_CUBLAS &v); // Addittion

  /** Member functions */
  void printmat();
  Vector_CPU matDeviceToHost();
  int getRows() { return h_rows; };
  int getColumns() { return h_columns; };
  double Dnrm2();
  Vector_CUBLAS transpose();
};