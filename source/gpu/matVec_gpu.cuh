#pragma once
#include "../cpu/matVec_cpu.hpp"
#include "utils.cuh"
#include <algorithm>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

class Vector_GPU {
protected:
public:
  /** Attributes */
  unsigned int h_rows, h_columns, *d_rows, *d_columns;
  double *d_mat;
  /** Constructors/Destructors and rule of 5 */
  Vector_GPU() : h_rows(0), h_columns(0) { // Default Constructor
    // printf("Vector_GPU Default constructor called\n");
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double)));
    cudaErrCheck(cudaMemcpy(d_rows, &ZERO, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &ZERO, sizeof(unsigned int), cudaMemcpyHostToDevice));
  };
  Vector_GPU(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) { // Constructor #1
    // printf("Vector_GPU Constructor #1 was called\n");
    // allocate to device
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * r * c));
    // copy to device
    cudaErrCheck(cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemset(d_mat, ZERO, r * c * sizeof(double)));
  };
  Vector_GPU(unsigned int r, unsigned int c, double *m) : Vector_GPU(r, c) { // Constructor #2
    // printf("Vector_GPU Constructor #2 was called\n");
    cudaErrCheck(cudaMemcpy(d_mat, m, sizeof(double) * r * c, cudaMemcpyHostToDevice));
  };
  Vector_GPU(const Vector_GPU &v) : Vector_GPU(v.h_rows, v.h_columns) { // Copy constructor
    // printf("Vector_GPU Copy Constructor was called\n");
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
  };
  Vector_GPU(Vector_CPU &v)
      : Vector_GPU(v.getRows(), v.getColumns(), &v.mat[0]){
            // Copy constructor from CPU
            // printf("Vector_GPU/CPU Copy Constructor was called\n");
        };
  Vector_GPU &operator=(const Vector_GPU &v) { // Copy assignment operator
    printf("Vector_GPU Copy assignment operator was called\n");
    cudaErrCheck(cudaFree(this->d_mat));
    this->h_rows = v.h_rows;
    this->h_columns = v.h_columns;
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    cudaErrCheck(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    return *this;
  };
  Vector_GPU(Vector_GPU &&v) noexcept : h_rows(v.h_rows), h_columns(v.h_columns) { // Move constructor
    printf("Vector_CPU Move Constructor was called\n");
    cudaErrCheck(cudaMalloc((void **)&d_rows, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_columns, sizeof(unsigned int)));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    cudaErrCheck(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaFree(v.d_mat)); // free old memor)y
    v.h_rows = 0;
    v.h_columns = 0;
    double temp[0]; // set old to 0, it will be freed in destructor
    cudaErrCheck(cudaMalloc((void **)&v.d_mat, sizeof(double)));
    cudaErrCheck(cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice));
  };
  Vector_GPU &operator=(Vector_GPU &&v) noexcept { // Move assignment operator
    // printf("Vector_GPU Move assignment operator was called\n");
    // Host
    h_rows = v.h_rows;
    h_columns = v.h_columns;
    // Device
    cudaErrCheck(cudaFree(d_mat));
    cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns));
    cudaErrCheck(cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaFree(v.d_mat)); // free old memor)y
    v.h_rows = 0;
    v.h_columns = 0;
    double temp[0]; // set old to 0, it will be freed in destructor
    cudaErrCheck(cudaMalloc((void **)&v.d_mat, sizeof(double)));
    cudaErrCheck(cudaMemcpy(v.d_rows, &v.h_rows, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(v.d_columns, &v.h_columns, sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(v.d_mat, &temp, sizeof(double), cudaMemcpyHostToDevice));
    return *this;
  }

  ~Vector_GPU() { // Destructor
    // printf("DESTRUCTOR CALLED\n");
    cudaErrCheck(cudaFree(d_mat));
    cudaErrCheck(cudaFree(d_rows));
    cudaErrCheck(cudaFree(d_columns));
  };

  /** Operator overloads */
  Vector_GPU operator*(Vector_GPU &v);       // Multiplication
  Vector_GPU operator*(double i);            // Scale
  void operator=(Vector_CPU &v);             // Assignment CPU
  Vector_GPU operator-(const Vector_GPU &v); // Subtraction
  Vector_GPU operator+(const Vector_GPU &v); // Addittion

  /** Member functions */
  void printmat();
  Vector_CPU matDeviceToHost();
  Vector_GPU multNai(Vector_GPU &v);
  // Vector_GPU mulTiled(Vector_GPU &v);
  int getRows() { return h_rows; };
  int getColumns() { return h_columns; };
  double Dnrm2();
  Vector_GPU transpose();
};

// matrix class for readability
class Matrix_GPU : public Vector_GPU {
  using Vector_GPU::Vector_GPU;
  Matrix_GPU(const Vector_GPU &a) : Vector_GPU(a) { std::cout << "Mat_GPU conversion\n"; }
};
