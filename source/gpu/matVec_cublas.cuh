#pragma once
#include "../cpu/matVec_cpu.hpp"
#include "cublas_v2.h"
#include <algorithm>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

extern cudaError_t cudaStat;
extern cublasStatus_t stat;
extern cublasHandle_t handle;
extern cudaStream_t stream;
extern cublasStatus_t stat1;
extern cudaError_t cudaStat1;
extern cublasStatus_t stat2;
void extern cublasReset();

class Vector_CUBLAS {
protected:
public:
  /** Attributes */
  unsigned int h_rows, h_columns, *d_rows, *d_columns;
  double *d_mat;
  /** Constructors/Destructors and rule of 5 */
  Vector_CUBLAS()
      : h_rows(0), h_columns(0){
                       // Default Constructor

                   };
  Vector_CUBLAS(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) { // Constructor #1
    // printf("Vector_CUBLAS Constructor #1 was called\n");
    // allocate to device
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_mat, sizeof(double) * r * c);
    // copy to device
    cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice);
  };
  Vector_CUBLAS(unsigned int r, unsigned int c, double *m) : Vector_CUBLAS(r, c) { // Constructor #2
    // printf("Vector_CUBLAS Constructor #2 was called\n");
    cudaMemcpy(d_mat, m, sizeof(double) * r * c, cudaMemcpyHostToDevice);
  };
  Vector_CUBLAS(const Vector_CUBLAS &v) : Vector_CUBLAS(v.h_rows, v.h_columns) { // Copy constructor
    // printf("Vector_CUBLAS Copy Constructor was called\n");
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
  };
  Vector_CUBLAS(Vector_CPU &v)
      : Vector_CUBLAS(v.getRows(), v.getColumns(), &v.mat[0]){
            // Copy constructor from CPU
            // printf("Vector_CUBLAS/CPU Copy Constructor was called\n");
        };
  Vector_CUBLAS &operator=(const Vector_CUBLAS &v) { // Copy assignment operator
    printf("Vector_CUBLAS Copy assignment operator was called\n");
    cudaFree(this->d_mat);
    this->h_rows = v.h_rows;
    this->h_columns = v.h_columns;
    cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns);
    cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
    return *this;
  };
  Vector_CUBLAS(Vector_CUBLAS &&v) noexcept : h_rows(v.h_rows), h_columns(v.h_columns) { // Move constructor
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
  Vector_CUBLAS &operator=(Vector_CUBLAS &&v) noexcept { // Move assignment operator
    // printf("Vector_CUBLAS Move assignment operator was called\n");
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
  Vector_CUBLAS multNai(Vector_CUBLAS &v);
  // Vector_CUBLAS mulTiled(Vector_CUBLAS &v);
  int getRows() { return h_rows; };
  int getColumns() { return h_columns; };
  double Dnrm2();
  double Dnrm2Leg();
  Vector_CUBLAS transpose();
};

// matrix class for readability
class Matrix_GPU : public Vector_CUBLAS {
  using Vector_CUBLAS::Vector_CUBLAS;
  Matrix_GPU(const Vector_CUBLAS &a) : Vector_CUBLAS(a) { std::cout << "Mat_GPU conversion\n"; }
};
