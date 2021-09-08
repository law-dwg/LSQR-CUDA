#pragma once
#include <algorithm>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> /* srand, rand */
#include <time.h>   /* time */
#include <vector>

#include "../cpu/matVec_cpu.h"
class Vector_GPU {
protected:
  /** Attributes */
  unsigned int h_rows, h_columns;
  unsigned int *d_rows, *d_columns;
  double *d_mat;

public:
  /** Constructors/Destructors and rule of 5 */
  Vector_GPU() { // Default Constructor
    printf("Vector_GPU Default constructor called\n");
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_mat, sizeof(double));
  };
  Vector_GPU(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) { // Constructor #1
    printf("Vector_GPU Constructor #1 was called\n");
    // allocate to device
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_mat, sizeof(double) * r * c);
    // copy to device
    cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice);
  };
  Vector_GPU(unsigned int r, unsigned int c, double *m) : Vector_GPU(r, c) { // Constructor #2
    printf("Vector_GPU Constructor #2 was called\n");
    cudaMemcpy(d_mat, m, sizeof(double) * r * c, cudaMemcpyHostToDevice);
  };
  Vector_GPU(const Vector_GPU &v) : Vector_GPU(v.h_rows, v.h_columns) { // Copy constructor
    printf("Vector_GPU Copy Constructor was called\n");
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
  };
  Vector_GPU(Vector_CPU &v) : Vector_GPU(v.getRows(), v.getColumns(), &v.mat[0]) { // Copy constructor from CPU
    printf("Vector_GPU/CPU Copy Constructor was called\n");
  };
  Vector_GPU &operator=(const Vector_GPU &v) { // Copy assignment operator
    printf("Vector_GPU Copy assignment operator was called\n");
    cudaFree(this->d_mat);
    this->h_rows = v.h_rows;
    this->h_columns = v.h_columns;
    cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns);
    cudaMemcpy(d_rows, v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_columns, v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
    return *this;
  };
  Vector_GPU(Vector_GPU &&v) noexcept : h_rows(v.h_rows), h_columns(v.h_columns) { // Move constructor
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
  Vector_GPU &operator=(Vector_GPU &&v) noexcept { // Move assignment operator
    printf("Vector_GPU Move assignment operator was called\n");
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

  ~Vector_GPU() { // Destructor
    printf("DESTRUCTOR CALLED\n");
    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_columns);
  };

  /** Operator overloads */

  Vector_GPU operator*(Vector_GPU &v);       // Multiplication
  Vector_GPU operator*(double i);            // Scale
  void operator=(Vector_CPU &v);             // Assignment CPU
  Vector_GPU operator-(const Vector_GPU &v); // Subtraction
  Vector_GPU operator+(const Vector_GPU &v); // Addittion
  /*
  double operator()(unsigned int i);
  double operator()(unsigned int r, unsigned int c);
  double operator[](unsigned int i);
  */
  /** Member functions */
  void printmat();
  Vector_CPU matDeviceToHost();
  Vector_GPU multNai(Vector_GPU &v);
  int getRows() {
    printf("number of rows: %i\n", h_rows);
    return h_rows;
  };
  int getColumns() {
    printf("number of columns: %i\n", h_columns);
    return h_columns;
  };
  double Dnrm2();
  Vector_GPU transpose();
};

// create matrix class for readability / sparsity attribute
class Matrix_GPU : public Vector_GPU {
public:
  using Vector_GPU::Vector_GPU; // inherit everything
  Matrix_GPU(unsigned int r, unsigned int c, double sparsity) : Vector_GPU(r, c) {
    int zeros = round(sparsity * r * c);
    int nonZeros = r * c - zeros;
    srand(time(0));
    std::vector<double> mat;
    mat.resize(h_rows * h_columns);
    for (int i = 0; i < mat.size(); i++) {
      if (i < nonZeros) {
        mat[i] = rand() % 100 + 1;
      } else {
        mat[i] = 0;
      };
    }
    std::random_shuffle(mat.begin(), mat.end());
    cudaMemcpy(d_mat, &mat[0], sizeof(double) * c * r, cudaMemcpyHostToDevice);
  };
  Matrix_GPU &operator=(const Matrix_GPU rhs) {
    Vector_GPU::operator=(rhs);
    return *this;
  };
  Matrix_GPU(const Vector_GPU &v) : Vector_GPU(v){};
};

class Matrix_CSR_GPU : public Vector_GPU {
  double *d_vals;
  int *rowIdx, *colIdx;
};