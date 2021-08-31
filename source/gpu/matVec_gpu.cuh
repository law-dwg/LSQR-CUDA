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
  /** Constructors/Destructors */
  Vector_GPU(){};                                                        // Default Constructor
  Vector_GPU(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) { // Constructor #1
    // printf("Vector_GPU Constructor #1 was called\n");
    // allocate to device
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_mat, sizeof(double) * r * c);
    // copy to device
    cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice);
  };
  Vector_GPU(unsigned int r, unsigned int c, double *m) : Vector_GPU(r, c) { // Constructor #2
    // printf("Vector_GPU Constructor #2 was called\n");
    cudaMemcpy(d_mat, m, sizeof(double) * r * c, cudaMemcpyHostToDevice);
  };
  Vector_GPU(const Vector_GPU &v) : Vector_GPU(v.h_rows, v.h_columns) { // Copy constructor
    printf("Vector_HPU Copy Constructor was called\n");
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
  };
  ~Vector_GPU() { // Destructor
    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_columns);
  };

  /** Operator overloads */
  Vector_GPU operator*(Vector_GPU &v);        // Multiplication
  Vector_GPU operator*(double i);             // Scale
  Vector_GPU &operator=(const Vector_GPU &v); // Assignment GPU
  const Vector_GPU &operator=(Vector_CPU &v) const {
    unsigned int r = v.getRows();
    unsigned int c = v.getColumns();
    double *m = v.getHMat();
    const Vector_GPU V1(r, c, m);
    return V1;
  };                                         // Assignment CPU
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
  }; /*
double Dnrm2();
double normalNorm();*/
  Vector_GPU transpose();
};

// create matrix class for readability / sparsity attribute
class Matrix_GPU : public Vector_GPU {
public:
  Matrix_GPU(unsigned int r, unsigned int c) : Vector_GPU(r, c){};
  Matrix_GPU(unsigned int r, unsigned int c, double *m) : Vector_GPU(r, c, m){};
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
  Matrix_GPU(const Matrix_GPU &v) : Vector_GPU(v){};
  Matrix_GPU(const Vector_GPU &v) : Vector_GPU(v){};
  ~Matrix_GPU() { // Destructor
    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_columns);
  };
  using Vector_GPU::operator=;
  Matrix_GPU &operator=(const Matrix_GPU rhs) {
    Vector_GPU::operator=(rhs);
    return *this;
  };
};

class Matrix_CSR_GPU : public Vector_GPU {
  double *d_vals;
  int *rowIdx, *colIdx;
};