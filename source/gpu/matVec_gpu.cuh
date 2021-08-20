#pragma once
#include <stdio.h>  //NULL, printf

#include "../cpu/matVec_cpu.h"
class Vector_GPU {
 public:
  ////////////////////////////////
  // Attributes
  ////////////////////////////////
  unsigned int h_rows, h_columns;
  unsigned int *d_rows, *d_columns;
  double *d_mat;
  ////////////////////////////////
  // Constructors/Destructors
  ////////////////////////////////
  Vector_GPU(){};
  Vector_GPU(unsigned int r, unsigned int c, double *m) : h_rows(r), h_columns(c) {
    // allocate on device
    printf("CONSTRUCTOR #1 CALLED\n");
    printf("%d x %d\n", h_rows, h_columns);
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_mat, sizeof(double) * r * c);
    // copy to device
    cudaMemcpy(d_rows, &r, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &c, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, m, sizeof(double) * r * c, cudaMemcpyHostToDevice);
  };
  // Vector_GPU(unsigned int p):columns(1),rows(p){this->mat.resize(p,5.0);};
  Vector_GPU(unsigned int r, unsigned int c) : h_rows(r), h_columns(c) {
    printf("CONSTRUCTOR #2 CALLED\n");
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_mat, sizeof(double) * r * c);

    cudaMemcpy(d_rows, &r, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, &c, sizeof(unsigned int), cudaMemcpyHostToDevice);
  };
  // Vector_GPU(unsigned int r, unsigned int c, double
  // v){this->rows=r;this->columns=c;mat.resize(r*c,v);}; copy constructor
  Vector_GPU(const Vector_GPU &v) : h_rows(v.h_rows), h_columns(v.h_columns) {  // Copy constructor
    printf("COPY CONSTRUCTOR INVOKED\n");
    cudaMalloc((void **)&d_rows, sizeof(unsigned int));
    cudaMalloc((void **)&d_columns, sizeof(unsigned int));
    cudaMalloc((void **)&d_mat, sizeof(double) * v.h_rows * v.h_columns);

    cudaMemcpy(d_rows, &v.d_rows, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_columns, &v.d_columns, sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_mat, v.d_mat, sizeof(double) * v.h_columns * v.h_rows, cudaMemcpyDeviceToDevice);
  };
  ~Vector_GPU() {  // Destructor
    cudaFree(d_mat);
    cudaFree(d_rows);
    cudaFree(d_columns);
  };

  ////////////////////////////////
  // Operator overloads
  ////////////////////////////////
  Vector_GPU operator*(Vector_GPU &v);         // Multiplication
  Vector_GPU operator*(double i);              // Scale
  Vector_GPU &operator=(const Vector_GPU &v);  // Assignment
  Vector_GPU operator-(const Vector_GPU &v);   // Subtraction
  Vector_GPU operator+(const Vector_GPU &v);   // Addittion
  /*double operator()(unsigned int i);
  double operator()(unsigned int r, unsigned int c);
  double operator[](unsigned int i);
  */
  ////////////////////////////////
  // Member functions
  ////////////////////////////////
  void printmat();
  Vector_CPU matDeviceToHost();
  /*
  std::vector<double> getMat(){return this->mat;};
  void print();*/
  int getRows();
  int getColumns(); /*
   double Dnrm2();
   double normalNorm();*/
  Vector_GPU transpose();
};

// create matrix class for readability / sparsity attribute
class Matrix_GPU : public Vector_GPU {
 public:
  double sparsity = .70;  // the number of 0-elements/non-0-elements
  Matrix_GPU(unsigned int r, unsigned int c);
};