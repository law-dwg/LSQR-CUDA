#pragma once
#include "VectorCUDA.cuh"

class VectorCUBLAS : public VectorGPU {
public:
  /** Inherit everything */
  using VectorGPU::VectorGPU;

  /** Operator overloads */
  VectorCUBLAS operator*(VectorCUBLAS &v);       // Multiplication
  VectorCUBLAS operator*(double i);              // Scale
  VectorCUBLAS operator-(const VectorCUBLAS &v); // Subtraction
  VectorCUBLAS operator+(const VectorCUBLAS &v); // Addittion
  void operator=(Vector_CPU &v);                 // CopyToDevice

  /** Member Functions */
  VectorCUBLAS transpose();     // Transpose
  void printmat();              // PrintKernel
  Vector_CPU matDeviceToHost(); // CopyToHost
  double Dnrm2();               // EuclideanNorm
};