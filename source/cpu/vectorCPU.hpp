#pragma once
#include <algorithm>
#include <iostream>
#include <iterator>
#include <vector>

class VectorCPU {
protected:
  // attributes
  unsigned rows;
  unsigned columns;
  std::vector<double> mat;

public:
  /** Constructors */
  VectorCPU() : VectorCPU(0, 0){};                          // Default Constr.
  VectorCPU(unsigned r, unsigned c) : rows(r), columns(c) { // Constr. #2
    this->mat.resize(r * c, 0);
  };
  VectorCPU(unsigned r, unsigned c, double *v) : rows(r), columns(c) {
    mat.resize(r * c);
    mat.assign(v, v + r * c);
  };
  VectorCPU(const VectorCPU &v) : columns(v.columns), rows(v.rows), mat(v.mat){};
  VectorCPU(VectorCPU &&v) noexcept : VectorCPU(v) {
    v.rows = 0;
    v.columns = 0;
    v.mat.clear();
  }
  ~VectorCPU() { mat.clear(); };
  /** Assignments */
  VectorCPU &operator=(const VectorCPU &v) { // Copy assignment operator
    rows = v.rows;
    columns = v.columns;
    mat = v.mat;
    return *this;
  };

  VectorCPU &operator=(VectorCPU &&v) noexcept { // Move assignment operator
    // call copy assignment
    *this = v;
    v.rows = 0;
    v.columns = 0;
    v.mat.clear();
    return *this;
  };
  // operator overloads
  VectorCPU operator*(VectorCPU &v);
  VectorCPU operator*(double i);
  VectorCPU operator-(VectorCPU v);
  VectorCPU operator+(VectorCPU v);
  double operator()(unsigned i);
  double operator()(unsigned r, unsigned c);
  double operator[](unsigned i);

  // member functions
  double *getMat() { return this->mat.data(); };
  void print();
  int getRows() { return this->rows; };
  int getColumns() { return this->columns; };
  double Dnrm2();
  double normalNorm();
  VectorCPU transpose();
};