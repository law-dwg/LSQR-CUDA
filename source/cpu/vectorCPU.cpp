#include "vectorCPU.hpp"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// operator overloads
VectorCPU VectorCPU::operator*(VectorCPU &v) {
  std::vector<double> lhs = this->mat;
  std::vector<double> rhs = v.mat;
  VectorCPU out(this->rows, v.columns);
  if (this->columns == v.rows) {
    for (int r = 0; r < this->rows; r++) {
      for (int c = 0; c < v.columns; c++) {
        for (int i = 0; i < v.rows; i++) {
          out.mat[r * out.columns + c] += lhs[r * this->columns + i] * rhs[c + i * v.columns];
        }
      }
    }
  } else {
    printf("Cannot perform multiplication, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  return out;
};

VectorCPU VectorCPU::operator*(double i) {
  // probably can find a better implementation
  VectorCPU out = *this;
  for (int e = 0; e < out.mat.size(); e++) {
    out.mat[e] *= i;
  }
  return out;
};

VectorCPU VectorCPU::operator-(VectorCPU v) {
  VectorCPU out(this->rows, this->columns);
  if (out.rows == v.rows && out.columns == v.columns) {
    for (int i = 0; i < this->mat.size(); i++) {
      out.mat[i] = this->mat[i] - v.mat[i];
    }
  } else {
    printf("Cannot perform subtraction, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  };
  return out;
};

VectorCPU VectorCPU::operator+(VectorCPU v) {
  if (this->mat.size() == 0) {
    return v;
  } else if (this->rows == v.rows && this->columns == v.columns) {
    VectorCPU out(this->rows, this->columns);
    for (int i = 0; i < this->mat.size(); i++) {
      out.mat[i] = this->mat[i] + v.mat[i];
    }
    return out;
  } else {
    printf("out.mat.size() = %i ", this->mat.size());
    printf("Cannot perform addition, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
    return (*this);
  };
};

double VectorCPU::operator()(unsigned i) { return (VectorCPU::operator[](i)); };

double VectorCPU::operator()(unsigned r, unsigned c) {
  if (r < this->rows && c < this->columns) {
    return (mat[r * this->columns + c]);
  } else {
    printf("please use valid indices: (r,c) where 0=<r<%i and 0=<c<%i\n", this->rows, this->columns);
    exit(1);
    return mat[0];
  }
};

double VectorCPU::operator[](unsigned i) { return (this->mat[i]); };

// member functions
void VectorCPU::print() {
  printf("#ofRows:%i #ofCols:%i\n", this->rows, this->columns);
  printf("PRINTING MATRIX\n[");
  for (int e = 0; e < (this->mat.size()); e++) {
    if (e % this->columns == 0) {
      (e == 0) ?: printf("\n ");
    };
    (e == mat.size() - 1) ? printf("%.15f", this->mat[e]) : printf("%.15f ", this->mat[e]);
  }
  printf("]\n");
};

VectorCPU VectorCPU::transpose() {
  VectorCPU out = (*this);
  out.rows = this->columns;
  out.columns = this->rows;
  for (int r = 0; r < this->rows; r++) {
    for (int c = 0; c < this->columns; c++) {
      out.mat[r + c * this->rows] = this->mat[c + r * this->columns];
    }
  };
  return out;
};

double VectorCPU::Dnrm2() {
  double sumScaled = 1.0;
  double magnitudeOfLargestElement = 0.0;
  for (int i = 0; i < this->mat.size(); i++) {
    if (this->mat[i] != 0) {
      double value = this->mat[i];
      double absVal = std::abs(value);
      if (magnitudeOfLargestElement < absVal) {
        // rescale sum to the range of the new element
        value = magnitudeOfLargestElement / absVal;
        sumScaled = sumScaled * (value * value) + 1.0;
        magnitudeOfLargestElement = absVal;
      } else {
        // rescale the new element to the range of the snum
        value = absVal / magnitudeOfLargestElement;
        sumScaled += value * value;
      }
    }
  }
  return magnitudeOfLargestElement * sqrt(sumScaled);
};

double VectorCPU::normalNorm() {
  double sumScaled = 0;
  for (int i = 0; i < this->mat.size(); i++) {
    if (this->mat[i] != 0) {
      double value = this->mat[i];
      sumScaled += value * value;
    }
  }
  return sqrt(sumScaled);
};