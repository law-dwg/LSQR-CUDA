#include "matVec_cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iostream>
#include <vector>

// constructors
Matrix_CPU::Matrix_CPU(unsigned int r, unsigned int c) {
  this->rows = r;
  this->columns = c;
  this->mat.resize(this->rows * this->columns);
  int zeros = round(this->sparsity * r * c);
  int nonZeros = r * c - zeros;
  // std::cout<<zeros<<std::endl;
  srand(time(0));

  // printf("%f sparsity for %i elements leads to %f zero values which rounds to
  // %d. That means there are %d nonzero
  // values\n",this->sparsity,r*c,this->sparsity * r*c,zeros,nonZeros);
  // printf("mat size: %i\n",mat.size());
  for (int i = 0; i < mat.size(); i++) {
    if (i < nonZeros) {
      mat[i] = rand() % 100;
    } else {
      mat[i] = 0;
    };
  }
  std::random_shuffle(mat.begin(), mat.end());
};

// operator overloads
Vector_CPU Vector_CPU::operator*(Vector_CPU &v) {
  std::vector<double> lhs = this->mat;
  std::vector<double> rhs = v.getMat();
  Vector_CPU out(this->rows);
  // std::cout<<lhs.size()<<std::endl;
  // std::cout<<rhs.size()<<std::endl;
  if (this->columns == rhs.size()) {
    for (int r = 0; r < this->rows; r++) {
      double sum = 0;
      for (int c = 0; c < this->columns; c++) {
        sum += lhs[r * this->columns + c] * rhs[c];
      }
      out.mat[r] = sum;
    }
    return out;
  } else {
    printf("cannot perform this multiplication\n");
    return v;
  }
};

Vector_CPU Vector_CPU::operator*(double i) {
  // probably can find a better implementation
  Vector_CPU out = *this;
  for (int e = 0; e < out.mat.size(); e++) {
    out.mat[e] *= i;
  }
  return out;
};

Vector_CPU Vector_CPU::operator-(Vector_CPU v) {
  Vector_CPU out(this->rows, this->columns);
  if (out.rows == v.rows && out.columns == v.columns) {
    for (int i = 0; i < this->mat.size(); i++) {
      out.mat[i] = this->mat[i] - v.mat[i];
    }
    return out;
  } else {
    printf("cannot perform this subtraction, vectors need to be the same size");
    return (*this);
  };
};

Vector_CPU Vector_CPU::operator+(Vector_CPU v) {
  if (this->mat.size() == 0) {
    return v;
  } else if (this->rows == v.rows && this->columns == v.columns) {
    Vector_CPU out(this->rows, this->columns, 0);
    for (int i = 0; i < this->mat.size(); i++) {
      out.mat[i] = this->mat[i] + v.mat[i];
    }
    return out;
  } else {
    printf("out.mat.size() = %i ", this->mat.size());
    printf("cannot perform this addition, vectors need to be the same size");
    return (*this);
  };
};

double Vector_CPU::operator()(unsigned int i) { return (Vector_CPU::operator[](i)); };

double Vector_CPU::operator()(unsigned int r, unsigned int c) {
  if (r < this->rows && c < this->columns) {
    return (mat[r * this->columns + c]);
  } else {
    printf("please use valid indices: (r,c) where 0=<r<%i and 0=<c<%i\n", this->rows,
           this->columns);
    // throw 505;
    return EXIT_FAILURE;
  }
};

double Vector_CPU::operator[](unsigned int i) { return (this->mat[i]); };

// member functions
void Vector_CPU::h_print() {
  printf("#ofRows:%i #ofCols:%i\n", this->rows, this->columns);
  printf("PRINTING MATRIX\n[");
  for (int e = 0; e < (this->rows * this->columns); e++) {
    if (e % this->columns == 0) {
      (e == 0) ?: printf("\n ");
    };
    std::cout << h_mat[e] << std::endl;
  }
  printf("]\n");
};
void Vector_CPU::print() {
  printf("#ofRows:%i #ofCols:%i\n", this->rows, this->columns);
  printf("PRINTING MATRIX\n[");
  for (int e = 0; e < (this->mat.size()); e++) {
    if (e % this->columns == 0) {
      (e == 0) ?: printf("\n ");
    };
    (e == mat.size() - 1) ? printf("%f", this->mat[e]) : printf("%f ", this->mat[e]);
  }
  printf("]\n");
};

int Vector_CPU::getRows() {
  // printf("number of rows: %i\n",this->rows);
  return this->rows;
};

int Vector_CPU::getColumns() {
  // printf("number of columns: %i\n",this->columns);
  return this->columns;
};

Vector_CPU Vector_CPU::transpose() {
  Vector_CPU out = (*this);
  out.rows = this->columns;
  out.columns = this->rows;
  for (int r = 0; r < this->rows; r++) {
    for (int c = 0; c < this->columns; c++) {
      // printf("new index: %i, old index:
      // %i\n",(r+c*this->rows),(c+r*this->columns));
      out.mat[r + c * this->rows] = this->mat[c + r * this->columns];
    }
  };
  return out;
};

double Vector_CPU::Dnrm2() {
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
  // printf("sumScaled: %f, magOfLargestEle:
  // %f\n",sumScaled,magnitudeOfLargestElement);
  return magnitudeOfLargestElement * sqrt(sumScaled);
};

double Vector_CPU::normalNorm() {
  double sumScaled = 0;
  for (int i = 0; i < this->mat.size(); i++) {
    if (this->mat[i] != 0) {
      double value = this->mat[i];
      sumScaled += value * value;
    }
  }
  return sqrt(sumScaled);
};