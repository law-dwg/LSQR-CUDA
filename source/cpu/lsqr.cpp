#include <cassert>
#include <iostream>
#include <vector>

#include "lsqr_cpu.h"
#include "matVec_cpu.h"

int main() {
  Matrix_CPU A(3, 2);
  Vector_CPU b(3);
  A.mat = {1.0, -2.2, 3.2, 4.0, 5.0, 6.1};
  A.print();
  b.mat = {0.01, 0.2, -0.2};
  Vector_CPU x = lsqr_cpu(A, b);
  return 0;
};