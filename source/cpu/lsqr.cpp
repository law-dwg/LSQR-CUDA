#include <cassert>
#include <iostream>
#include <vector>

#include "lsqr_cpu.h"
#include "matVec_cpu.h"

int main() {
  double *a_heap = new double[3 * 3]{1.0, -2.2, 0, 3.2, 4.0, 0, 5.0, 0, 6.1};
  double *b_heap = new double[3 * 1]{0.01, 0.2, -0.2};
  Vector_CPU A(3, 3, a_heap);
  Vector_CPU b(3, 1, b_heap);
  A.print();
  b.print();
  Vector_CPU x = lsqr_cpu(A, b);
  x.print();
  delete a_heap, b_heap;
  return 0;
};