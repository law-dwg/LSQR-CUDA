#include <cassert>
#include <iostream>
#include <vector>

#include "lsqr_cpu.h"
#include "matVec_cpu.h"

int main() {
  double *a_heap, *b_heap;
  int a_rows, b_rows, a_cols, b_cols;
  a_rows = b_rows = 10;
  a_cols = 9;
  b_cols = 1;
  a_heap = new double[a_rows * a_cols];
  b_heap = new double[b_rows * b_cols];
  for (int i = 0; i < (a_rows * a_cols); i++) {
    if (i < b_rows) {
      a_heap[i] = i;
      b_heap[i] = i;
    } else {
      a_heap[i] = i;
    }
  }
  Vector_CPU A(a_rows, a_cols, a_heap);
  Vector_CPU b(b_rows, b_cols, b_heap);
  Vector_CPU x = lsqr_cpu(A, b);
  x.print();
  delete a_heap, b_heap;
  return 0;
};