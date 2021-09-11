#include "lsqr_cpu.h"
#include <fstream>
#include <iostream>
#include <math.h>

void writeArrayToFile(const char *dest, unsigned size, double *arr) {
  std::ofstream myfileA(dest);
  if (myfileA.is_open()) {
    for (int count = 0; count < size; count++) {
      if (count != (size - 1)) {
        myfileA << arr[count] << " ";
      } else {
        myfileA << arr[count];
      }
    }
    myfileA.close();
  } else
    std::cout << "Unable to open file";
};
double D2Norm(double a, double b) {
  const double scale = std::abs(a) + std::abs(b);
  const double zero = 0.0;

  if (scale == zero) {
    return zero;
  }

  const double sa = a / scale;
  const double sb = b / scale;

  return scale * sqrt(sa * sa + sb * sb);
};
