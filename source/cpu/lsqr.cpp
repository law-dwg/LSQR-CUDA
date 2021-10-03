#include "lsqr.hpp"
#include <math.h>
#include <stdio.h>

double D2Norm(double a, double b) {
  const double scale = std::abs(a) + std::abs(b);
  const double zero = 0.0;

  if (scale == zero) {
    return zero;
  }

  const double sa = a / scale;
  const double sb = b / scale;
  // printf("D2N: %f\n", scale * sqrt(sa * sa + sb * sb));
  return scale * sqrt(sa * sa + sb * sb);
};