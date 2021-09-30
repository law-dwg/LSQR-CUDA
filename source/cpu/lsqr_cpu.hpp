#pragma once
#include "matVec_cpu.hpp"

void writeArrayToFile(const char *dest, unsigned size, double *arr);
double D2Norm_cpu(double a, double b);
Vector_CPU lsqr_cpu(Vector_CPU &A, Vector_CPU &b);