#include "utils.cuh"

void __global__ print(double *input, unsigned *r, unsigned *c);
void __global__ maxVal(double *in1, unsigned r, unsigned c, double *out);
void __global__ dnrm2(double *in1, unsigned r, unsigned c, double *max, double *out);
