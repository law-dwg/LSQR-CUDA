#include "utils.cuh"
/** CUDA kernels */
/** VectorCUDA and MatrixCUDA Kernels */
void __global__ print(double *input, unsigned *r, unsigned *c);
void __global__ maxVal(double *in1, unsigned r, unsigned c, double *out);
void __global__ dnrm2(double *in1, unsigned r, unsigned c, double *max, double *out);
/** VectorCUDA Kernels */
void __global__ multiplyNaive(double *in1, unsigned *rows1, unsigned *cols1, double *in2, unsigned *rows2, unsigned *cols2, double *output);
void __global__ multiplyTiled(double *in1, unsigned *rows1, unsigned *cols1, double *in2, unsigned *rows2, unsigned *cols2, double *output);
void __global__ scale(double *input, double scalar, double *output, unsigned *r, unsigned *c, bool inverse);
void __global__ add(double *in1, double *in2, unsigned *rows, unsigned *cols, double *out, bool add);
void __global__ transposeTiled(double *in1, double *output, unsigned *rows, unsigned *cols);
/** MatrixCUDA Kernels */
void __global__ spmvNaive(unsigned *rows, int *csrRowPtr, int *csrColIdn, double *csrVal, double *rhs, double *out);
void __global__ spmvCSRVector(unsigned *rows, int *csrRowPtr, int *csrColInd, double *csrVal, double *rhs, double *out);
void __global__ spmvCSRVectorShared(unsigned *rows, int *csrRowPtr, int *csrColInd, double *csrVal, double *rhs, double *out);