#pragma once
#include "../cpu/lsqr_cpu.h"
#include "matVec_gpu.cuh"

bool compareMat(double *MC, int rowC, int colC, double *MG, int rowG, int colG);
bool compareVal(double *VC, double *VG);
int checkDevice();
Vector_GPU lsqr_gpu(Matrix_GPU &A, Vector_GPU &b);