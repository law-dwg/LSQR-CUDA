#!/bin/bash
rm gpu/out gpu/out.txt gpu/memErrs.txt
clang-format -i --style={'ColumnLimit: 150'} cpu/*.cpp cpu/*.h gpu/*.cu gpu/*.cuh
nvcc -std=c++17 -arch=sm_37 matrixBuilder.cpp lsqr.cu cpu/*cpp gpu/matVec_gpu.cu gpu/lsqr_gpu.cu -o gpu/out
nvprof gpu/out --device-buffer-size 256MB
python3 ../python/lsqr.py
#cuda-memcheck ./gpu/out #> gpu/memErrs.txt
#./gpu/out #> gpu/out.txt