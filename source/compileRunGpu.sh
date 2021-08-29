#!/bin/bash
rm gpu/out gpu/out.txt gpu/memErrs.txt
clang-format -i --style={'ColumnLimit: 150'} cpu/*.cpp cpu/*.h gpu/*.cu gpu/*.cuh
nvcc -std=c++17 -arch=sm_37 gpu/*.cu cpu/matVec_cpu.cpp -o gpu/out
nvprof gpu/out --device-buffer-size 256MB
cuda-memcheck ./gpu/out > gpu/memErrs.txt
./gpu/out > gpu/out.txt