#!/bin/bash
rm gpu/out
clang-format -i --style={'BasedOnStyle: Google, ColumnLimit: 100'} cpu/*.cpp cpu/*.h gpu/*.cu gpu/*.cuh
nvcc -std=c++17 -arch=sm_37 gpu/*.cu cpu/matVec_cpu.cpp -o gpu/out && ./gpu/out