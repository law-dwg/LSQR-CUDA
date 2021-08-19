#!/bin/bash
nvcc -std=c++17 -arch=sm_37 gpu/*.cu cpu/matVec_cpu.cpp -o out && ./out
