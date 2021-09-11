#pragma once
#include <iostream>
#include <string>
void writeArrayToFile(std::string dest, unsigned rows, unsigned cols, double *arr);
double rands();
void matrixBuilder(unsigned int r, unsigned int c, double sparsity, const char *prefix);