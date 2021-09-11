#pragma once
#include <iostream>
#include <string>
#include <vector>
void writeArrayToFile(std::string dest, unsigned rows, unsigned cols, double *arr);
void readArrayFromFile(const char *path, unsigned r, unsigned c, std::vector<double> &mat);
double rands();
void matrixBuilder(unsigned int r, unsigned int c, double sparsity, const char *dir, const char *matLetter);