#pragma once
#include <iostream>
#include <string>
#include <vector>

bool compareMat(double *MC, int rowC, int colC, double *MG, int rowG, int colG);

bool compareVal(double *VC, double *VG);

void writeArrayToFile(std::string dest, unsigned rows, unsigned cols, double *arr);

void readArrayFromFile(const char *path, unsigned r, unsigned c, std::vector<double> &mat);

double rands();

void matrixBuilder(unsigned r, unsigned c, double sparsity, const char *dir, const char *matLetter);

void loading();

bool yesNo();

void fileParserLoader(std::string file, unsigned &A_r, unsigned &A_c, std::vector<double> &A, unsigned &b_r, unsigned &b_c, std::vector<double> &b);

template <typename T> T valInput(T lowLim, T upLim) {
  T input;
  std::cin >> std::ws;
  std::cin >> input;
  if ((lowLim <= input && input <= upLim) && (std::cin)) {
    return input;
  } else {
    std::cout << "Invalid input, a default value of 0 will be used\n";
    return 0;
  }
}