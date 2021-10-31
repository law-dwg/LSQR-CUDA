#pragma once
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

bool compareMat(double *MC, int rowC, int colC, double *MG, int rowG, int colG);

void sizeCheck(unsigned A_r, unsigned A_c, std::vector<double> &A, unsigned b_r, unsigned b_c, std::vector<double> &b, std::string A_f,
               std::string b_f);

void writeArrayToFile(std::string dest, unsigned rows, unsigned cols, double *arr);

void readArrayFromFile(const char *path, unsigned r, unsigned c, std::vector<double> &mat);

double rands();

void matrixBuilder(unsigned r, unsigned c, double sparsity, const char *dir, const char *matLetter);

bool yesNo();

void fileParserLoader(std::string file, unsigned &A_r, unsigned &A_c, std::vector<double> &A, unsigned &b_r, unsigned &b_c, std::vector<double> &b,
                      double &sp);

std::string timeNowString();

template <typename T> bool valInput(T lowLim, T upLim, T &out) {
  while (true) {
    T input;
    std::cin >> std::ws;
    std::cin >> input;
    if ((lowLim <= input && input <= upLim) && (std::cin)) {
      out = std::ceil(input * 100.0) / 100.0; // round up to 2 decimal places
      return false;
    } else {
      printf("Invalid input, please enter a value between %0.2f and %0.2f: ", lowLim, upLim);
    }
  }
}