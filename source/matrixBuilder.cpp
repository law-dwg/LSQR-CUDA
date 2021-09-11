#include "matrixBuilder.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <math.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <time.h>
#include <vector>
void writeArrayToFile(std::string dest, unsigned rows, unsigned cols, double *arr) {
  std::ofstream myfileA(dest);
  if (myfileA.is_open()) {
    for (int count = 0; count < rows * cols; count++) {
      if (count != ((rows * cols) - 1)) {
        myfileA << arr[count] << " ";
      } else {
        myfileA << arr[count];
      }
    }
    myfileA.close();
    std::cout << "Data written to " << dest << std::endl;
  } else
    std::cout << "Unable to open file";
};

void readArrayFromFile(const char *path, unsigned r, unsigned c, std::vector<double> &mat) {
  // mat.resize(r*c); // not necessary
  std::ifstream file(path);
  assert(file.is_open());
  std::copy(std::istream_iterator<double>(file), std::istream_iterator<double>(), std::back_inserter(mat));
  file.close();
};

double rands() {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  static std::uniform_int_distribution<std::mt19937::result_type> dist25(1, 25); // distribution in range [1, 6]
  // std::cout << dist25(rng) << std::endl;
  return dist25(rng);
}

void matrixBuilder(unsigned int r, unsigned int c, double sparsity, const char *dir, const char *matLetter) {
  // typedef std::mt19937 MyRNG; // the Mersenne Twister with a popular choice of parameters
  // uint32_t seed_val;          // populate somehow
  //
  // MyRNG rng; // e.g. keep one global instance (per thread)
  //
  // void initialize() { rng.seed(seed_val); }
  rands();
  std::vector<double> mat(r * c, 0.0);
  int zeros = round(sparsity * r * c);
  int nonZeros = r * c - zeros;
  // std::cout<<zeros<<std::endl;

  // printf("%f sparsity for %i elements leads to %f zero values which rounds to
  // %d. That means there are %d nonzero
  // values\n",this->sparsity,r*c,this->sparsity * r*c,zeros,nonZeros);
  // printf("mat size: %i\n",mat.size());
  for (int i = 0; i < nonZeros; i++) {
    mat[i] = rands();
  }
  std::random_shuffle(mat.begin(), mat.end());

  std::stringstream fileName;
  fileName << dir << r << "_" << c << "_" << matLetter << ".txt";
  writeArrayToFile(fileName.str(), r, c, mat.data());
};

// int main() {
//   matrixBuilder(10, 10, 0.01, "input/A");
//   return 0;
// }