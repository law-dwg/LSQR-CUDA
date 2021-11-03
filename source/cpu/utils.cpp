#include "utils.hpp"
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <math.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

bool compareMat(double *MC, int rowC, int colC, double *MG, int rowG, int colG) {
  bool same = true;
  double epsilon = 1e-9;
  if (rowC != rowG || colC != colG || !same) {
    printf("MATRICIES SIZE  DO NOT MATCH matCPU(%d x %d) != matGPU(%d, %d)\n", rowC, colC, rowG, rowC);
    same = false;
  }
  if (same) {
    for (int i = 0; i < rowC * colC; i++) {
      if (!(std::abs(MG[i] - MC[i]) < epsilon)) {
        printf("MATRICIES SIZE (%d x %d) DO NOT MATCH DISCREPANCY AT INDEX %d; DIFF = %f, %f "
               "== %f\n",
               rowC, colC, i, std::abs(MG[i] - MC[i]), MG[i], MC[i]);
        printf("MG[%d]=%f, MC[%d]=%f\nMG[%d]=%f, MC[%d]=%f\nMG[%d]=%f, "
               "MC[%d]=%f\nMG[%d]=%f, MC[%d]=%f\n",
               i - 1, MG[i - 1], i - 1, MC[i - 1], i, MG[i], i, MC[i], i + 1, MG[i + 1], i + 1, MC[i + 1], i + 2, MG[i + 2], i + 2, MC[i + 2]);
        printf("LAST ELEMENTS MG[%d]=%f, MC[%d]=%f\n", (rowC * colC) - 1, MG[(rowC * colC) - 1], (rowC * colC) - 1, MC[(rowC * colC) - 1]);
        same = false;
        break;
      }
    }
  };
  if (same) {
    printf("MATRICES MATCH FOR (%d x %d)\n", rowC, colC);
  };
  return same;
};

void sizeCheck(unsigned A_r, unsigned A_c, std::vector<double> &A, unsigned b_r, unsigned b_c, std::vector<double> &b, std::string A_f,
               std::string b_f) {
  bool A_sizecheck, b_sizecheck, Ab_rowscheck, all_checks;
  A_sizecheck = A.size() == A_r * A_c && A_r != 0 && A_c != 0;
  b_sizecheck = b.size() == b_r * b_c && b_r != 0 && b_c == 1;
  Ab_rowscheck = A_r == b_r;
  all_checks = A_sizecheck && b_sizecheck && Ab_rowscheck;
  if (!all_checks) {
    printf("\n\nERROR, please check the matrix file naming convention of\nA - '%s' == 'input/#rows_#cols_A_#sparsity.mat'\nb - '%s'"
           " == input/'#rows_1_b.vec'\n\nThese values (rows * columns) must match the number of values in each file\n\n",
           A_f.c_str(), b_f.c_str());
    exit(1);
  }
};

void writeArrayToFile(std::string dest, unsigned rows, unsigned cols, double *arr) {
  typedef std::numeric_limits<double> dbl;
  std::ofstream myfileA(dest);
  if (myfileA.is_open()) {
    for (int count = 0; count < rows * cols; count++) {
      if (count % cols == 0 && count != 0) {
        myfileA << "\n";
      }
      if ((count % cols) == (cols - 1)) {
        myfileA << std::setprecision(16) << arr[count];
      } else {
        myfileA << std::setprecision(16) << arr[count] << " ";
      }
    }
    myfileA.close();
    std::cout << "Data written to " << dest << std::endl;
  } else
    std::cout << "Unable to open file";
};

void readArrayFromFile(const char *path, unsigned r, unsigned c, std::vector<double> &mat) {
  std::ifstream file(path);
  assert(file.is_open());
  std::copy(std::istream_iterator<double>(file), std::istream_iterator<double>(), std::back_inserter(mat));
  file.close();
};

double rands() {
  static std::random_device rd;
  static std::mt19937 rng(rd());
  static std::uniform_int_distribution<std::mt19937::result_type> dist25(1, 25); // distribution in range [1, 6]
  return dist25(rng);
}

void matrixBuilder(unsigned r, unsigned c, double sparsity, const char *dir, const char *matLetter) {
  std::vector<double> mat(r * c, 0.0);
  int zeros = round(sparsity * r * c);
  int nonZeros = r * c - zeros;
  for (int i = 0; i < nonZeros; i++) {
    mat[i] = rands();
  }
  std::random_shuffle(mat.begin(), mat.end());
  std::string fileExt = (*matLetter == 'A') ? "mat" : "vec";
  std::string sparsityStr = (*matLetter == 'A') ? ("_" + std::to_string((int)(sparsity * 100))) : "";
  std::stringstream fileName;
  fileName << dir << r << "_" << c << sparsityStr << "_" << matLetter << "." << fileExt;
  writeArrayToFile(fileName.str(), r, c, mat.data());
};

bool yesNo() {
  while (true) {
    std::string s;
    std::cin >> std::ws;
    getline(std::cin, s);
    if (s.empty())
      continue;
    switch (toupper(s[0])) {
    case 'Y':
      return true;
    case 'N':
      return false;
    }
    std::cout << "Invalid input, please enter Y for yes and N for no: ";
  }
}

void fileParserLoader(std::string file, unsigned &A_r, unsigned &A_c, std::vector<double> &A, unsigned &b_r, unsigned &b_c, std::vector<double> &b,
                      double &sp) {
  std::string path = file.c_str(); // keep path for reading data

  // parse filename
  std::vector<std::string> delim{"/\\", ".", "_"};
  size_t dot = file.find_last_of(delim[1]);           // file extension location
  size_t slash = file.find_last_of(delim[0]);         // file prefix location
  file.erase(file.begin() + dot, file.end());         // remove file extension
  file.erase(file.begin(), file.begin() + slash + 1); // remove file prefix
  size_t undersLast = file.find_last_of(delim[2]);    // underscore at end of filename
  size_t undersFirst = file.find(delim[2]);           // underscore at beginning of filename

  // read and allocate data now in '#rows_#cols_#sp_A' or '#rows_1_b' format
  if (file.substr(undersLast + 1) == "A") {                                        // A matrix
    std::string middle = file.substr(undersFirst + 1, (undersLast - undersFirst)); // '#cols_#sp_'

    size_t undersFirstMiddle = middle.find(delim[2]);
    size_t undersLastMiddle = middle.find_last_of(delim[2]);

    A_r = std::stoi(file.substr(0, undersFirst));                                                                 // read rows from filename
    A_c = std::stoi(middle.substr(0, undersFirstMiddle));                                                         // read cols from filename
    sp = (double)std::stoi(middle.substr(undersFirstMiddle + 1, undersLastMiddle - undersFirstMiddle - 1)) / 100; // read sp from filename
    printf("Loading matrix A(%d,%d), sparsity %0.2f...", A_r, A_c, sp);
    readArrayFromFile(path.c_str(), A_r, A_c, A);
    printf(" done\n");
  } else if (file.substr(undersLast + 1) == "b") {                             // b Vector
    b_r = std::stoi(file.substr(0, undersFirst));                              // read rows from filename
    b_c = std::stoi(file.substr(undersFirst + 1, (undersLast - undersFirst))); // read cols from filename
    printf("Loading matrix b(%d,%d)...", b_r, b_c);
    readArrayFromFile(path.c_str(), b_r, b_c, b);
    printf(" done\n");
  } else { // err
    printf("Error while trying to read %s, please rename to either \"#rows_1_b.vec\" or \"#rows_#columns_sparsity(0-95)_A.mat\" \n", path);
  }
};

std::string timeNowString() {
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  tm t = *localtime(&now_time);
  std::string out;
  std::string min = (t.tm_min < 10) ? "0" + std::to_string(t.tm_min) : std::to_string(t.tm_min);
  out =
      std::to_string(t.tm_year + 1900) + "-" + std::to_string(t.tm_mon + 1) + "-" + std::to_string(t.tm_mday) + "T" + std::to_string(t.tm_hour) + min;
  return out;
};