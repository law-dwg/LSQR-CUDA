#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif
#include "matrixBuilder.h"
#include <ctime>
#include <ctype.h>
#include <filesystem>
#include <iostream>
#include <set>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;
void loading() {
  std::cout << "Loading";
  std::cout.flush();
  for (;;) {
    for (int i = 0; i < 3; i++) {
      std::cout << ".";
      std::cout.flush();
      sleep(1);
    }
    std::cout << "\b\b\b   \b\b\b";
  }
}

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

void fileParserLoader(std::string file, unsigned &A_r, unsigned &A_c, std::vector<double> &A, unsigned &b_r, unsigned &b_c, std::vector<double> &b) {
  std::string path = file.c_str(); // keep path for reading data

  // parse filename
  std::vector<std::string> delim{"/\\", ".", "_"};
  size_t dot = file.find_last_of(delim[1]);           // file extension location
  size_t slash = file.find_last_of(delim[0]);         // file prefix location
  file.erase(file.begin() + dot, file.end());         // remove file extension
  file.erase(file.begin(), file.begin() + slash + 1); // remove file prefix
  size_t unders2 = file.find_last_of(delim[2]);       // underscore at end of filename
  size_t unders1 = file.find(delim[2]);               // underscore at beginning of filename

  // read and allocate data
  if (file.substr(unders2 + 1) == "A") {                            // A Matrix
    A_r = std::stoi(file.substr(0, unders1));                       // read rows from filename
    A_c = std::stoi(file.substr(unders1 + 1, (unders2 - unders1))); // read cols from filename
    printf("Loading matrix A(%d,%d)...", A_r, A_c);
    readArrayFromFile(path.c_str(), A_r, A_c, A);
    printf(" done\n");
  } else if (file.substr(unders2 + 1) == "b") {                     // b Vector
    b_r = std::stoi(file.substr(0, unders1));                       // read rows from filename
    b_c = std::stoi(file.substr(unders1 + 1, (unders2 - unders1))); // read cols from filename
    printf("Loading matrix b(%d,%d)...", b_r, b_c);
    readArrayFromFile(path.c_str(), b_r, b_c, b);
    printf(" done\n");
  } else { // err
    printf("Error while trying to read %s, please rename to either \"NumOfRows_1_b.txt\" or \"NumOfRows_NumOfCols_A.txt\" \n", path);
  }
}

template <typename T> T valInput(T lowLim, T upLim) {
  while (true) {
    T input;
    std::cin >> std::ws;
    std::cin >> input;
    if ((lowLim <= input && input <= upLim) && (std::cin)) {
      return input;
    } else {
      std::cout << "Invalid input, a default value of 0.75 will be used\n";
      return 0.75;
    }
  }
}

int main() {
  double sp;
  std::string userName;
  std::cout << "Welcome to law-dwg's lsqr cuda and cpp implementations!\nYou can use ctrl+c to kill this program at any time.\n\nBefore we begin, "
               "please type in your name: ";
  std::cin >> std::ws; // skip leading whitespace
  std::getline(std::cin, userName);
  // start
  std::cout << "Hello " << userName << ", Would you like to build the test matrices from scratch? (y/n): ";
  bool matBuild = yesNo();
  if (matBuild) { // build matrices
    std::cout << "\nGreat, lets get started\n\nWhat sparsity should matrix A have? Please enter a number between 0.0-1.0: ";
    sp = valInput<double>(0.0, 1.0);
    std::cout << "Building A Matrices of sparsity " << sp << "\n";
    for (int i = 500; i < 10000; i += 500) {
      matrixBuilder(i, i, sp, "input/", "A");
      matrixBuilder(i, 1, 0, "input/", "b");
    }
  }

  std::string path_name = "input/";
  std::set<fs::path> sorted_by_name;
  for (auto &entry : fs::directory_iterator(path_name)) // alphabetical listing of files in input
    sorted_by_name.insert(entry.path());

  if (sorted_by_name.size() == 0) { // empty input folder
    std::cout << "Looks like there are no files in the input folder. Please add your own matricies in \"NumOfRows_NumOfCols_A.txt\" and "
                 "\"NumOfRows_1_b.txt\" format, or rerun the "
                 "program to autobuild matrices\n"
              << std::endl;
    return 0;
  };

  std::set<fs::path>::iterator it = sorted_by_name.begin();
  while (it != sorted_by_name.end()) { // iterate through sorted files
    std::string file1, file2;
    file1 = *it;
    ++it;
    file2 = *it;
    ++it; // iterate every two files
    std::vector<std::string> files{file1, file2};
    unsigned A_rows, A_cols, b_rows, b_cols;
    std::vector<double> A, b;
    for (auto file : files) {
      fileParserLoader(file, A_rows, A_cols, A, b_rows, b_cols, b);
    }
    bool A_sizecheck, b_sizecheck, Ab_rowscheck, b_colscheck, all_checks;
    A_sizecheck = A.size() == A_rows * A_cols && A_rows != 0 && A_cols != 0;
    b_sizecheck = b.size() == b_rows * b_cols && b_rows != 0 && b_cols == 1;
    Ab_rowscheck = A_rows == b_rows;
    all_checks = A_sizecheck && b_sizecheck && Ab_rowscheck;
    if (all_checks) {
      continue;
    } else {
      printf("Error, please check the matrix file naming convention (\"NumOfRows_NumOfCols_A.txt\" and "
             "\"NumOfRows_1_b.txt\" format) and make sure the naming convention (rows * columns) matches the number of values in each file\n");
      return 0;
    }
  }
}