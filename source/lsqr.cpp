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
  std::cout << "Hello " << userName << ", Would you like to build the test matrices from scratch? (y/n): ";
  bool matBuild = yesNo();
  if (matBuild) {
    std::cout << "\nGreat, lets get started\n\nWhat sparsity should matrix A have? Please enter a number between 0.0-1.0: ";
    sp = valInput<double>(0.0, 1.0);
    std::cout << "Building A Matrices of sparsity " << sp << "\n";
    for (int i = 100; i < 1000; i += 100) {
      matrixBuilder(i, i, sp, "input/A");
      matrixBuilder(i, 1, 0, "input/b");
    }
  }
  
  std::string path_name = "input/";
  std::set<fs::path> sorted_by_name;
  for (auto &entry : fs::directory_iterator(path_name)) // alphabetical listing of files in input
    sorted_by_name.insert(entry.path());
  
  if (sorted_by_name.size() == 0) {
    std::cout << "Looks like there are no files in the input folder. Please add your own matricies in \"A_NumOfRowsxNumOfCols.txt\" and "
                 "\"b_NumOfRowsx1.txt\" format, or rerun the "
                 "program to autobuild matrices\n"
              << std::endl;
    return 0;
  };
  
  auto first = sorted_by_name.begin();
  for (int i = 0; i < sorted_by_name.size() / 2; ++i) {
    std::advance(first, i);
    std::string filename = *first.c_str();
    // std::cout<<"Running lsqr for matrix size "
    std::cout << filename << std::endl;
  }
}