// CPU
#include "../cpu/lsqr.hpp"
#include "../cpu/matVec_cpu.hpp"
#include "../cpu/matrixBuilder.hpp"
// GPU
#include "MatrixCSR.cuh"
#include "MatrixCUSPARSE.cuh"
#include "VectorCUBLAS.cuh"
#include "VectorCUDA.cuh"
#include "utils.cuh"
// Libs
#include <cassert>
#include <chrono>
#include <ctime>
#include <ctype.h>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <set>
#include <sstream>
#include <string>
#include <time.h>

namespace fs = std::filesystem;

int main() {
  double sp;
  std::string userName;
  std::cout << "Welcome to law-dwg's lsqr cuda and cpp implementations!\nYou can use ctrl+c to kill this program at any time.\n\nBefore we begin, "
               "please type in your name: ";
  // std::cin >> std::ws; // skip leading whitespace
  // std::getline(std::cin, userName);
  // start
  std::cout << "Hello " << userName << ", Would you like to build the test matrices from scratch? (y/n): ";
  // bool matBuild = yesNo();
  bool matBuild = true;

  if (matBuild) { // build matrices
    std::cout << "\nGreat, lets get started\n\nWhat sparsity should matrix A have? Please enter a number between 0.0-1.0: ";
    // sp = valInput<double>(0.0, 1.0);
    sp = 0.5;
    std::cout << "Building A Matrices of sparsity " << sp << "\n";
    for (int i = 100; i <= 1000; i += 177) {
      matrixBuilder(i + 333, i, sp, "input/", "A");
      matrixBuilder(i + 333, 1, 0, "input/", "b");
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
  checkDevice();
  std::set<fs::path>::iterator it = sorted_by_name.begin();
  while (it != sorted_by_name.end()) { // iterate through sorted files
    cudaErrCheck(cudaDeviceReset());
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
    assert(all_checks);
    // if (all_checks) {
    //  continue;
    //} else {
    //  printf("Error, please check the matrix file naming convention (\"NumOfRows_NumOfCols_A.txt\" and "
    //         "\"NumOfRows_1_b.txt\" format) and make sure the naming convention (rows * columns) matches the number of values in each file\n");
    //  return 0;
    //}
    if (true) {
      printf("---------------------------------------------\n");
      printf("Running lsqr-CPU implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
      Vector_CPU A_c(A_rows, A_cols, A.data());
      Vector_CPU b_c(b_rows, b_cols, b.data());
      // A_c.print();
      // b_c.print();
      std::clock_t c_start = std::clock();
      Vector_CPU x_c = lsqr<Vector_CPU, Vector_CPU>(A_c, b_c);
      std::clock_t c_end = std::clock();
      long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
      std::cout << "CPU time used = " << time_elapsed_ms << " ms for lsqr\n";
      std::string file_out = "output/" + std::to_string(A_cols) + "_1_x_CPU.txt";
      writeArrayToFile(file_out, x_c.getRows(), x_c.getColumns(), x_c.getMat());
      printf("---------------------------------------------\n");
    }
    if (true) {
      printf("---------------------------------------------\n");
      printf("Running lsqr-GPU implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
      MatrixCSR A_vg(A_rows, A_cols, A.data());
      cublasStart();
      cusparseStart();
      // MatrixCUSPARSE A_g(A_rows, A_cols, A.data());
      VectorCUDA b_g(b_rows, b_cols, b.data());
      cudaEvent_t start, stop;
      cudaErrCheck(cudaEventCreate(&start));
      cudaErrCheck(cudaEventCreate(&stop));
      cudaErrCheck(cudaEventRecord(start));
      VectorCUDA x_g = lsqr<MatrixCSR, VectorCUDA>(A_vg, b_g);
      cudaErrCheck(cudaDeviceSynchronize());
      cudaErrCheck(cudaEventRecord(stop));
      cudaErrCheck(cudaEventSynchronize(stop));
      float milliseconds = 0;
      cudaErrCheck(cudaEventElapsedTime(&milliseconds, start, stop));
      printf("GPU time used = %f ms for lsqr\n", milliseconds);
      std::string file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU.txt";
      Vector_CPU x_g_out = x_g.matDeviceToHost();
      writeArrayToFile(file_out, x_g_out.getRows(), x_g_out.getColumns(), x_g_out.getMat());
      cusparseStop();
      cublasStop();
      printf("---------------------------------------------\n");
    }
    cudaErrCheck(cudaDeviceReset());

    if (true) {
      printf("---------------------------------------------\n");
      printf("Running lsqr-GPU-cublas implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
      cublasStart();
      cusparseStart();
      MatrixCUSPARSE A_g(A_rows, A_cols, A.data());
      // VectorCUBLAS A_g_cublas(A_rows, A_cols, A.data());
      VectorCUBLAS b_g_cublas(b_rows, b_cols, b.data());
      cudaEvent_t start2, stop2;
      cudaErrCheck(cudaEventCreate(&start2));
      cudaErrCheck(cudaEventCreate(&stop2));
      cudaErrCheck(cudaEventRecord(start2));
      VectorCUBLAS x_g_cublas = lsqr<MatrixCUSPARSE, VectorCUBLAS>(A_g, b_g_cublas);
      cudaErrCheck(cudaDeviceSynchronize());
      cudaErrCheck(cudaEventRecord(stop2));
      cudaErrCheck(cudaEventSynchronize(stop2));
      float milliseconds = 0;
      cudaErrCheck(cudaEventElapsedTime(&milliseconds, start2, stop2));
      printf("GPU time used = %f ms for lsqr\n", milliseconds);
      std::string file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU_CUBLAS.txt";
      Vector_CPU x_g_cublas_out = x_g_cublas.matDeviceToHost();
      writeArrayToFile(file_out, x_g_cublas_out.getRows(), x_g_cublas_out.getColumns(), x_g_cublas_out.getMat());
      cusparseStop();
      cublasStop();
      printf("---------------------------------------------\n");
    }
    cudaErrCheck(cudaDeviceReset());
    cudaLastErrCheck();
  }
}