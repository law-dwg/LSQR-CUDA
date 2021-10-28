// CPU
#include "../cpu/lsqr.hpp"
#include "../cpu/utils.hpp"
#include "../cpu/vectorCPU.hpp"
// GPU
#include "matrixCUDA.cuh"
#include "matrixCUSPARSE.cuh"
#include "utils.cuh"
#include "vectorCUBLAS.cuh"
#include "vectorCUDA.cuh"
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
  printf("\nWelcome to law-dwg's LSQR-CUDA solver!\n\n"
         "The LSQR-algorithm is used to iteratively solve for the vector x in the following expression: "
         "A * x = b,\n"
         "where A is typically a large matrix of size (m*n) and sparsity sp, and b and x are vectors of sizes (m*1) and (n*1), respectively\n\n"
         "LSQR was designed and first authored by C. C. Paige and M. A. Saunders here: "
         "https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf\n\n"
         "This program runs 5 different implmentations of the LSQR algorithm using C++ and CUDA for both dense and sparse A-matricies:\n"
         "    1. CPU C++ - Dense Matrix Implementation\n"
         "    2. GPU CUDA - Dense Matrix Implementation\n"
         "    3. GPU CUDA - Sparse Matrix Implementation\n"
         "    4. GPU CUBLAS -  Dense Matrix Implementation\n"
         "    5. GPU CUBLAS/CUSPARSE - Sparse Matrix Implementation\n\n");
  printf("This solver requires inputs:\n"
         "    A - 'input/m_n_A_sp.mat'\n"
         "    b - 'input/m_1_b.vec'\n"
         "and writes the solution to:\n"
         "    x - 'output/n_1_x_implmentation.vec'\n"
         "    report - 'output/speedups.csv'\n\n");
  std::cout << "Would you like me to make some test data for you? (y/n): ";
  bool matBuild = yesNo();

  if (matBuild) { // build matrices
    unsigned start = 100;
    unsigned end = 1000;
    unsigned increment = 100;
    unsigned numOfTests = ((end-start)/increment) + 1;
    printf("\nGreat, I will create %d sets of inputs for you\n\nWhat sparsity should matrix A have? Please enter a number between 0.0-0.95: ",numOfTests);
    sp = valInput<double>(0.0, 0.95);
    std::cout << "Building A Matrices of sparsity " << sp << "\n";
    for (int i = start; i <= end; i += increment) {
      matrixBuilder(i, i, sp, "input/", "A");
      matrixBuilder(i, 1, 0, "input/", "b");
    }
  }

  std::string path_name = "input/";
  std::set<fs::path> sorted_by_name;
  for (auto &entry : fs::directory_iterator(path_name)) // alphabetical listing of files in input
    sorted_by_name.insert(entry.path());

  if (sorted_by_name.size() == 0) { // empty input folder
    std::cout << "Looks like there are no files in the input folder. Please add your own matricies in \"NumOfRows_NumOfCols_A.mat\" and "
                 "\"NumOfRows_1_b.vec\" format, or rerun the "
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
    if (!all_checks) {
      printf("\n\nERROR, please check the matrix file naming convention (\"NumOfRows_NumOfCols_A_sp.mat\" and "
             "\"NumOfRows_1_b.vec\" format) and make sure the naming convention (rows * columns) matches the number of values in each file\n\n");
      exit(1);
    }
    if (true) {
      printf("---------------------------------------------\n");
      printf("Running lsqr-CPU implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
      VectorCPU A_c(A_rows, A_cols, A.data());
      VectorCPU b_c(b_rows, b_cols, b.data());
      std::clock_t c_start = std::clock();
      VectorCPU x_c = lsqr<VectorCPU, VectorCPU>(A_c, b_c);
      std::clock_t c_end = std::clock();
      long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
      std::cout << "CPU time used = " << time_elapsed_ms << " ms for lsqr\n";
      std::string file_out = "output/" + std::to_string(A_cols) + "_1_x_CPU.vec";
      writeArrayToFile(file_out, x_c.getRows(), x_c.getColumns(), x_c.getMat());
      printf("---------------------------------------------\n");
    }
    if (true) {
      printf("---------------------------------------------\n");
      printf("Running lsqr-GPU implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
      MatrixCUDA A_vg(A_rows, A_cols, A.data());
      cublasStart();
      cusparseStart();
      // MatrixCUSPARSE A_g(A_rows, A_cols, A.data());
      VectorCUDA b_g(b_rows, b_cols, b.data());
      cudaEvent_t start, stop;
      cudaErrCheck(cudaEventCreate(&start));
      cudaErrCheck(cudaEventCreate(&stop));
      cudaErrCheck(cudaEventRecord(start));
      VectorCUDA x_g = lsqr<MatrixCUDA, VectorCUDA>(A_vg, b_g);
      cudaErrCheck(cudaDeviceSynchronize());
      cudaErrCheck(cudaEventRecord(stop));
      cudaErrCheck(cudaEventSynchronize(stop));
      float milliseconds = 0;
      cudaErrCheck(cudaEventElapsedTime(&milliseconds, start, stop));
      printf("GPU time used = %f ms for lsqr\n", milliseconds);
      std::string file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU.vec";
      VectorCPU x_g_out = x_g.matDeviceToHost();
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
      std::string file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU_CUBLAS.vec";
      VectorCPU x_g_cublas_out = x_g_cublas.matDeviceToHost();
      writeArrayToFile(file_out, x_g_cublas_out.getRows(), x_g_cublas_out.getColumns(), x_g_cublas_out.getMat());
      cusparseStop();
      cublasStop();
      printf("---------------------------------------------\n");
    }
    cudaErrCheck(cudaDeviceReset());
    cudaLastErrCheck();
  }
}