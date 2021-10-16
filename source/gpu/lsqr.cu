// CPU
#include "../cpu/lsqr.hpp"
#include "../cpu/matVec_cpu.hpp"
#include "../cpu/matrixBuilder.hpp"
// GPU
#include "matVec_cublas.cuh"
#include "matVec_gpu.cuh"
#include "mat_cusparse.cuh"
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
    sp = 0.6;
    std::cout << "Building A Matrices of sparsity " << sp << "\n";
    for (int i = 1000; i <= 1100; i += 500) {
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
    if (all_checks) {
      continue;
    } else {
      printf("Error, please check the matrix file naming convention (\"NumOfRows_NumOfCols_A.txt\" and "
             "\"NumOfRows_1_b.txt\" format) and make sure the naming convention (rows * columns) matches the number of values in each file\n");
      return 0;
    }
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
    printf("Running lsqr-GPU implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
    Vector_GPU A_vg(A_rows, A_cols, A.data());
    cublasStart();
    cusparseStart();
    Matrix_cuSPARSE A_g(A_rows, A_cols, A.data());
    Vector_GPU b_g(b_rows, b_cols, b.data());
    cudaEvent_t start, stop;
    cudaErrCheck(cudaEventCreate(&start));
    cudaErrCheck(cudaEventCreate(&stop));
    cudaErrCheck(cudaEventRecord(start));
    Vector_GPU x_g = lsqr<Vector_GPU, Vector_GPU>(A_vg, b_g);
    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
    float milliseconds = 0;
    cudaErrCheck(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("GPU time used = %f ms for lsqr\n", milliseconds);
    file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU.txt";
    Vector_CPU x_g_out = x_g.matDeviceToHost();
    writeArrayToFile(file_out, x_g_out.getRows(), x_g_out.getColumns(), x_g_out.getMat());
    cusparseStop();
    cublasStop();
    printf("---------------------------------------------\n");
    printf("Running lsqr-GPU-cublas implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
    cublasStart();
    cusparseStart();
    // Vector_CUBLAS A_g_cublas(A_rows, A_cols, A.data());
    Matrix_cuSPARSE A_g_cublas(A_rows, A_cols, A.data());
    Vector_CUBLAS b_g_cublas(b_rows, b_cols, b.data());
    cudaEvent_t start2, stop2;
    cudaErrCheck(cudaEventCreate(&start2));
    cudaErrCheck(cudaEventCreate(&stop2));
    cudaErrCheck(cudaEventRecord(start2));
    Vector_CUBLAS x_g_cublas = lsqr<Matrix_cuSPARSE, Vector_CUBLAS>(A_g_cublas, b_g_cublas);
    cudaErrCheck(cudaDeviceSynchronize());
    cudaErrCheck(cudaEventRecord(stop2));
    cudaErrCheck(cudaEventSynchronize(stop2));
    milliseconds = 0;
    cudaErrCheck(cudaEventElapsedTime(&milliseconds, start2, stop2));
    printf("GPU time used = %f ms for lsqr\n", milliseconds);
    file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU_CUBLAS.txt";
    Vector_CPU x_g_cublas_out = x_g_cublas.matDeviceToHost();
    writeArrayToFile(file_out, x_g_cublas_out.getRows(), x_g_cublas_out.getColumns(), x_g_cublas_out.getMat());
    cusparseStop();
    cublasStop();
    printf("---------------------------------------------\n");
    // printf("Running cuSolver implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
    // cusparseStart();
    // cublasStart();
    // cusolverStart();
    // // Vector_GPU x_sol_out(A_g_cusparse.getRows(), 1);
    //
    // int singularity = 0;
    // const int reorder = 0; /* no reordering */
    // int rankA;
    // int p;
    // double min_norm;
    // double *b_test = b.data();
    // double *x_c_out = new double[A_g_cusparse.getColumns()];
    // double *c_csr_values = new double[A_g_cusparse.h_nnz];
    // int *c_csr_columns = new int[A_g_cusparse.h_nnz];
    // int *c_csr_offsets = new int[A_g_cusparse.getRows() + 1];
    // cudaErrCheck(cudaMemcpy(c_csr_values, A_g_cusparse.d_csr_values, A_g_cusparse.h_nnz * sizeof(double), cudaMemcpyDeviceToHost));
    // cudaErrCheck(cudaMemcpy(c_csr_columns, A_g_cusparse.d_csr_columns, A_g_cusparse.h_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    // cudaErrCheck(cudaMemcpy(c_csr_offsets, A_g_cusparse.d_csr_offsets, (A_g_cusparse.getRows() + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    // cusparseMatDescr_t descr = NULL;
    // cusparseErrCheck(cusparseCreateMatDescr(&descr));
    // cusparseErrCheck(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    // cusparseErrCheck(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    // cusolverErrCheck(cusolverSpDcsrlsqvqrHost(solHandle, A_g_cusparse.getRows(), A_g_cusparse.getColumns(), A_g_cusparse.h_nnz, descr,
    // c_csr_values,
    //                                           c_csr_offsets, c_csr_columns, b_test, TOL, &rankA, x_c_out, &p, &min_norm));
    // cudaErrCheck(cudaDeviceSynchronize());
    // printf("HERE %d\n", p);
    // // Vector_CPU x_c_sol_out = x_sol_out.matDeviceToHost();
    // // Vector_CPU x_c_sol_out(A_g_cusparse.getRows(), 1, x_c_out);
    // // x_c_sol_out.print();
    // // file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU_CUSOLVER.txt";
    // // writeArrayToFile(file_out, x_c_sol_out.getRows(), x_c_sol_out.getColumns(), x_c_sol_out.getMat());
    // // cusolverStop();
    // cusparseStop();
    // cublasStop();
    cudaLastErrCheck();
  }
}