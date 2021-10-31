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
#include <fstream>
#include <iostream>
#include <math.h>
#include <set>
#include <sstream>
#include <string>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <string>

const std::string NOW = timeNowString();

template <typename Mat, typename Vec>
float lsqrGPU(unsigned A_r, unsigned A_c, double *A, unsigned b_r, unsigned b_c, double *b, std::string implem, std::string fout, double sp) {
  printf("---------------------------------------------\n");
  printf("%s\n\nAx=b where A(%d,%d) and b(%d,1)\n", implem.c_str(), A_r, A_c, b_r);
  cublasStart();
  cusparseStart();
  Mat A_g(A_r, A_c, A);
  Vec b_g(b_r, b_c, b);
  cudaEvent_t start, stop;
  cudaErrCheck(cudaEventCreate(&start));
  cudaErrCheck(cudaEventCreate(&stop));
  cudaErrCheck(cudaEventRecord(start));
  Vec x_g = lsqr<Mat, Vec>(A_g, b_g);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaEventRecord(stop));
  cudaErrCheck(cudaEventSynchronize(stop));
  float duration = 0;
  cudaErrCheck(cudaEventElapsedTime(&duration, start, stop));
  printf("GPU time used = %f ms for lsqr\n", duration);
  std::string sparsityStr = ((int)fout.find("SPARSE"))>(-1) ?  "-"+std::to_string((int)(sp*100))  : "";
  std::string file_out = "output/" + NOW + "/" +std::to_string(A_c) + "_1_x_" + fout + sparsityStr + ".vec";
  VectorCPU x_g_out = x_g.matDeviceToHost();
  writeArrayToFile(file_out, x_g_out.getRows(), x_g_out.getColumns(), x_g_out.getMat());
  cusparseStop();
  cublasStop();
  printf("---------------------------------------------\n");
  return duration;
};

float lsqrCPU(unsigned A_r, unsigned A_c, double *A, unsigned b_r, unsigned b_c, double *b, std::string implem, std::string fout) {
  printf("---------------------------------------------\n");
  printf("%s\n\nAx=b where A(%d,%d) and b(%d,1)\n", implem.c_str(), A_r, A_c, b_r);
  VectorCPU A_cpu(A_r, A_c, A);
  VectorCPU b_cpu(b_r, b_c, b);
  std::clock_t c_start = std::clock();
  VectorCPU x = lsqr<VectorCPU, VectorCPU>(A_cpu, b_cpu);
  std::clock_t c_end = std::clock();
  float duration = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
  printf("CPU time used = %f ms for lsqr\n", duration);
  std::string file_out = "output/" + NOW + "/"+ std::to_string(A_c) + "_1_x_" + fout + ".vec";
  writeArrayToFile(file_out, x.getRows(), x.getColumns(), x.getMat());
  printf("---------------------------------------------\n");
  return duration;
};

namespace fs = std::filesystem;

int main() {
  std::vector<std::string> fileOut = {"Cpp-DENSE", "CUDA-DENSE", "CUDA-SPARSE", "CUBLAS-DENSE", "CUSPARSE-SPARSE"};
  std::vector<std::string> implementations = {"1. CPU C++ - Dense Matrix Implementation", "2. GPU CUDA - Dense Matrix Implementation",
                                              "3. GPU CUDA - Sparse Matrix Implementation", "4. GPU CUBLAS -  Dense Matrix Implementation",
                                              "5. GPU CUBLAS/CUSPARSE - Sparse Matrix Implementation"};
  printf("\nWelcome to law-dwg's LSQR-CUDA solver!\n\n"
         "The LSQR-algorithm is used to iteratively solve for the vector x in the following expression: "
         "A * x = b,\n"
         "where A is typically a large matrix of size (m*n) and sparsity sp, and b and x are vectors of sizes (m*1) and (n*1), respectively\n\n"
         "LSQR was designed and first authored by C. C. Paige and M. A. Saunders here: "
         "https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf\n\n"
         "This program runs 5 different implementations of the LSQR algorithm using C++ and CUDA for both dense and sparse A-matricies:\n"
         "    %s\n"
         "    %s\n"
         "    %s\n"
         "    %s\n"
         "    %s\n\n",
         implementations[0].c_str(), implementations[1].c_str(), implementations[2].c_str(), implementations[3].c_str(), implementations[4].c_str());
  printf("This solver requires inputs, feel free to add your own:\n"
         "    A - 'input/m_n_A_sp.mat'\n"
         "    b - 'input/m_1_b.vec'\n"
         "and writes outputs to:\n"
         "    x - 'output/%s/n_1_x_implementation.vec'\n"
         "    report - 'output/%s/%s_LSQR-CUDA.csv'\n\n",NOW.c_str(),NOW.c_str(),NOW.c_str());
  std::cout << "Would you like me to make some test data for you? (y/n): ";
  bool matBuild = yesNo();
  if (matBuild) { // build matrices
    unsigned start = 100;
    unsigned end = 100;
    unsigned increment = 500;
    unsigned numOfTests = ((end - start) / increment) + 1;
    printf("\nGreat, I will create %d sets of inputs for you\n\nWhat sparsity should matrix A have? Please enter a number between 0.0-0.95: ",
           numOfTests);
    double sp; 
    bool test = valInput<double>(0.0, 0.95, sp);
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
  fs::create_directory("output/"+NOW);
  auto now = std::chrono::system_clock::now();
  std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  tm t = *localtime(&now_time);
  std::cout << std::ctime(&now_time) << std::endl;
  std::ofstream report;
  std::stringstream reportName;
  //reportName << "output/" << t.tm_year + 1900 << "-" << t.tm_mon + 1 << "-" << t.tm_mday << "T" << t.tm_hour << ":" << t.tm_min << "_LSQR-CUDA.csv";
  reportName << "output/" << NOW << "/" << timeNowString() << "_LSQR-CUDA.csv";
  report.open(reportName.str());
  report << "IMPLEMENTATION,A_ROWS,A_COLUMNS,SPARSITY,TIME(ms),SPEEDUP\n";
  report.close();
  report.open(reportName.str(), std::ios_base::app);
  while (it != sorted_by_name.end()) { // iterate through sorted files
    cudaErrCheck(cudaDeviceReset());
    std::string file1, file2;
    std::string A_file, b_file;
    A_file = *it;
    printf("%s\n",A_file.c_str());
    // parse filename  
    std::vector<std::string> delim{"/\\", ".", "_"};
  
    size_t slash = A_file.find_last_of(delim[0]);         // file prefix location
    A_file.erase(A_file.begin(), A_file.begin() + slash + 1); // remove file prefix
    size_t dot = A_file.find_last_of(delim[1]);           // file extension location
    
    std::string ext = A_file;
    std::string A_rowStr = A_file;
    size_t unders = A_file.find("_");
    ext.erase(ext.begin(),ext.begin()+dot+1);
    printf("159\n");
    A_rowStr.erase(A_rowStr.begin()+unders,A_rowStr.end());
    
    if (ext == "mat") {
      b_file = "input/"+A_rowStr + "_1_b.vec";
      std::vector<std::string> files{*it, b_file};
      printf("%s %s\n",A_file.c_str(), b_file.c_str());
      unsigned A_rows, A_cols, b_rows, b_cols;
      std::vector<double> A, b;
      double sp;
      for (auto file : files) {
        fileParserLoader(file, A_rows, A_cols, A, b_rows, b_cols, b, sp);
      }
      printf("sp = %f\n",sp);
      std::string sparsityStr = std::to_string(sp);
      sparsityStr.erase ( sparsityStr.find_last_not_of('0') + 1, std::string::npos );
      bool A_sizecheck, b_sizecheck, Ab_rowscheck, all_checks;
      A_sizecheck = A.size() == A_rows * A_cols && A_rows != 0 && A_cols != 0;
      b_sizecheck = b.size() == b_rows * b_cols && b_rows != 0 && b_cols == 1;
      Ab_rowscheck = A_rows == b_rows;
      all_checks = A_sizecheck && b_sizecheck && Ab_rowscheck;
      if (!all_checks) {
        printf("\n\nERROR, please check the matrix file naming convention (\"NumOfRows_NumOfCols_A_sp.mat\" and "
               "\"NumOfRows_1_b.vec\" format) and make sure the naming convention (rows * columns) matches the number of values in each file\n\n");
        exit(1);
      }
      /** 1. CPU C++ - Dense Matrix Implementation */
      long double CPU_Cpp_ms;
      if (true) {
        CPU_Cpp_ms = lsqrCPU(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[0], fileOut[0]);
        report << fileOut[0] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << ",0,"
               << std::to_string(CPU_Cpp_ms) << ","
               << "0"
               << "\n";
      }
      /** 2. GPU CUDA - Dense Matrix Implementation */
      if (true) {
        float CUDA_DENSE_ms = lsqrGPU<VectorCUDA, VectorCUDA>(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[1], fileOut[1], sp);
        cudaErrCheck(cudaDeviceReset());
        report << fileOut[1] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << ",0,"
               << std::to_string(CUDA_DENSE_ms) << "," << CPU_Cpp_ms / CUDA_DENSE_ms << "\n";
      }
      if (true) {
        float CUDA_SPARSE_ms = lsqrGPU<MatrixCUDA, VectorCUDA>(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[2], fileOut[2], sp);
        cudaErrCheck(cudaDeviceReset());
        report << fileOut[2] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << "," << sparsityStr << ","
               << std::to_string(CUDA_SPARSE_ms) << "," << CPU_Cpp_ms / CUDA_SPARSE_ms << "\n";
      }

      if (true) {
        float CUBLAS_DENSE_ms =
            lsqrGPU<VectorCUBLAS, VectorCUBLAS>(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[3], fileOut[3], sp);
        cudaErrCheck(cudaDeviceReset());
        report << fileOut[3] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << ",0,"
               << std::to_string(CUBLAS_DENSE_ms) << "," << CPU_Cpp_ms / CUBLAS_DENSE_ms << "\n";
      }

      if (true) {
        float CUSPARSE_SPARSE_ms =
            lsqrGPU<MatrixCUSPARSE, VectorCUBLAS>(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[4], fileOut[4], sp);
        cudaErrCheck(cudaDeviceReset());
        report << fileOut[4] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << "," << sparsityStr << ","
               << std::to_string(CUSPARSE_SPARSE_ms) << "," << CPU_Cpp_ms / CUSPARSE_SPARSE_ms << "\n";
      }

      cudaLastErrCheck();
    };
    ++it;
  }
  report.close();
}