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
#include <algorithm>
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
  std::string sparsityStr = ((int)fout.find("SPARSE")) > (-1) ? "-" + std::to_string((int)(sp * 100)) : "";
  std::string file_out = "output/" + NOW + "/" + std::to_string(A_c) + "_1_x_" + fout + sparsityStr + ".vec";
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
  std::string file_out = "output/" + NOW + "/" + std::to_string(A_c) + "_1_x_" + fout + ".vec";
  writeArrayToFile(file_out, x.getRows(), x.getColumns(), x.getMat());
  printf("---------------------------------------------\n");
  return duration;
};

namespace fs = std::filesystem;

int main() {
  /** Intro */
  std::vector<std::string> fileName = {"Cpp-DENSE", "CUDA-DENSE", "CUDA-SPARSE", "CUBLAS-DENSE", "CUSPARSE-SPARSE"};
  std::vector<std::string> implementations = {"1. CPU C++ - Dense Matrix Implementation", "2. GPU CUDA - Dense Matrix Implementation",
                                              "3. GPU CUDA - Sparse Matrix Implementation", "4. GPU CUBLAS -  Dense Matrix Implementation",
                                              "5. GPU CUBLAS/CUSPARSE - Sparse Matrix Implementation"};
  printf("\nWelcome to law-dwg's LSQR-CUDA solver!\n\n"
         "The LSQR-algorithm is used to iteratively solve for the vector x in the following expression: "
         "A * x = b,\n"
         "where A is typically a large matrix of size (m*n) and sparsity sp, and b and x are vectors of sizes (m*1) and (n*1), respectively\n\n"
         "LSQR was designed and first authored by C. C. Paige and M. A. Saunders here: "
         "https://web.stanford.edu/group/SOL/software/lsqr/lsqr-toms82a.pdf\n\n"
         "This program runs 5 different implementations of the LSQR algorithm using C++ and CUDA for both dense and sparse A:\n"
         "    %s\n"
         "    %s\n"
         "    %s\n"
         "    %s\n"
         "    %s\n\n",
         implementations[0].c_str(), implementations[1].c_str(), implementations[2].c_str(), implementations[3].c_str(), implementations[4].c_str());
  printf("This solver requires the following inputs (feel free to add your own):\n"
         "    A - 'input/m_n_A_sp.mat' (in dense format)\n"
         "    b - 'input/m_1_b.vec'\n"
         "and writes the corresponding outputs to:\n"
         "    x - 'output/%s/n_1_x_implementation.vec'\n"
         "\nA report csv of duration of executions is written to:\n"
         "    report - 'output/%s/%s_LSQR-CUDA.csv'\n\n",
         NOW.c_str(), NOW.c_str(), NOW.c_str());

  /** Create inputs */
  std::cout << "Would you like me to make some test data for you? (y/n): ";
  bool matBuild = yesNo();
  if (matBuild) { // build matrices
    unsigned start = 1000;
    unsigned end = 1500;
    unsigned increment = 500;
    unsigned numOfTests = ((end - start) / increment) + 1;
    system("exec rm -r input/*");
    printf("\nGreat, I will create %d set(s) of inputs for you\n\nWhat sparsity should matrix A have? Please enter a number between 0.0-0.95: ",
           numOfTests);
    double sp;
    bool test = valInput<double>(0.0, 0.95, sp);
    printf("Matricies (A) will be built with sparsity %0.2f\n\n", sp);
    for (int i = start; i <= end; i += increment) {
      matrixBuilder(i, i, sp, "input/", "A");
      matrixBuilder(i, 1, 0, "input/", "b");
    }
  }

  /** Check for inputs */
  std::string input = "input/";
  std::set<fs::path> sorted_by_name;
  for (auto &entry : fs::directory_iterator(input)) {
    sorted_by_name.insert(entry.path()); // alphabetically sort inputs
  };
  if (sorted_by_name.size() == 0) { // no inputs
    std::cout << "Looks like there are no files in the input folder. Please add your own matricies in \"#rows_#cols_A_#sparsity.mat\" and "
                 "\"#rows_1_b.vec\" format, or rerun the "
                 "program to autobuild matrices\n"
              << std::endl;
    return 0;
  };
  std::set<fs::path>::iterator it = sorted_by_name.begin();

  /** Prepare report */
  std::ofstream report;
  std::stringstream reportName;
  fs::create_directory("output/" + NOW);
  reportName << "output/" << NOW << "/" << timeNowString() << "_LSQR-CUDA.csv";
  report.open(reportName.str());
  report << "IMPLEMENTATION,A_ROWS,A_COLUMNS,SPARSITY,TIME(ms)\n";
  report.close();

  printf("\nCUDA capable device check:\n");
  /** Iterate through files in inputs/ */
  if (checkDevice() > 0) { // only uses first device
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0); // first device
    std::ofstream deviceProps;
    std::stringstream devicePropsFile;
    devicePropsFile << "output/" << NOW << "/deviceProps.csv";
    deviceProps.open(devicePropsFile.str());
    deviceProps << "DEVICE_NAME,COMPUTE_CAPABILITY\n" << deviceProp.name << "," << deviceProp.major << "." << deviceProp.minor << "\n";
    deviceProps.close();
    while (it != sorted_by_name.end()) { // iterate through sorted files
      cudaErrCheck(cudaDeviceReset());

      // parse filename
      std::string A_file, b_file, ext, A_rowStr;
      A_file = *it;
      std::vector<std::string> delim{"/\\", ".", "_"};
      size_t slash = A_file.find_last_of(delim[0]);             // file prefix location
      A_file.erase(A_file.begin(), A_file.begin() + slash + 1); // remove file prefix
      ext = A_file;
      size_t dot = A_file.find_last_of(delim[1]);    // file extension location
      ext.erase(ext.begin(), ext.begin() + dot + 1); // find file extension

      if (ext == "mat") { // A mat
        A_rowStr = A_file;
        size_t unders = A_file.find("_");                          // find number of rows
        A_rowStr.erase(A_rowStr.begin() + unders, A_rowStr.end()); // find number of rows
        b_file = "input/" + A_rowStr + "_1_b.vec";                 // corresponding b vec

        std::vector<std::string> files{*it, b_file}; // inputs
        unsigned A_rows, A_cols, b_rows, b_cols;     // dimensions
        std::vector<double> A, b;                    // data
        double sp;                                   // sparsity
        for (auto file : files) {
          fileParserLoader(file, A_rows, A_cols, A, b_rows, b_cols, b, sp); // load inputs
        }
        std::string sparsityStr = std::to_string(sp);
        sparsityStr.erase(sparsityStr.find_last_not_of('0') + 1, std::string::npos); // remove trailing 0

        sizeCheck(A_rows, A_cols, A, b_rows, b_cols, b, *it, b_file); // sanity check

        /** Run LSQR implementations */
        // 1. CPU C++ - Dense Matrix Implementation
        report.open(reportName.str(), std::ios_base::app);
        long double CPU_Cpp_ms;
        if (true) {
          CPU_Cpp_ms = lsqrCPU(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[0], fileName[0]);
          report << fileName[0] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << ",0," << std::to_string(CPU_Cpp_ms) << "\n";
        };
        // 2. GPU CUDA - Dense Matrix Implementation
        if (true) {
          float CUDA_DENSE_ms =
              lsqrGPU<VectorCUDA, VectorCUDA>(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[1], fileName[1], sp);
          cudaErrCheck(cudaDeviceReset());
          report << fileName[1] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << ",0," << std::to_string(CUDA_DENSE_ms) << "\n";
        };
        // 3. GPU CUDA - Sparse Matrix Implementation
        if (true) {
          float CUDA_SPARSE_ms =
              lsqrGPU<MatrixCUDA, VectorCUDA>(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[2], fileName[2], sp);
          cudaErrCheck(cudaDeviceReset());
          report << fileName[2] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << "," << sparsityStr << ","
                 << std::to_string(CUDA_SPARSE_ms) << "\n";
        };
        // 4. GPU CUBLAS -  Dense Matrix Implementation
        if (true) {
          float CUBLAS_DENSE_ms =
              lsqrGPU<VectorCUBLAS, VectorCUBLAS>(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[3], fileName[3], sp);
          cudaErrCheck(cudaDeviceReset());
          report << fileName[3] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << ",0," << std::to_string(CUBLAS_DENSE_ms) << "\n";
        };
        // 5. GPU CUBLAS/CUSPARSE - Sparse Matrix Implementation
        if (true) {
          float CUSPARSE_SPARSE_ms =
              lsqrGPU<MatrixCUSPARSE, VectorCUBLAS>(A_rows, A_cols, A.data(), b_rows, b_cols, b.data(), implementations[4], fileName[4], sp);
          cudaErrCheck(cudaDeviceReset());
          report << fileName[4] << "," << std::to_string(A_rows) << "," << std::to_string(A_cols) << "," << sparsityStr << ","
                 << std::to_string(CUSPARSE_SPARSE_ms) << "\n";
        };
        cudaLastErrCheck(); // in case an error was missed
        report.close();     // write to report
      };
      ++it;
    }
  } else {
    printf("This program requires a cuda capable device. If you do have the proper hardware, please double check your drivers and installation of "
           "cuda\n");
  };

  return 0;
};