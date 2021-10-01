#include "../cpu/lsqr_cpu.hpp"
#include "../cpu/matVec_cpu.hpp"
#include "../cpu/matrixBuilder.hpp"
// #include "lsqr_gpu.cuh"
#include "matVec_gpu.cuh"
#include <cassert>
#include <chrono>
#include <ctime>
#include <ctype.h>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <set>
#include <sstream>
#include <stdio.h>
#include <string>
#include <time.h>
#include <vector>

namespace fs = std::filesystem;

int checkDevice() {
  // Check Cuda Capabale Device
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;

  if (deviceCount > 0) {
    for (device = 0; device < deviceCount; ++device) {
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, device);
      printf("Device %s has compute capability %d.%d.\n", deviceProp.name, deviceProp.major, deviceProp.minor);
      printf("Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
      printf("Clock rate: %d Hz\n", deviceProp.clockRate);
      printf("Total amount of global memory: %d KB\n", deviceProp.totalGlobalMem / 1024);
      printf("Total amount of constant memory: %d KB\n", deviceProp.totalConstMem / 1024);
      printf("Total amount of shared memory per block: %d KB\n", deviceProp.sharedMemPerBlock / 1024);
      printf("Total amount of shared memory per SM: %d KB\n", 64);
      printf("Warp size: %d\n", deviceProp.warpSize);
      printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
      printf("Maximum number of blocks per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock);
      printf("Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
      printf("Maximum number of warps per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor / 32);
      printf("Maximum Grid size: (%d,%d,%d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
      printf("Maximum block dimension: (%d,%d,%d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    }
  } else {
    printf("NO CUDA DEVICE AVAILABLE");
  }
  return deviceCount;
};

template <typename Vec> Vec lsqr(Vec &A, Vec &b) {
  // Iteration
  unsigned int istop = 0;
  unsigned int itn = 0;
  double rho = 0;
  double phi = 0;
  double c = 0;
  double s = 0;
  double theta = 0;
  double tau = 0;
  double res = 0;
  double res1 = 0;
  double ddnorm = 0;
  double Anorm = 0;
  double Acond = 0;
  double damp = 0;
  double dnorm = 0;
  double dknorm = 0;
  double res2 = 0;
  double xnorm = 0;
  double xxnorm = 0;
  double z = 0;
  double sn2 = 0;
  double ctol = 0;
  double rtol = 0;
  double dampsq = 0;
  double cs2 = -1;
  double atol = 1e-8;
  double btol = 1e-8;
  double conlim = 1e8;
  double res_old = 1e10;
  double epsilon = 1e-16;
  double alpha = 0;
  double beta = 0;
  Vec u, v, w, res_v;
  if (conlim > 0) {
    ctol = 1 / conlim;
  };
  dampsq = damp * damp;

  /*1. Initialize*/
  Vec x(A.getColumns(), 1);
  beta = b.Dnrm2();
  // std::cout << beta << std::endl;
  if (beta > 0) {
    u = b * (1 / beta);
    v = A.transpose() * u;
    alpha = v.Dnrm2();
  } else {
    v = x;
    alpha = 0;
  };
  // std::cout << alpha << std::endl;
  // v.print();

  if (alpha > 0) {
    v = v * (1 / alpha);
    w = v;
  };
  double rhobar = alpha;
  double phibar = beta;

  // Norms
  double rnorm = beta;
  double r1norm = rnorm;
  double r2norm = rnorm;
  double Arnorm = alpha * beta;

  // 2. For i=1,2,3....
  printf("2. For i=1,2,3....\n");
  do {
    if (itn == A.getRows() / 2 || itn == A.getRows() || itn == (A.getRows() + A.getRows() / 2))
      printf("itn = %d\n", itn);
    itn++;
    Vec A_T = A.transpose();

    // 3. Continue the bidiagonialization
    /* Important equations for understanding.
    ubar_i+1 = beta_i+1 * u_i+1 = (A*v_i) - (alpha_i*u_i)
    beta_i+1 = ||ubar_i+1||
    u_i+1 =  ubar_i+1 * (1/beta_i+1)
    vbar_i+1 = alpha_i+1 * v_i+1 = (A_t*u_i+1) - (beta_i*v_i)
    alpha_i+1 = ||vbar_i+1||
    v_i+1 =  vbar_i+1 * (1/alpha_i+1)
    */
    // printf("3. Continue the bidiagonialization\n");
    u = A * v - u * alpha; // ubar_i+1

    beta = u.Dnrm2(); // beta_i+1 = ||ubar_i+1||
    if (beta > 0) {
      u = u * (1 / beta);         // u_i+1
      v = (A_T * u) - (v * beta); // vbar_i+1
      alpha = v.Dnrm2();          // alpha_i+1
      if (alpha > 0) {
        v = v * (1 / alpha); // v_i+1
      }
    }

    // 4. Construct and apply next orthogonal transformation
    // printf("4. Construct and apply next orthogonal transformation\n");

    rho = D2Norm(rhobar, beta); // rho_i
    c = rhobar / rho;           // c_i
    s = beta / rho;             // s_i
    theta = s * alpha;          // theta_i+1
    rhobar = -c * alpha;        // rhobar_i+1
    phi = c * phibar;           // phi_i = c_i*phibar_i
    phibar = s * phibar;        // phibar_i+1 = s_i*phibar_i

    // used for stopping critera
    tau = s * phi;
    Arnorm = alpha * std::abs(tau);

    // 5. Update x,w
    // save values for stopping criteria
    Vec dk = w * (1 / rho);
    double dknrm2 = dk.Dnrm2();
    dknorm = dknrm2 * dknrm2 + ddnorm;

    /* Important equations
    x_i = x_i-1 + (phi_i/rho_i) *w_i
    w_i+1 = v_i+1 - (theta_i+1/rho_i)*w_i
    */
    x = x + w * (phi / rho);
    w = v - (w * (theta / rho));
    // residual
    res_v = b - (A * x);
    res = res_v.Dnrm2();

    // 6. Test for convergence
    // printf("6. Test for convergence\n");
    /*Test 1 for convergence
    stop if ||r|| =< btol*||b|| + atol*||A||*||x||
    */
    if (res <= (btol * b.Dnrm2() + atol * A.Dnrm2() * x.Dnrm2())) {
      istop = 1;
    }

    /*Test 2 for convergence
    stop if ||A_T*r||/||A||*||r|| <= atol
    */
    if (Arnorm / (A.Dnrm2() * res) <= atol) {
      istop = 2;
    }

    /*Test 3 for convergence
    stop if cond(A) => conlim

    Acond = A.Dnrm2() * sqrt(ddnorm);
    if (Acond < conlim){
        istop=3;
    }

    res1 = phibar*phibar;
    double test3 = 1/ (Acond + epsilon);
    std::cout.precision(25);
    std::cout<<std::fixed<<res<<std::endl;

    if(res <= btol * b.Dnrm2() + atol * A.Dnrm2() * x.Dnrm2());

    if(test3<=ctol){
        istop=3;
        printf("%i\n",itn);
    };
    double test2 = Arnorm / (A.Dnrm2()*res + epsilon);
    if( 1 + test2 <= 1){
        istop=5;
        printf("%i\n",itn);
    };*/

    // printf("iteration %i\n", itn);
    // printf("istop %i\n", istop);
    // x.print();
    // printf("%f\n", Arnorm);
  } while (istop == 0 && itn < 2 * A.getRows());
  printf("ran through %d iterations \n", itn);
  return x;
}

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
    sp = 0;
    std::cout << "Building A Matrices of sparsity " << sp << "\n";
    for (int i = 1000; i < 1100; i += 100) {
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
    //   continue;
    // } else {
    //   printf("Error, please check the matrix file naming convention (\"NumOfRows_NumOfCols_A.txt\" and "
    //          "\"NumOfRows_1_b.txt\" format) and make sure the naming convention (rows * columns) matches the number of values in each file\n");
    //   return 0;
    // }
    printf("Running lsqr-CPU implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
    Vector_CPU A_c(A_rows, A_cols, A.data());
    Vector_CPU b_c(b_rows, b_cols, b.data());
    // A_c.print();
    // b_c.print();
    std::clock_t c_start = std::clock();
    Vector_CPU x_c = lsqr<Vector_CPU>(A_c, b_c);
    std::clock_t c_end = std::clock();
    long double time_elapsed_ms = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
    std::cout << "CPU time used = " << time_elapsed_ms << " ms for lsqr\n";
    std::string file_out = "output/" + std::to_string(A_cols) + "_1_x_CPU.txt";
    writeArrayToFile(file_out, x_c.getRows(), x_c.getColumns(), x_c.getMat());
    printf("---------------------------------------------\n");
    printf("Running lsqr-GPU implementation\nAx=b where A(%d,%d) and b(%d,1)\n", A_rows, A_cols, b_rows);
    Vector_GPU A_g(A_rows, A_cols, A.data());
    Vector_GPU b_g(b_rows, b_cols, b.data());
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    Vector_GPU x_g = lsqr<Vector_GPU>(A_g, b_g);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU time used = %f ms for lsqr\n", milliseconds);
    file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU.txt";
    Vector_CPU x_g_out = x_g.matDeviceToHost();
    writeArrayToFile(file_out, x_g_out.getRows(), x_g_out.getColumns(), x_g_out.getMat());
    cublasDestroy(handle);
    cudaDeviceReset();
    cublasReset();
  }
  cublasDestroy(handle);
}