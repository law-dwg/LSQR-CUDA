#include "cpu/lsqr_cpu.h"
#include "gpu/lsqr_gpu.cuh"
#include "cpu/matVec_cpu.h"
#include "gpu/matVec_gpu.cuh"
#include "matrixBuilder.h"
#include <cassert>
#include <ctype.h>
#include <filesystem>
#include <iostream>
#include <math.h>
#include <cuda.h>
#include <set>
#include <sstream>
#include <stdio.h>
#include <string>
#include <vector>

namespace fs = std::filesystem;

template <typename Mat, typename Vec> Vec lsqr(Mat &A, Vec &b) {
  // Iteration
  unsigned int istop, itn;
  istop = itn = 0;
  double rho, phi, c, s, theta, tau, res, res1, atol, btol, ddnorm, Anorm, Acond, damp, dnorm, dknorm, res2, xnorm, xxnorm, z, sn2;
  double ctol, rtol, conlim, cs2, dampsq;
  rtol = ctol = damp = Anorm = Acond = ddnorm = res2 = xnorm = xxnorm = z = sn2 = 0;
  cs2 = -1;
  atol = btol = 1e-8;
  conlim = 1e8;

  if (conlim > 0) {
    ctol = 1 / conlim;
  };
  dampsq = damp * damp;
  double res_old = 1e10;
  double epsilon = 1e-16;
  /*1. Initialize*/
  Vec u, v, w, res_v;
  double alpha;
  Vec x(A.getColumns(), 1);
  double beta = b.Dnrm2();
  std::cout << beta << std::endl;
  if (beta > 0) {
    u = b * (1 / beta);
    v = A.transpose() * u;
    alpha = v.Dnrm2();
  } else {
    v = x;
    alpha = 0;
  };
  //std::cout << alpha << std::endl;
  //v.print();

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
    itn++;
    Mat A_T = A.transpose();

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
    beta = u.Dnrm2();      // beta_i+1 = ||ubar_i+1||
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
    dknorm = dk.Dnrm2() * dk.Dnrm2() + ddnorm;
    /* Important equations
    x_i = x_i-1 + (phi_i/rho_i) *w_i
    w_i+1 = v_i+1 - (theta_i+1/rho_i)*w_i
    */
    x = x + w * (phi / rho);
    w = v - (w * (theta / rho));
    // residual
    res_v = b - A * x;

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
  } while (istop == 0 && itn < 2 * A.getColumns());

  return x;
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
    for (int i = 10; i < 50; i += 10) {
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
    Vector_CPU x_c = lsqr<Vector_CPU, Vector_CPU>(A_c, b_c);
    std::string file_out = "output/" + std::to_string(A_cols) + "_1_x_CPU.txt";
    writeArrayToFile(file_out, x_c.getRows(), x_c.getColumns(), x_c.getMat());
    Vector_GPU A_g(A_rows, A_cols, A.data());
    Vector_GPU b_g(b_rows, b_cols, b.data());
    Vector_GPU test_g = A_g*b_g;
    Vector_CPU test_g_out = test_g.matDeviceToHost();
    Vector_CPU test_c = A_c* b_c;
    test_c.print();
    test_g_out.print();
    Vector_GPU x_g = lsqr<Vector_GPU,Vector_GPU>(A_g, b_g);
    file_out = "output/" + std::to_string(A_cols) + "_1_x_GPU.txt";
    Vector_CPU x_g_out = x_g.matDeviceToHost();
    writeArrayToFile(file_out, x_g_out.getRows(), x_g_out.getColumns(), x_g_out.getMat());
    
  }
}