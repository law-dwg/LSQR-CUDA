#include "matVec_gpu.cuh"
//#include "../cpu/matVec_cpu.h"
#include "../cpu/lsqr_cpu.h"
#include "device_launch_parameters.h"
#include "lsqr_gpu.cuh"
#include <assert.h>
#include <chrono>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> //srand, rand
#include <string.h>
#include <time.h>

int main() {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  checkDevice();

  double *a_heap = new double[3 * 3]{1.0, -2.2, 0, 3.2, 4.0, 0, 5.0, 0, 6.1};
  double *b_heap = new double[3 * 1]{0.01, 0.2, -0.2};
  Vector_CPU A_c(3, 3, a_heap);
  Vector_CPU b_c(3, 1, b_heap);
  A_c.print();
  b_c.print();
  Vector_GPU A = A_c;
  Vector_GPU b = b_c;

  unsigned int istop, itn = 0;

  double ddnorm, Anorm, Acond, damp, dnorm, dknorm, res2, xnorm, xxnorm, z, sn2, rtol = 0;
  double rho, phi, c, s, theta, tau, res, res1;
  double cs2 = -1;
  double atol, btol = 1e-8;

  double conlim = 1e8;
  double ctol = 0;
  if (conlim > 0) {
    ctol = 1 / conlim;
  };

  double dampsq = damp * damp;

  double res_old = 1e10;
  double epsilon = 1e-16;

  /*1. Initialize*/
  Vector_GPU u, v, w, res_v;
  double alpha;
  Vector_GPU x(A.getColumns(), 1);
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
  std::cout << alpha << std::endl;
  v.printmat();

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
    Vector_GPU A_T = A.transpose();

    // 3. Continue the bidiagonialization
    /* Important equations for understanding.
    ubar_i+1 = beta_i+1 * u_i+1 = (A*v_i) - (alpha_i*u_i)
    beta_i+1 = ||ubar_i+1||
    u_i+1 =  ubar_i+1 * (1/beta_i+1)
    vbar_i+1 = alpha_i+1 * v_i+1 = (A_t*u_i+1) - (beta_i*v_i)
    alpha_i+1 = ||vbar_i+1||
    v_i+1 =  vbar_i+1 * (1/alpha_i+1)
    */
    printf("3. Continue the bidiagonialization\n");
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
    printf("4. Construct and apply next orthogonal transformation\n");

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
    printf("5. Update x,w\n");
    // save values for stopping criteria
    Vector_GPU dk = w * (1 / rho);
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
    printf("6. Test for convergence\n");
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
    printf("iteration %i\n", itn);
    printf("istop %i\n", istop);
    x.printmat();
    printf("%f\n", Arnorm);
  } while (istop == 0);
  delete a_heap, b_heap;
  cudaError_t err = cudaGetLastError(); // add
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  } // add
  cudaProfilerStop();

  return 0;
};