#include "lsqr_gpu.cuh"
#include "../matrixBuilder.h"
#include "matVec_gpu.cuh"
#include <limits>

//double D2Norm(double a, double b) {
//  const double scale = std::abs(a) + std::abs(b);
//  const double zero = 0.0;
//
//  if (scale == zero) {
//    return zero;
//  }
//
//  const double sa = a / scale;
//  const double sb = b / scale;
//
//  return scale * sqrt(sa * sa + sb * sb);
//};

bool compareMat(double *MC, int rowC, int colC, double *MG, int rowG, int colG) {
  bool same = true;
  double epsilon = 1e-9;
  if (rowC != rowG || colC != colG || !same) {
    printf("MATRICIES SIZE  DO NOT MATCH matCPU(%d x %d) != matGPU(%d, %d)\n", rowC, colC, rowG, rowC);
    same = false;
  }
  if (same) {
    for (int i = 0; i < rowC * colC; i++) {
      // printf("MG[%d] = %f, MC[%d] = %f\n", i, MG[i], i, MC[i]);
      // printf("DIFF = %f, %f == %f\n", std::abs(MG[i] - MC[i]), MG[i], MC[i]);
      if (!(std::abs(MG[i] - MC[i]) < epsilon)) {
        printf("MATRICIES SIZE (%d x %d) DO NOT MATCH DISCREPANCY AT INDEX %d; DIFF = %f, %f "
               "== %f\n",
               rowC, colC, i, std::abs(MG[i] - MC[i]), MG[i], MC[i]);
        printf("MG[%d]=%f, MC[%d]=%f\nMG[%d]=%f, MC[%d]=%f\nMG[%d]=%f, "
               "MC[%d]=%f\nMG[%d]=%f, MC[%d]=%f\n",
               i - 1, MG[i - 1], i - 1, MC[i - 1], i, MG[i], i, MC[i], i + 1, MG[i + 1], i + 1, MC[i + 1], i + 2, MG[i + 2], i + 2, MC[i + 2]);
        printf("LAST ELEMENTS MG[%d]=%f, MC[%d]=%f\n", (rowC * colC) - 1, MG[(rowC * colC) - 1], (rowC * colC) - 1, MC[(rowC * colC) - 1]);
        same = false;
        break;
      }
    }
  };
  if (same) {
    printf("MATRICES MATCH FOR (%d x %d)\n", rowC, colC);
  };
  return same;
};

bool compareVal(double *VC, double *VG) {
  typedef std::numeric_limits<double> dbl;
  bool same = false;
  printf("GPU: %20f\n", *VG);
  printf("CPU: %20f\n", *VC);
  std::cout.precision(dbl::max_digits10);
  std::cout << *VC << std::endl;
  std::cout << *VG << std::endl;
  std::cout << std::abs(*VC - *VG) << std::endl;
  if (std::abs(*VC - *VG) < 1e-15) {
    printf("THEY ARE SAME\n");
    same = true;
  } else {
    printf("THEY ARE NOT SAME\n");
  }
  return same;
}

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

Vector_GPU lsqr_gpu(Vector_GPU &A, Vector_GPU &b) {
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
  // v.printmat();

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
    c = rhobar / rho;               // c_i
    s = beta / rho;                 // s_i
    theta = s * alpha;              // theta_i+1
    rhobar = -c * alpha;            // rhobar_i+1
    phi = c * phibar;               // phi_i = c_i*phibar_i
    phibar = s * phibar;            // phibar_i+1 = s_i*phibar_i

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
    // x.printmat();
    printf("%f\n", Arnorm);
  } while (istop == 0);
  return x;
};