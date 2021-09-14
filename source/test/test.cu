//#include "../cpu/lsqr_cpu.h"
#include "../cpu/matVec_cpu.h"
#include "../gpu/lsqr_gpu.cuh"
#include "../matrixBuilder.h"
#include "device_launch_parameters.h"
//#include "matCsr_gpu.cuh"
#include "../gpu/matVec_gpu.cuh"
#include <assert.h>
#include <chrono>
#include <cmath>
#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdio.h>  //NULL, printf
#include <stdlib.h> //srand, rand
#include <string.h>
#include <time.h>

int main() {
  checkDevice();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // int a_csr_rows, b_csr_rows, a_csr_cols, b_csr_cols;
  // a_csr_rows = b_csr_rows = a_csr_cols = b_csr_cols = 3;
  // double *a_csr_heap = new double[9]{0, 1, 2, 4, 0, 5, 6, 7, 0};
  // Matrix_CPU S(33, 1);
  // double *b_heap = new double[3]{1, 2, 3};
  // // double *b_heap = new double[100];
  // // for (int i = 0; i < (100); ++i) {
  // //  b_heap[i] = i;
  // //}
  // Matrix_CSR_GPU A(a_csr_rows, a_csr_cols, a_csr_heap);
  // // Matrix_CSR_GPU A(S.rows, S.columns, S.getMat());
  // Vector_GPU b(3, 1, b_heap);
  // A *b;
  // delete a_csr_heap, b_heap;

  double *a_heap, *b_heap;
  int a_rows, b_rows, a_cols, b_cols;
  a_rows = b_rows = 160;
  a_cols = 160;
  b_cols = 1;
  a_heap = new double[a_rows * a_cols];
  // b_heap = new double[b_rows * b_cols];
  // for (int i = 0; i < (a_rows * a_cols); i++) {
  //  if (i < b_rows) {
  //    a_heap[i] = -rands();
  //    b_heap[i] = -rands();
  //  } else {
  //    a_heap[i] = -rands();
  //  }
  //}
  b_heap = new double[b_rows * b_cols]{
      106.966407714283207, 71.122612066451453,  119.292513864348365, 49.308002729960492,  24.358046965410608,  82.271252029795264,
      48.404195639346639,  141.074960017726426, 32.733550264872292,  106.915744715752083, 76.478109025567733,  80.664056324986817,
      77.939250793881413,  62.462340189677462,  -44.040726789585676, 28.102825552074393,  -48.149024649602467, -8.243224585917204,
      32.607365060803204,  154.014853028806328, -9.280470468425364,  50.255545865976202,  99.472020575895186,  123.220937017888502,
      56.211278868164044,  124.586761674524070, -67.044202784000021, 16.389433843293290,  151.350260196737395, 94.527209521039438,
      99.180743762288287,  106.635048653651438, -50.538499923305693, -45.316661348307775, 97.264467041960827,  110.720920165729154,
      83.725570808684310,  -59.890296670363739, -52.387022228789505, 23.421141018340819,  22.297839596677676,  94.759877430303035,
      -37.622384463198927, 91.975462081273108,  88.221091991870182,  82.985239217845205,  -42.630227335438008, -37.505978221665657,
      30.765520195557485,  97.769718765585580,  153.629415019343867, -70.092062052545913, -45.426209497597682, 119.111135922987813,
      147.024000727718686, 26.806074489265228,  110.334053614675810, -91.046837975149742, 128.481637008978225, -63.953189341671504,
      8.556436918463504,   80.135911901947182,  136.240549506163632, -65.884463677386179, 36.669117101749976,  -37.907919618286684,
      -60.380366722700273, 32.261911492196077,  35.653567067607796,  -70.082028069839595, -33.770565062853962, -26.216333445607745,
      59.281152033725405,  -32.222477071419746, -8.180065041696338,  32.302454366665501,  119.333029896917708, 81.003310578022237,
      146.918836984754563, -47.962433671045602, 136.361299002044746, 109.869579209702181, -38.304656888886342, 0.658192480894854,
      17.814838505561653,  17.672316364637766,  59.156428045554435,  71.051862955030501,  -75.497170816707154, 25.203450217075357,
      8.952716965513986,   108.315381231194834, -67.017255283347311, 101.171842000872061, 31.173640223385746,  -8.729058859881945,
      144.424263425077811, -81.668857334823031, -66.191611076372425, -76.644190489888132, 57.161349699025578,  75.950035891925708,
      -1.856989655459188,  99.052835731111728,  62.910474253900702,  -22.421937438243674, 128.141604513352661, -25.467393768389400,
      139.230492079976869, -21.453046206426961, 59.005359871346371,  130.204678093194048, 144.269951583280658, 45.956117618835350,
      16.685418402402689,  -19.247539163905003, -8.922086893711992,  14.408158797990637,  -35.635688778418086, 34.520671497941834,
      152.674968584955508, 36.413719972983628,  46.071110604239607,  105.971465931690275, 147.144280793356586, 46.756581816514526,
      -57.411932645832621, 153.045771691929559, -70.096425449378728, -54.718345821698534, -66.110489791110780, -1.597421880183759,
      127.347529736305745, -83.163769268086384, 131.677814085894909, 161.652487192036176, -76.292626717455931, 31.389653038768984,
      22.896837094842795,  -73.083430631062043, 46.987120063402145,  42.129154346729550,  80.585293530627595,  72.522902176414547,
      -66.381410294571339, 89.797995643499760,  26.040283563947270,  42.384985369416484,  -51.713322461534375, 148.749536940849055,
      -2.821555997157532,  52.752236989710767,  43.018243038763515,  87.019176620554120,  58.068044010240655,  -37.866746912429676,
      131.405806419121546, 37.886879620441277,  -38.065582061768993, 129.878145918854983};
  // writeArrayToFile("input/A.txt", a_rows * a_cols, a_heap);
  // writeArrayToFile("input/b.txt", b_rows * b_cols, b_heap);
  // Vector_CPU A_c(a_rows, a_cols, a_heap);
  double test_beta = 1016.201896;
  Vector_CPU b_c(b_rows, b_cols, b_heap);
  // Vector_GPU A_g(a_rows, a_cols, a_heap);
  Vector_GPU b_g(b_rows, b_cols, b_heap);
  Vector_CPU u_c(b_rows, b_cols, b_heap);
  Vector_GPU u(b_rows, b_cols, b_heap);
  u_c = u_c * (1 / test_beta);
  u = u * (1 / test_beta);
  std::cout << u_c.Dnrm2() << std::endl;
  std::cout << u.Dnrm2() << std::endl;
  // b_g.printmat();
  // Vector_GPU C_g = ((b_g.transpose() * b_g) * b_g.Dnrm2()) * A_g.Dnrm2();
  // Vector_CPU C_c =  ((b_c.transpose() * b_c) * b_g.Dnrm2()) * A_c.Dnrm2();
  double beta = b_g.Dnrm2();
  double beta_g = b_g.Dnrm2();
  double beta_c = b_c.Dnrm2();

  std::cout << std::setprecision(17) << beta_g << std::endl;
  std::cout << std::setprecision(17) << beta_c << std::endl;
  printf("beta_g = %f, beta_c = %f\n", beta_g, beta_c);
  // assert(beta_g == beta_c);
  // Vector_GPU C_g = A_g*beta - A_g*-12.34;
  // Vector_CPU C_c = A_c*beta - A_c*-12.34;
  // Vector_GPU C_g = A_g.transpose()*1.8;
  // Vector_CPU C_c = A_c.transpose()*1.8;
  // C_g = C_g.transpose();
  // C_c = C_c.transpose();
  // Vector_CPU C_g_out = C_g.matDeviceToHost();
  // // C_g_out.print();
  // // C_c.print();
  // bool ans = compareMat(C_g_out.getMat(), C_g_out.getRows(), C_g_out.getColumns(), C_c.getMat(), C_c.getRows(), C_c.getColumns());
  // Matrix_GPU c_c(b_rows, b_cols, b_heap);
  // Matrix_GPU A = A_c;
  // Vector_GPU b = b_c;
  // printf("STARTING LSQR\n");
  // Vector_GPU x_G = lsqr_gpu(A, b);
  // Vector_CPU x_C = lsqr_cpu(A_c, b_c);
  // Vector_CPU x_G_out = x_G.matDeviceToHost();
  // bool ans = compareMat(x_G_out.getMat(), x_G_out.getRows(), x_G_out.getColumns(), x_C.getMat(), x_C.getRows(), x_C.getColumns());
  // x_G_out.print();
  // x_C.print();
  // writeArrayToFile("output/x_c.txt", x_G_out.getRows() * x_G_out.getColumns(), x_G_out.getMat());
  // writeArrayToFile("output/x_g.txt", x_C.getRows() * x_C.getColumns(), x_C.getMat());
  delete a_heap, b_heap;
  cudaError_t err = cudaGetLastError(); // add
  if (err != cudaSuccess) {
    std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
  } // add
  cudaProfilerStop();

  return 0;
};