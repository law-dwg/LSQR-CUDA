#ifndef CAFFE_COMMON_CUH_
#define CAFFE_COMMON_CUH_
#include <cuda.h>
#include "cublas_v2.h"

cudaError_t cudaStat;
cublasHandle_t handle;
cudaStream_t stream;
cublasStatus_t stat1 = cublasCreate(&handle);
cudaError_t cudaStat1 = cudaStreamCreate(&stream);
cublasStatus_t stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
cublasStatus_t stat2 = cublasSetStream(handle, stream);
void cublasReset() {
  stat1 = cublasCreate(&handle);
  cudaStat1 = cudaStreamCreate(&stream);
  stat = cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  stat2 = cublasSetStream(handle, stream);
}

// Operator overloads

// Vector_GPU Vector_GPU::operator*(Vector_GPU &v) {
//   cublasLtMatmul(cublasLtHandle_t               lightHandle,
//     cublasLtMatmulDesc_t           computeDesc,
//     const void                    *alpha,
//     const void                    *A,
//     cublasLtMatrixLayout_t         Adesc,
//     const void                    *B,
//     cublasLtMatrixLayout_t         Bdesc,
//     const void                    *beta,
//     const void                    *C,
//     cublasLtMatrixLayout_t         Cdesc,
//     void                          *D,
//     cublasLtMatrixLayout_t         Ddesc,
//     const cublasLtMatmulAlgo_t    *algo,
//     void                          *workspace,
//     size_t                         workspaceSizeInBytes,
//     cudaStream_t                   stream);
// }

double Vector_GPU::Dnrm2() {
  // int blockX = ((this->h_rows * this->h_columns + TILE_DIM_X - 1) / TILE_DIM_X);
  // dim3 blocks(blockX, 1);
  double h_out;
  int *version;
  double *d_out;
  double zero = 0.0;
  // double c_d_mat[h_rows*h_columns];
  int incre = 1;
  gpuErrchk(cudaMalloc(&d_out, sizeof(double)));
  // gpuErrchk(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  // gpuErrchk(cudaMemcpy(&c_d_mat,d_mat, sizeof(double)*h_rows*h_columns,cudaMemcpyDeviceToHost));

  // for (int i = 0; i<100;++i){
  //  printf("%f\n",c_d_mat[i]);
  //}
  // gpuErrchk(cudaMalloc(&c_d_mat, sizeof(double)*h_rows*h_columns));
  // gpuErrchk(cudaMemcpy(&c_d_mat, d_mat, sizeof(double) * h_columns * h_rows, cudaMemcpyDeviceToDevice));
  int size = (this->h_rows * this->h_columns);

  // stat = cublasSetVector(size, sizeof(double), (void **)&c_d_mat, incre, d_d_mat, incre);
  // if (stat != CUBLAS_STATUS_SUCCESS) {
  //   printf ("CUBLAS Version failed\n");
  //   return EXIT_FAILURE;
  // }

  cublasStatus_t stat3 = cublasDnrm2(handle, size, this->d_mat, incre, d_out);
  cudaDeviceSynchronize();
  if (stat3 != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS DNRM2 failed\n");
    return EXIT_FAILURE;
  }

  // stat = cublasDestroy(handle);
  // if (stat != CUBLAS_STATUS_SUCCESS) {
  //   printf ("CUBLAS destroy failed\n");
  //   return EXIT_FAILURE;
  // }
  // cublasDestroy(handle);
  gpuErrchk(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  assert(!(h_out != h_out));
  return h_out;
};