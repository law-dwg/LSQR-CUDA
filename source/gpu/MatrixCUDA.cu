#include "MatrixCUDA.cuh"
#include "Kernels.cuh"
#include <assert.h>
#include <vector>
#define TILE_DIM_X 16
#define TILE_DIM_Y 16

__global__ void spmvNaive(unsigned *rows, unsigned *col, int *rowPtr, int *colIdx, double *val, double *rhs, double *out) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int bid;

  if (gid < *rows) {
    out[gid] = double(0.0);
    for (bid = rowPtr[gid]; bid < rowPtr[gid + 1]; ++bid) {
      out[gid] += val[bid] * rhs[colIdx[bid]];
    }
  }
}

VectorCUDA MatrixCUDA::operator*(VectorCUDA &v) { // Multiplication
  VectorCUDA out(this->h_rows, 1);
  unsigned blocksX = ((this->h_rows * this->h_columns) / (TILE_DIM_X * TILE_DIM_X)) + 1;
  dim3 grid(blocksX, 1, 1);
  dim3 block(TILE_DIM_X * TILE_DIM_X, 1, 1);
  // printf("grid(%d,%d,%d), block(%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
  spmvNaive<<<grid, block>>>(this->d_rows, this->d_columns, this->d_csrRowPtr, this->d_csrColInd, this->d_csrVal, v.d_mat, out.d_mat);
  return out;
}

MatrixCUDA MatrixCUDA::transpose() {
  MatrixCUDA out(this->h_columns, this->h_rows, this->h_nnz);
  cudaErrCheck(cudaMemset(out.d_csrColInd, ZERO, out.h_nnz * sizeof(int)));
  cudaErrCheck(cudaMemset(out.d_csrVal, ZERO, out.h_nnz * sizeof(double)));
  cusparseErrCheck(cusparseCsr2cscEx2_bufferSize(spHandle, this->h_rows, this->h_columns, this->h_nnz, this->d_csrVal, this->d_csrRowPtr,
                                                 this->d_csrColInd, out.d_csrVal, out.d_csrRowPtr, out.d_csrColInd, CUDA_R_64F,
                                                 CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &out.bufferSize));
  cudaErrCheck(cudaMalloc(&out.dBuffer, out.bufferSize));
  cusparseErrCheck(cusparseCsr2cscEx2(spHandle, this->h_rows, this->h_columns, this->h_nnz, this->d_csrVal, this->d_csrRowPtr, this->d_csrColInd,
                                      out.d_csrVal, out.d_csrRowPtr, out.d_csrColInd, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1, out.dBuffer));
  return out;
};

double MatrixCUDA::Dnrm2() {
  dim3 threads(TILE_DIM_X, 1);
  int blockX = ((this->h_nnz + TILE_DIM_X - 1) / TILE_DIM_X);
  dim3 blocks(blockX, 1);
  double *d_out, *d_max;
  double zero = 0.0;
  double h_max;
  double h_out;
  cudaErrCheck(cudaMalloc(&d_out, sizeof(double)));
  cudaErrCheck(cudaMalloc(&d_max, sizeof(double)));

  cudaErrCheck(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_max, &zero, sizeof(double), cudaMemcpyHostToDevice));
  maxVal<<<blocks, threads, 16 * sizeof(double)>>>(this->d_csrVal, this->h_nnz, 1, d_max);
  // cudaErrCheck(cudaPeekAtLastError());
  cudaErrCheck(cudaDeviceSynchronize());
  // unsigned s_mem = sizeof(double) * TILE_DIM_X;
  dnrm2<<<blocks, threads, 16 * sizeof(double)>>>(this->d_csrVal, this->h_nnz, 1, d_max, d_out);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost));
  assert(!(h_out != h_out));
  assert(h_out > 0);
  return (std::abs(h_max) * sqrt(h_out));
}