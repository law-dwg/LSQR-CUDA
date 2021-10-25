#include "Kernels.cuh"
#include "MatrixCUDA.cuh"
#include <assert.h>
#include <vector>

VectorCUDA MatrixCUDA::operator*(VectorCUDA &v) { // Multiplication
  VectorCUDA out(this->h_rows, 1);
  unsigned blocksX = ((this->h_rows * this->h_columns) / (BLOCK_SIZE_X * BLOCK_SIZE_X)) + 1;
  dim3 grid(blocksX, 1, 1);
  dim3 block(BLOCK_SIZE_X * BLOCK_SIZE_X, 1, 1);
  // printf("grid(%d,%d,%d), block(%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
  spmvNaive<<<grid, block>>>(this->d_rows, this->d_columns, this->d_csrRowPtr, this->d_csrColInd, this->d_csrVal, v.getMat(), out.getMat());
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
  dim3 threads(BLOCK_SIZE_X * BLOCK_SIZE_X, 1);
  int blockX = ((this->h_nnz + threads.x - 1) / threads.x);
  dim3 blocks(blockX, 1);
  double *d_out, *d_max;
  double zero = 0.0;
  double h_max;
  double h_out;
  cudaErrCheck(cudaMalloc(&d_out, sizeof(double)));
  cudaErrCheck(cudaMalloc(&d_max, sizeof(double)));

  cudaErrCheck(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_max, &zero, sizeof(double), cudaMemcpyHostToDevice));
  maxVal<<<blocks, threads, threads.x * sizeof(double)>>>(this->d_csrVal, this->h_nnz, 1, d_max);
  // cudaErrCheck(cudaPeekAtLastError());
  cudaErrCheck(cudaDeviceSynchronize());
  dnrm2<<<blocks, threads, threads.x * sizeof(double)>>>(this->d_csrVal, this->h_nnz, 1, d_max, d_out);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost));
  assert(!(h_out != h_out));
  assert(h_out > 0);
  return (std::abs(h_max) * sqrt(h_out));
}