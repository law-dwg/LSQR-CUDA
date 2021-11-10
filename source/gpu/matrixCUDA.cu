#include "kernels.cuh"
#include "matrixCUDA.cuh"
#include <assert.h>
#include <vector>

VectorCUDA MatrixCUDA::operator*(VectorCUDA &v) { // Multiplication
  VectorCUDA out(this->h_rows, 1);
  int kern = 2;
  dim3 block(BLOCK_SIZE_X * BLOCK_SIZE_X, 1, 1);

  if (this->h_columns == v.getRows()) {
    unsigned totalNumThreads = this->getRows() * WARP_SIZE; // one warp per row
    if (kern == 2) {
      dim3 grid((totalNumThreads / block.x) + 1, 1, 1);
      spmvCSRVector<<<grid, block>>>(this->d_rows, this->d_csrRowPtr, this->d_csrColInd, this->d_csrVal, v.getMat(), out.getMat());
    } else if (kern == 1) {
      dim3 grid((totalNumThreads / block.x) + 1, 1, 1);
      spmvCSRVectorShared<<<grid, block, block.x * sizeof(double)>>>(this->d_rows, this->d_csrRowPtr, this->d_csrColInd, this->d_csrVal, v.getMat(),
                                                                     out.getMat());
    } else {
      dim3 grid(((this->h_rows * this->h_columns) / block.x) + 1, 1, 1); // one thread per entry
      spmvNaive<<<grid, block>>>(this->d_rows, this->d_csrRowPtr, this->d_csrColInd, this->d_csrVal, v.getMat(), out.getMat());
    }
  } else {
    printf("Cannot perform multiplication, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  // cudaLastErrCheck();
  return out;
};

MatrixCUDA MatrixCUDA::transpose() {
  MatrixCUDA out(this->h_columns, this->h_rows, this->h_nnz);
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
  dim3 block(BLOCK_SIZE_X * BLOCK_SIZE_X, 1);
  int gridX = ((this->h_nnz + block.x - 1) / block.x);
  dim3 grid(gridX, 1);
  double *d_out, *d_max;
  double zero = 0.0;
  double h_max;
  double h_out;

  cudaErrCheck(cudaMalloc(&d_out, sizeof(double)));
  cudaErrCheck(cudaMalloc(&d_max, sizeof(double)));

  cudaErrCheck(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_max, &zero, sizeof(double), cudaMemcpyHostToDevice));
  maxVal<<<grid, block, block.x * sizeof(double)>>>(this->d_csrVal, this->h_nnz, 1, d_max); // find max value for better precision
  cudaErrCheck(cudaDeviceSynchronize());                                                    // necessary synchronize between two kernels
  dnrm2<<<grid, block, block.x * sizeof(double)>>>(this->d_csrVal, this->h_nnz, 1, d_max, d_out);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost));
  // cudaLastErrCheck();

  // sanity checks
  assert(!(h_out != h_out));
  assert(h_out > 0);
  return (std::abs(h_max) * sqrt(h_out));
};