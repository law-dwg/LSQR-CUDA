#include "MatrixCSR.cuh"
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
__global__ void spmvTiled(unsigned *rows, unsigned *col, int *d_nnz, int *rowPtr, int *colIdx, double *val, double *rhs, double *out) {
  __shared__ double temp[TILE_DIM_X * TILE_DIM_X];
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = thread_id & (32 - 1); // thread index within the warp
  int warp_id = thread_id / 32;    // global warp index
  // one warp per row
  int row = warp_id;
  printf("gid=%d, row = %d, wid=%d\n", thread_id, row, lane);
  if (row < *rows) {
    int row_start = rowPtr[row];
    int row_end = rowPtr[row + 1];
    temp[threadIdx.x] = 0.0;
    double sum = 0.0;
    for (int i = row_start + lane; i < row_end; i += 32) {
      sum += val[i] * rhs[colIdx[i]];
      printf("sum=%f += val[%d] * rhs[colIdx[%d]] = %f\n", sum, i, i, temp[threadIdx.x]);
    }
    temp[threadIdx.x] = sum;
    if (lane < 16)
      temp[threadIdx.x] += temp[threadIdx.x + 16];
    if (lane < 8)
      temp[threadIdx.x] += temp[threadIdx.x + 8];
    if (lane < 4)
      temp[threadIdx.x] += temp[threadIdx.x + 4];
    if (lane < 2)
      temp[threadIdx.x] += temp[threadIdx.x + 2];
    if (lane < 1)
      temp[threadIdx.x] += temp[threadIdx.x + 1];

    if (lane == 0) {
      out[row] += temp[threadIdx.x];
      printf("wid=0 called, out[%d]=%f\n", row, out[row]);
    }
  }
}

Vector_GPU MatrixCSR::operator*(Vector_GPU &v) { // Multiplication
  Vector_GPU out(this->h_rows, 1);
  unsigned int blocksX = ((this->h_rows * this->h_columns) / (TILE_DIM_X * TILE_DIM_X)) + 1;
  dim3 grid(blocksX, 1, 1);
  dim3 block(TILE_DIM_X * TILE_DIM_X, 1, 1);
  printf("grid(%d,%d,%d), block(%d,%d,%d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
  spmvTiled<<<grid, block>>>(this->d_rows, this->d_columns, this->d_nnz, this->d_csrRowPtr, this->d_csrColInd, this->d_csrVal, v.d_mat, out.d_mat);
  return out;
}

MatrixCSR MatrixCSR::transpose() {
  MatrixCSR out(this->h_columns, this->h_rows, this->h_nnz);
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