#include "matCsr_gpu.cuh"
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
  int warp_id = thread_id / 32;    // global warp index
  int lane = thread_id & (32 - 1); // thread index within the warp
  // one warp per row
  int row = warp_id;
  printf("gid=%d, row = %d, wid=%d\n", thread_id, row, lane);
  if (row < *rows) {
    int row_start = rowPtr[row];
    int row_end = rowPtr[row + 1];
    temp[threadIdx.x] = 0.0;
    for (int i = row_start + lane; i < row_end; i += 32) {
      temp[threadIdx.x] += val[i] * rhs[colIdx[i]];
      printf("temp[%d] += val[%d] * rhs[colIdx[%d]] = %f\n", threadIdx.x, i, i, temp[threadIdx.x]);
    }
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

Vector_GPU MatrixCsrGPU::operator*(Vector_GPU &v) { // Multiplication
  Vector_GPU out(this->h_rows, 1);
  unsigned int blocksX = (this->h_rows / TILE_DIM_X) + 1;
  unsigned int blocksY = (this->h_columns / TILE_DIM_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(TILE_DIM_X, TILE_DIM_Y, 1);
  // printf("threadsinblock(%d x %d)=%d, blocksingrid(%d, %d)=%d\n", blocks, 1, threads, blocks, 1, blocks);
  spmvTiled<<<grid, block>>>(this->d_rows, this->d_columns, this->d_nnz, this->d_csrRowPtr, this->d_csrColInd, this->d_csrVal, v.d_mat, out.d_mat);
  // cudaMemcpy(&h_out, d_out, sizeof(double) * h_rows, cudaMemcpyDeviceToHost);

  // double out[h_rows], h_vals[h_nnz], rhs_mat[v.h_rows * v.h_columns];
  // int h_rowPtr[h_rows + 1], h_colIdx[h_nnz];
  // cudaMemcpy(&rhs_mat, v.d_mat, sizeof(double) * v.h_rows * v.h_columns, cudaMemcpyDeviceToHost);
  // cudaMemcpy(&h_vals, this->d_csrVal, sizeof(double) * h_nnz, cudaMemcpyDeviceToHost);
  // cudaMemcpy(&h_rowPtr, this->d_csrRowPtr, sizeof(int) * h_rows + 1, cudaMemcpyDeviceToHost);
  // cudaMemcpy(&h_colIdx, this->d_csrColInd, sizeof(int) * h_nnz, cudaMemcpyDeviceToHost);
  // double y;
  // for (int i = 0; i < h_rows + 1; ++i) {
  //  printf("h_rowPtr[%d]=%f\n", i, h_rowPtr[i]);
  //}
  // for (int i = 0; i < h_rows; ++i) {
  //
  //  for (int j = h_rowPtr[i]; j < h_rowPtr[i + 1]; ++j) {
  //    printf("h_vals[%d]=%f\n", j, h_vals[j]);
  //    y = h_vals[j] * rhs_mat[h_colIdx[j]];
  //    out[i] += y;
  //  }
  //}
  // for (int i = 0; i < h_rows; ++i) {
  //   printf("h_out[%d]=%f\n", i, h_out[i]);
  // }
  return out;
}