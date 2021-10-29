#include "kernels.cuh"
#include "vectorCUDA.cuh"
#include <assert.h>
#include <cmath>
#include <stdio.h>

/** Operator overloads */
VectorCUDA VectorCUDA::operator*(double h_i) {
  VectorCUDA out(this->h_rows, this->h_columns);
  unsigned blocksX = (this->h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (this->h_columns / BLOCK_SIZE_X) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_X, 1);
  scale<<<grid, block>>>(this->d_mat, h_i, out.d_mat, this->d_rows, this->d_columns, false);
  return out;
}

VectorCUDA VectorCUDA::operator*(VectorCUDA &v) {
  VectorCUDA out(this->h_rows, v.h_columns);
  bool tiled = true;
  dim3 threads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  unsigned blocksY = (out.h_rows / threads.x) + 1;
  unsigned blocksX = (out.h_columns / threads.y) + 1;
  dim3 blocks(blocksX, blocksY, 1);

  if (h_columns == v.h_rows) {
    if (tiled) {
      multiplyTiled<<<blocks, threads, ((BLOCK_SIZE_X * (BLOCK_SIZE_Y + 1) * sizeof(double)) * 2)>>>(this->d_mat, this->d_rows, this->d_columns,
                                                                                                     v.d_mat, v.d_rows, v.d_columns, out.d_mat);
    } else {
      multiplyNaive<<<blocks, threads>>>(this->d_mat, this->d_rows, this->d_columns, v.d_mat, v.d_rows, v.d_columns, out.d_mat);
    }
  } else {
    printf("Cannot perform multiplication, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }

  return out;
}

VectorCUDA VectorCUDA::operator-(const VectorCUDA &v) {
  VectorCUDA out(this->h_rows, this->h_columns);
  unsigned blocksX = (this->h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (this->h_columns / BLOCK_SIZE_X) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_X, 1);
  if (this->h_rows == v.h_rows && this->h_columns == v.h_columns) {
    add<<<grid, block>>>(this->d_mat, v.d_mat, this->d_rows, this->d_columns, out.d_mat, false);
  } else {
    printf("Cannot perform subtraction, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  return out;
}

void VectorCUDA::operator=(VectorCPU &v) { // Copy assignment VectorCPU -> this VectorCUDA
  cudaErrCheck(cudaFree(d_mat));
  h_rows = v.getRows();
  h_columns = v.getColumns();
  cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.getRows() * v.getColumns()));
  cudaErrCheck(cudaMemcpy(d_rows, &h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_columns, &h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_mat, v.getMat(), sizeof(double) * h_rows * h_columns, cudaMemcpyHostToDevice));
}

VectorCUDA VectorCUDA::operator+(const VectorCUDA &v) {
  VectorCUDA out(this->h_rows, this->h_columns);
  unsigned blocksX = (this->h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (this->h_columns / BLOCK_SIZE_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  if (this->h_rows == v.h_rows && this->h_columns == v.h_columns) {
    add<<<grid, block>>>(this->d_mat, v.d_mat, this->d_rows, this->d_columns, out.d_mat, true);
  } else {
    printf("Cannot perform addition, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  return out;
}

/** Member Functions */
void VectorCUDA::printMat() {
  unsigned blocksX = (this->h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (this->h_columns / BLOCK_SIZE_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  print<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns);
}

VectorCPU VectorCUDA::matDeviceToHost() {
  double *out = new double[this->h_columns * this->h_rows]; // heap to prevent a stack overflow
  unsigned rows, cols;
  cudaErrCheck(cudaMemcpy(out, this->d_mat, sizeof(double) * this->h_columns * this->h_rows, cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&rows, this->d_rows, sizeof(unsigned), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&cols, this->d_columns, sizeof(unsigned), cudaMemcpyDeviceToHost));
  if (rows != this->h_rows || cols != this->h_columns) {
    printf("Cannot perform move to host, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
  VectorCPU v_cpu(this->h_rows, this->h_columns, out);
  return v_cpu;
}

VectorCUDA VectorCUDA::transpose() {
  VectorCUDA out(this->h_columns, this->h_rows);
  dim3 numOfThreadsInBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  unsigned blocksX = (out.h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (out.h_columns / BLOCK_SIZE_Y) + 1;
  dim3 numOfBlocksInGrid(blocksX, blocksY, 1);
  transposeTiled<<<numOfBlocksInGrid, numOfThreadsInBlock, (BLOCK_SIZE_X) * (BLOCK_SIZE_Y + 1) * sizeof(double)>>>(this->d_mat, out.d_mat,
                                                                                                                   this->d_rows, this->d_columns);
  return out;
};

double VectorCUDA::Dnrm2() {
  dim3 threads(BLOCK_SIZE_X * BLOCK_SIZE_X, 1);
  int blockX = ((this->h_rows * this->h_columns + (threads.x) - 1) / (threads.x));
  dim3 blocks(blockX, 1);
  double *d_out, *d_max;
  double h_max, h_out;
  cudaErrCheck(cudaMalloc(&d_out, sizeof(double)));
  cudaErrCheck(cudaMalloc(&d_max, sizeof(double)));
  cudaErrCheck(cudaMemcpy(d_out, &ZERO, sizeof(double), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_max, &ZERO, sizeof(double), cudaMemcpyHostToDevice));
  maxVal<<<blocks, threads, threads.x * sizeof(double)>>>(this->d_mat, this->h_rows, this->h_columns, d_max);
  cudaErrCheck(cudaDeviceSynchronize());
  dnrm2<<<blocks, threads, threads.x * sizeof(double)>>>(this->d_mat, this->h_rows, this->h_columns, d_max, d_out);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost));
  // sanity checks
  assert(!(h_out != h_out));
  assert(h_out > 0);
  return (std::abs(h_max) * std::sqrt(h_out));
};