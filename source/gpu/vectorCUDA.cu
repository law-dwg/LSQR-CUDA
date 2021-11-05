#include "kernels.cuh"
#include "vectorCUDA.cuh"
#include <assert.h>
#include <cmath>
#include <stdio.h>

/** Operator overloads */
VectorCUDA VectorCUDA::operator*(double h_i) {
  VectorCUDA out(this->h_rows, this->h_columns);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_X, 1);
  unsigned gridX = (this->h_rows / block.x) + 1;
  unsigned gridY = (this->h_columns / block.x) + 1;
  dim3 grid(gridX, gridY, 1);
  scale<<<grid, block>>>(this->d_mat, h_i, out.d_mat, this->d_rows, this->d_columns, false);
  return out;
}

VectorCUDA VectorCUDA::operator*(VectorCUDA &v) {
  VectorCUDA out(this->h_rows, v.h_columns);
  bool tiled = true;
  dim3 block(BLOCK_SIZE_X / 2, BLOCK_SIZE_Y / 2, 1); // halved to keep shared memory below 1024
  unsigned gridY = (out.h_rows / block.x) + 1;
  unsigned gridX = (out.h_columns / block.y) + 1;
  dim3 grid(gridX, gridY, 1);

  if (h_columns == v.h_rows) {
    if (tiled) {
      multiplyTiled<<<grid, block, (block.x * block.y * sizeof(double) * 2)>>>(this->d_mat, this->d_rows, this->d_columns, v.d_mat, v.d_rows,
                                                                               v.d_columns, out.d_mat);
    } else {
      multiplyNaive<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns, v.d_mat, v.d_rows, v.d_columns, out.d_mat);
    }
  } else {
    printf("Cannot perform multiplication, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }

  return out;
}

VectorCUDA VectorCUDA::operator-(const VectorCUDA &v) {
  VectorCUDA out(this->h_rows, this->h_columns);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_X, 1);
  unsigned blocksX = (this->h_rows / block.x) + 1;
  unsigned blocksY = (this->h_columns / block.y) + 1;
  dim3 grid(blocksX, blocksY, 1);
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
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  unsigned blocksX = (this->h_rows / block.x) + 1;
  unsigned blocksY = (this->h_columns / block.y) + 1;
  dim3 grid(blocksX, blocksY, 1);
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
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  unsigned gridX = (this->h_rows / block.x) + 1;
  unsigned gridY = (this->h_columns / block.y) + 1;
  dim3 grid(gridX, gridY, 1);
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
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  unsigned gridX = (out.h_rows / block.x) + 1;
  unsigned gridY = (out.h_columns / block.y) + 1;
  dim3 grid(gridX, gridY, 1);
  transposeTiled<<<grid, block, (block.x) * (block.y + 1) * sizeof(double)>>>(this->d_mat, out.d_mat, this->d_rows, this->d_columns);
  return out;
};

double VectorCUDA::Dnrm2() {
  dim3 block(BLOCK_SIZE_X * BLOCK_SIZE_X, 1);
  int gridX = ((this->h_rows * this->h_columns + (block.x) - 1) / (block.x));
  dim3 grid(gridX, 1);
  double *d_out, *d_max;
  double h_max, h_out;
  cudaErrCheck(cudaMalloc(&d_out, sizeof(double)));
  cudaErrCheck(cudaMalloc(&d_max, sizeof(double)));
  cudaErrCheck(cudaMemcpy(d_out, &ZERO, sizeof(double), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_max, &ZERO, sizeof(double), cudaMemcpyHostToDevice));
  maxVal<<<grid, block, block.x * sizeof(double)>>>(this->d_mat, this->h_rows, this->h_columns, d_max);
  cudaErrCheck(cudaDeviceSynchronize());
  dnrm2<<<grid, block, block.x * sizeof(double)>>>(this->d_mat, this->h_rows, this->h_columns, d_max, d_out);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost));
  // sanity checks
  assert(!(h_out != h_out));
  assert(h_out > 0);
  return (std::abs(h_max) * std::sqrt(h_out));
};