#include "Kernels.cuh"
#include "VectorCUDA.cuh"
#include <assert.h>
#include <iostream>
#include <math.h>

/** Operator overloads */

VectorCUDA VectorCUDA::operator*(double h_i) {
  // printf("scale %f\n", h_i);
  VectorCUDA out(this->h_rows, this->h_columns);
  unsigned blocksX = (this->h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (this->h_columns / BLOCK_SIZE_X) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_X, 1);

  scale<<<grid, block>>>(this->d_mat, h_i, out.d_mat, this->d_rows, this->d_columns, false);
  // cudaDeviceSynchronize();
  printf("SCALE CALLED\n");
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
      multiplyTiled<<<blocks, threads, (BLOCK_SIZE_X * (BLOCK_SIZE_Y + 1) * sizeof(double)) * 2>>>(this->d_mat, this->d_rows, this->d_columns,
                                                                                                   v.d_mat, v.d_rows, v.d_columns, out.d_mat);
    } else {
      multiplyNaive<<<blocks, threads>>>(this->d_mat, this->d_rows, this->d_columns, v.d_mat, v.d_rows, v.d_columns, out.d_mat);
    }
  } else {
    printf("MATRICIES CANNOT BE MULTIPLED, INVALID SIZES");
  }

  return out;
}

VectorCUDA VectorCUDA::operator-(const VectorCUDA &v) {
  // printf("SUBTRACT CALLED\n");
  VectorCUDA out(this->h_rows, this->h_columns);
  unsigned blocksX = (this->h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (this->h_columns / BLOCK_SIZE_X) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_X, 1);
  if (this->h_rows == v.h_rows && this->h_columns == v.h_columns) {
    add<<<grid, block>>>(this->d_mat, v.d_mat, this->d_rows, this->d_columns, out.d_mat, false);
  } else {
    printf("SUBTRACT: ARRAYS ARE NOT THE SAME SIZE, canot perform operation %d!=%d\n", this->h_rows, v.h_rows);
  }
  return out;
}

void VectorCUDA::operator=(Vector_CPU &v) { // Copy assignment Vector_CPU -> this VectorCUDA
  printf("ASSIGNMENT #2 called\n");
  cudaErrCheck(cudaFree(d_mat));
  h_rows = v.getRows();
  h_columns = v.getColumns();
  cudaErrCheck(cudaMalloc((void **)&d_mat, sizeof(double) * v.rows * v.columns));
  cudaErrCheck(cudaMemcpy(d_rows, &h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_columns, &h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_mat, &v.mat[0], sizeof(double) * h_rows * h_columns, cudaMemcpyHostToDevice));
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
    printf("ADDITION: ARRAYS ARE NOT THE SAME SIZE, canot perform operation %d!=%d\n", this->h_rows, v.h_rows);
  }
  return out;
}

/** Member Functions */
void VectorCUDA::printmat() {
  unsigned blocksX = (this->h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (this->h_columns / BLOCK_SIZE_Y) + 1;
  dim3 grid(blocksX, blocksY, 1);
  dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  print<<<grid, block>>>(this->d_mat, this->d_rows, this->d_columns);
}

Vector_CPU VectorCUDA::matDeviceToHost() {
  double *out = new double[this->h_columns * this->h_rows]; // heap to prevent a stack overflow
  unsigned rows;
  unsigned cols;
  cudaErrCheck(cudaMemcpy(out, this->d_mat, sizeof(double) * this->h_columns * this->h_rows, cudaMemcpyDeviceToHost));
  // cudaMemcpy(out, this, size, cudaMemcpyDeviceToHost);
  cudaErrCheck(cudaMemcpy(&rows, this->d_rows, sizeof(unsigned), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&cols, this->d_columns, sizeof(unsigned), cudaMemcpyDeviceToHost));
  // std::cout << "d_rows=" << rows << "=h_rows=" << this->h_rows << std::endl;
  // std::cout << "d_columns=" << cols << "=h_columns=" << this->h_columns << std::endl;
  if (rows != this->h_rows || cols != this->h_columns) {
    printf("INCONSISTENT ROWS AND COLS BETWEEN HOST AND DEVICE\n");
  }
  Vector_CPU v_cpu(this->h_rows, this->h_columns, out);
  return v_cpu;
}

VectorCUDA VectorCUDA::transpose() {
  VectorCUDA out(this->h_columns, this->h_rows);

  dim3 numOfThreadsInBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1);
  unsigned blocksX = (out.h_rows / BLOCK_SIZE_X) + 1;
  unsigned blocksY = (out.h_columns / BLOCK_SIZE_Y) + 1;
  dim3 numOfBlocksInGrid(blocksX, blocksY, 1);
  printf("threadsinblock(%d x %d)=%d, blocksingrid(%d, %d)=%d\n", numOfThreadsInBlock.x, numOfThreadsInBlock.y,
         numOfThreadsInBlock.x * numOfThreadsInBlock.y, numOfBlocksInGrid.x, numOfBlocksInGrid.y, numOfBlocksInGrid.x * numOfBlocksInGrid.y);
  transposeTiled<<<numOfBlocksInGrid, numOfThreadsInBlock, (BLOCK_SIZE_X) * (BLOCK_SIZE_Y + 1) * sizeof(double)>>>(this->d_mat, out.d_mat,
                                                                                                                   this->d_rows, this->d_columns);
  // cudaDeviceSynchronize();
  return out;
};

double VectorCUDA::Dnrm2() {
  dim3 threads(BLOCK_SIZE_X * BLOCK_SIZE_X, 1);
  int blockX = ((this->h_rows * this->h_columns + (threads.x) - 1) / (threads.x));
  dim3 blocks(blockX, 1);
  double *d_out, *d_max;
  double zero = 0.0;
  double h_max;
  double h_out;
  cudaErrCheck(cudaMalloc(&d_out, sizeof(double)));
  cudaErrCheck(cudaMalloc(&d_max, sizeof(double)));
  cudaErrCheck(cudaMemcpy(d_out, &zero, sizeof(double), cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_max, &zero, sizeof(double), cudaMemcpyHostToDevice));

  // printf("dnrm2 threads(%d x %d)=%d, blocks(%d, %d)=%d\n", threads.x, threads.y, threads.x * threads.y, blocks.x, blocks.y, blocks.x * blocks.y);
  // unsigned s_mem = sizeof(double) * BLOCK_SIZE_X;

  maxVal<<<blocks, threads, threads.x * sizeof(double)>>>(this->d_mat, this->h_rows, this->h_columns, d_max);
  // cudaErrCheck(cudaPeekAtLastError());
  cudaErrCheck(cudaDeviceSynchronize());
  // unsigned s_mem = sizeof(double) * BLOCK_SIZE_X;
  dnrm2<<<blocks, threads, threads.x * sizeof(double)>>>(this->d_mat, this->h_rows, this->h_columns, d_max, d_out);
  cudaErrCheck(cudaDeviceSynchronize());
  cudaErrCheck(cudaMemcpy(&h_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));
  cudaErrCheck(cudaMemcpy(&h_max, d_max, sizeof(double), cudaMemcpyDeviceToHost));
  assert(!(h_out != h_out));
  assert(h_out > 0);
  return (std::abs(h_max) * sqrt(h_out));
};