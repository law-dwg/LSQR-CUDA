#include "mat_cusparse.cuh"
#include <assert.h>
Vector_GPU Matrix_cuSPARSE::operator*(Vector_GPU &v) {
  Vector_GPU out(this->h_rows, 1);
  cusparseDnVecDescr_t rhs_desc, out_desc;
  cusparseErrCheck(cusparseCreateDnVec(&rhs_desc, out.getRows(), v.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseCreateDnVec(&out_desc, out.getRows(), out.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseSpMV_bufferSize(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->d_matDescr_sparse, rhs_desc, &ZERO, out_desc,
                                           CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
  // cudaErrCheck(cudaMalloc(&dBuffer, this->bufferSize));
  cusparseErrCheck(cusparseSpMV(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->d_matDescr_sparse, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                CUSPARSE_MV_ALG_DEFAULT, dBuffer));

  return out;
};
Vector_CUBLAS Matrix_cuSPARSE::operator*(Vector_CUBLAS &v) {
  Vector_CUBLAS out(this->h_rows, 1);
  cusparseDnVecDescr_t rhs_desc, out_desc;
  cusparseErrCheck(cusparseCreateDnVec(&rhs_desc, out.getRows(), v.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseCreateDnVec(&out_desc, out.getRows(), out.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseSpMV_bufferSize(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->d_matDescr_sparse, rhs_desc, &ZERO, out_desc,
                                           CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
  // cudaErrCheck(cudaMalloc(&dBuffer, this->bufferSize));
  cusparseErrCheck(cusparseSpMV(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->d_matDescr_sparse, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                CUSPARSE_MV_ALG_DEFAULT, dBuffer));

  return out;
};
Matrix_cuSPARSE Matrix_cuSPARSE::transpose() {
  Matrix_cuSPARSE out(this->h_columns, this->h_rows);
  out.h_nnz = this->h_nnz;
  printf("%d\n", out.h_nnz);
  cudaErrCheck(cudaMalloc((void **)&out.d_csr_columns, out.h_nnz * sizeof(int)));
  cudaErrCheck(cudaMalloc((void **)&out.d_csr_values, out.h_nnz * sizeof(double)));
  cudaErrCheck(cudaMemset(out.d_csr_columns, ZERO, out.h_nnz * sizeof(int)));
  cudaErrCheck(cudaMemset(out.d_csr_values, ZERO, out.h_nnz * sizeof(double)));
  cusparseErrCheck(cusparseCsr2cscEx2_bufferSize(spHandle, this->h_rows, this->h_columns, this->h_nnz, this->d_csr_values, this->d_csr_offsets,
                                                 this->d_csr_columns, out.d_csr_values, out.d_csr_offsets, out.d_csr_columns, CUDA_R_64F,
                                                 CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &out.bufferSize));
  cudaErrCheck(cudaMalloc(&out.dBuffer, out.bufferSize));
  cusparseErrCheck(cusparseCsr2cscEx2(spHandle, this->h_rows, this->h_columns, this->h_nnz, this->d_csr_values, this->d_csr_offsets,
                                      this->d_csr_columns, out.d_csr_values, out.d_csr_offsets, out.d_csr_columns, CUDA_R_64F,
                                      CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, out.dBuffer));

  cusparseErrCheck(cusparseCreateCsr(&out.d_matDescr_sparse, out.h_rows, out.h_columns, out.h_nnz, out.d_csr_offsets, out.d_csr_columns,
                                     out.d_csr_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

  // cusparseErrCheck(cusparseCsrSetPointers(d_matDescr_sparse, d_csr_offsets, d_csr_columns, d_csr_values));

  return out;
};
double Matrix_cuSPARSE::Dnrm2() {
  double h_out;
  int incre = 1;
  cublasErrCheck(cublasDnrm2(handle, (int)this->h_nnz, this->d_csr_values, incre, &h_out));
  assert(!(h_out != h_out));
  return h_out;
};