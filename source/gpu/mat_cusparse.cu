#include "mat_cusparse.cuh"
#include <assert.h>
Vector_GPU Matrix_cuSPARSE::operator*(Vector_GPU &v) {
  Vector_GPU out(this->h_rows, 1);
  cusparseDnVecDescr_t rhs_desc, out_desc;
  cusparseErrCheck(cusparseCreateDnVec(&rhs_desc, out.getRows(), v.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseCreateDnVec(&out_desc, out.getRows(), out.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseSpMV_bufferSize(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->d_matDescr_sparse, rhs_desc, &ZERO, out_desc,
                                           CUDA_R_64F, CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
  cudaErrCheck(cudaMalloc(&dBuffer, this->bufferSize));
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
  cudaErrCheck(cudaMalloc(&dBuffer, this->bufferSize));
  cusparseErrCheck(cusparseSpMV(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->d_matDescr_sparse, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                CUSPARSE_MV_ALG_DEFAULT, dBuffer));

  return out;
};