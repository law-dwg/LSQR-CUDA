#include "MatrixCUSPARSE.cuh"
#include <assert.h>
Vector_GPU MatrixCUSPARSE::operator*(Vector_GPU &v) {
  Vector_GPU out(this->h_rows, 1);
  cusparseDnVecDescr_t rhs_desc, out_desc;
  cusparseErrCheck(cusparseCreateDnVec(&rhs_desc, out.getRows(), v.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseCreateDnVec(&out_desc, out.getRows(), out.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseSpMV_bufferSize(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->spMatDescr, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                           CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
  // cudaErrCheck(cudaMalloc(&dBuffer, this->bufferSize));
  cusparseErrCheck(cusparseSpMV(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->spMatDescr, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                CUSPARSE_MV_ALG_DEFAULT, dBuffer));

  return out;
};
Vector_CUBLAS MatrixCUSPARSE::operator*(Vector_CUBLAS &v) {
  Vector_CUBLAS out(this->h_rows, 1);
  cusparseDnVecDescr_t rhs_desc, out_desc;
  cusparseErrCheck(cusparseCreateDnVec(&rhs_desc, out.getRows(), v.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseCreateDnVec(&out_desc, out.getRows(), out.d_mat, CUDA_R_64F));
  cusparseErrCheck(cusparseSpMV_bufferSize(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->spMatDescr, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                           CUSPARSE_MV_ALG_DEFAULT, &bufferSize));
  // cudaErrCheck(cudaMalloc(&dBuffer, this->bufferSize));
  cusparseErrCheck(cusparseSpMV(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, this->spMatDescr, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                CUSPARSE_MV_ALG_DEFAULT, dBuffer));

  return out;
};
MatrixCUSPARSE MatrixCUSPARSE::transpose() {
  MatrixCUSPARSE out(this->h_columns, this->h_rows, this->h_nnz);
  cudaErrCheck(cudaMemset(out.d_csrColInd, ZERO, out.h_nnz * sizeof(int)));
  cudaErrCheck(cudaMemset(out.d_csrVal, ZERO, out.h_nnz * sizeof(double)));
  cusparseErrCheck(cusparseCsr2cscEx2_bufferSize(spHandle, this->h_rows, this->h_columns, this->h_nnz, this->d_csrVal, this->d_csrRowPtr,
                                                 this->d_csrColInd, out.d_csrVal, out.d_csrRowPtr, out.d_csrColInd, CUDA_R_64F,
                                                 CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &out.bufferSize));
  cudaErrCheck(cudaMalloc(&out.dBuffer, out.bufferSize));
  cusparseErrCheck(cusparseCsr2cscEx2(spHandle, this->h_rows, this->h_columns, this->h_nnz, this->d_csrVal, this->d_csrRowPtr, this->d_csrColInd,
                                      out.d_csrVal, out.d_csrRowPtr, out.d_csrColInd, CUDA_R_64F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                                      CUSPARSE_CSR2CSC_ALG1, out.dBuffer));

  cusparseErrCheck(cusparseCreateCsr(&out.spMatDescr, out.h_rows, out.h_columns, out.h_nnz, out.d_csrRowPtr, out.d_csrColInd, out.d_csrVal,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  return out;
};
double MatrixCUSPARSE::Dnrm2() {
  double h_out;
  int incre = 1;
  cublasErrCheck(cublasDnrm2(handle, (int)this->h_nnz, this->d_csrVal, incre, &h_out));
  assert(!(h_out != h_out));
  return h_out;
};