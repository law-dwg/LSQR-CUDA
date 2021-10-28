#include "matrixCUSPARSE.cuh"
#include <assert.h>

template <typename T> void SpMV(MatrixCUSPARSE &M, T &v, T &out) {
  if (M.getColumns() == v.getRows()) {
    cusparseDnVecDescr_t rhs_desc, out_desc;
    cusparseErrCheck(cusparseCreateDnVec(&rhs_desc, v.getRows(), v.getMat(), CUDA_R_64F));
    cusparseErrCheck(cusparseCreateDnVec(&out_desc, out.getRows(), out.getMat(), CUDA_R_64F));
    cusparseErrCheck(cusparseSpMV_bufferSize(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, M.spMatDescr, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                             CUSPARSE_MV_ALG_DEFAULT, &M.bufferSize));
    // cudaErrCheck(cudaMalloc(&dBuffer, this->bufferSize));
    cusparseErrCheck(cusparseSpMV(spHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, &ONE, M.spMatDescr, rhs_desc, &ZERO, out_desc, CUDA_R_64F,
                                  CUSPARSE_MV_ALG_DEFAULT, M.dBuffer));
  } else {
    printf("Cannot perform multiplication, dimension mismatch %s(%d)\n", __FILE__, __LINE__);
    exit(1);
  }
};

VectorCUDA MatrixCUSPARSE::operator*(VectorCUDA &v) {
  VectorCUDA out(this->h_rows, 1);
  SpMV(*this, v, out);
  return out;
};

VectorCUBLAS MatrixCUSPARSE::operator*(VectorCUBLAS &v) {
  VectorCUBLAS out(this->h_rows, 1);
  SpMV(*this, v, out);
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
  // sanity checks
  assert(!(h_out != h_out));
  assert(0 < h_out);
  return h_out;
};