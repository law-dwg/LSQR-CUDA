#pragma once
#include "matrixCUDA.cuh"
#include "vectorCUBLAS.cuh"

class MatrixCUSPARSE : public MatrixGPU {
protected:
  cusparseSpMatDescr_t spMatDescr = NULL;
  void denseToCUSPARSE(double *m) {
    // temp gpu allocation
    cusparseDnMatDescr_t dnMatDescr;
    double *d_mat;
    cudaErrCheck(cudaMalloc((void **)&d_mat, h_rows * h_columns * sizeof(double)));
    cudaErrCheck(cudaMemcpy(d_mat, m, h_rows * h_columns * sizeof(double), cudaMemcpyHostToDevice));

    // cusparse dense and sparse matricies
    cusparseErrCheck(cusparseCreateDnMat(&dnMatDescr, h_rows, h_columns, h_columns, d_mat, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    cusparseErrCheck(cusparseCreateCsr(&spMatDescr, h_rows, h_columns, 0, d_csrRowPtr, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    cusparseErrCheck(cusparseDenseToSparse_bufferSize(spHandle, dnMatDescr, spMatDescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
    cudaErrCheck(cudaMalloc(&dBuffer, bufferSize));
    cusparseErrCheck(cusparseDenseToSparse_analysis(spHandle, dnMatDescr, spMatDescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

    // find nnz
    int64_t rows_tmp, cols_tmp, h_nnz_temp;
    cusparseErrCheck(cusparseSpMatGetSize(spMatDescr, &rows_tmp, &cols_tmp, &h_nnz_temp));
    h_nnz = (unsigned)h_nnz_temp;
    cudaErrCheck(cudaMalloc((void **)&d_csrColInd, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&d_csrVal, h_nnz * sizeof(double)));

    // set CSR
    cusparseErrCheck(cusparseCsrSetPointers(spMatDescr, d_csrRowPtr, d_csrColInd, d_csrVal));
    cusparseErrCheck(cusparseDenseToSparse_convert(spHandle, dnMatDescr, spMatDescr, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, dBuffer));

    // free gpu temp allocations
    cusparseDestroyDnMat(dnMatDescr);
    cudaFree(d_mat);
  }

public:
  /** Constructors */
  MatrixCUSPARSE() : MatrixGPU(0,0,0u){};                                          // Default Constr.
  MatrixCUSPARSE(unsigned r, unsigned c) : MatrixGPU(r, c, 0u){};                // Constr. #1
  MatrixCUSPARSE(unsigned r, unsigned c, unsigned n) : MatrixGPU(r, c, n){}; // Constr. #2
  MatrixCUSPARSE(unsigned r, unsigned c, double *m) : MatrixGPU(r,c,0u) {          // Constr. #3
    // free resources that will be reallocated
    cudaErrCheck(cudaFree(d_csrVal));
    cudaErrCheck(cudaFree(d_csrColInd));
  
    // dense to sparse
    denseToCUSPARSE(m);
  };
  MatrixCUSPARSE(const MatrixCUSPARSE &m) : MatrixGPU(m) { // Copy Constr.
    cusparseErrCheck(cusparseCreateCsr(&spMatDescr, h_rows, h_columns, h_nnz, d_csrRowPtr, d_csrColInd, d_csrVal, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  };
  MatrixCUSPARSE(MatrixCUSPARSE &&m) noexcept : MatrixGPU(std::move(m)) { // Move Constr.
    // free old resources
    cusparseErrCheck(cusparseDestroySpMat(m.spMatDescr));
    // placeholder
    cusparseErrCheck(cusparseCreateCsr(&m.spMatDescr, m.h_rows, m.h_columns, m.h_nnz, m.d_csrRowPtr, m.d_csrColInd, m.d_csrVal, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  };

  /** Destructor */
  ~MatrixCUSPARSE() { cusparseErrCheck(cusparseDestroySpMat(spMatDescr)); };

  /** Assignments */
  MatrixCUSPARSE &operator=(const MatrixCUSPARSE &m) { // Copy Assignment
    // free + memory allocation
    if (h_rows != m.h_rows) {
      h_rows = m.h_rows;
      cudaErrCheck(cudaMemcpy(d_rows, m.d_rows, sizeof(unsigned), cudaMemcpyDeviceToDevice));
      cudaErrCheck(cudaFree(d_csrRowPtr));
      cudaErrCheck(cudaMalloc((void **)&d_csrRowPtr, sizeof(int) * (m.h_rows + 1)));
    }
    if (h_nnz != m.h_nnz) {
      h_nnz = m.h_nnz;
      cudaErrCheck(cudaFree(d_csrVal));
      cudaErrCheck(cudaFree(d_csrColInd));
      cudaErrCheck(cudaMalloc((void **)&d_csrVal, sizeof(double) * h_nnz));
      cudaErrCheck(cudaMalloc((void **)&d_csrColInd, sizeof(int) * h_nnz));
    }
    if (h_columns != m.h_columns) {
      h_columns = m.h_columns;
      cudaErrCheck(cudaMemcpy(d_columns, m.d_columns, sizeof(unsigned), cudaMemcpyDeviceToDevice));
    }
    // copy to device
    cudaErrCheck(cudaMemcpy(d_csrVal, m.d_csrVal, h_nnz * sizeof(double), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csrColInd, m.d_csrColInd, h_nnz * sizeof(int), cudaMemcpyDeviceToDevice));
    cudaErrCheck(cudaMemcpy(d_csrRowPtr, m.d_csrRowPtr, (h_rows + 1) * sizeof(int), cudaMemcpyDeviceToDevice));
    // cusparse
    cusparseErrCheck(cusparseCreateCsr(&spMatDescr, h_rows, h_columns, h_nnz, d_csrRowPtr, d_csrColInd, d_csrVal, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    return *this;
  };
  MatrixCUSPARSE &operator=(MatrixCUSPARSE &&m) noexcept { // Move Assignment
    // call copy assignment
    *this = m;
    m.h_rows = ZERO;
    m.h_nnz = ZERO;
    m.h_columns = ZERO;
    // freeing memory handled by destructor, potential err. blocked via rows = cols = 0
    return *this;
  };

  /** Operator overloads */
  VectorCUDA operator*(VectorCUDA &v); // Multiplication
  VectorCUBLAS operator*(VectorCUBLAS &v);

  /** Member Functions */
  MatrixCUSPARSE transpose();
  double Dnrm2();
  template <typename T> friend void SpMV(MatrixCUSPARSE &M, T &v, T &out);
};