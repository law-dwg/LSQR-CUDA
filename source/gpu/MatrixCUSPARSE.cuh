#pragma once
#include "MatrixCUDA.cuh"
#include "VectorCUBLAS.cuh"

class MatrixCUSPARSE : public MatrixGPU {
protected:
  cusparseSpMatDescr_t spMatDescr = NULL;

public:
  /** Constructors */
  MatrixCUSPARSE(unsigned r, unsigned c) : MatrixGPU(r, c){};                                                                            // Constr. #1
  MatrixCUSPARSE(unsigned r, unsigned c, int n) : MatrixGPU(r, c, n){};                                                                  // Constr. #2
  MatrixCUSPARSE(unsigned r, unsigned c, int n, double *values, int *colInd, int *rowPtr) : MatrixGPU(r, c, n, values, colInd, rowPtr) { // Constr. #3
    // cusparse
    cusparseErrCheck(cusparseCreateCsr(&spMatDescr, h_rows, h_columns, h_nnz, d_csrRowPtr, d_csrColInd, d_csrVal, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  };
  MatrixCUSPARSE(unsigned r, unsigned c, double *m) : MatrixGPU(r, c) { // Constr. (entry point)
    // convert dense to sparse
    cusparseDnMatDescr_t dnMatDescr;
    double *d_mat;
    cudaErrCheck(cudaMalloc((void **)&d_mat, h_rows * h_columns * sizeof(double)));
    cudaErrCheck(cudaMemcpy(d_mat, m, r * c * sizeof(double), cudaMemcpyHostToDevice));

    // cusparse dense and sparse matricies
    cusparseErrCheck(cusparseCreateDnMat(&dnMatDescr, r, c, c, d_mat, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    cusparseErrCheck(cusparseCreateCsr(&spMatDescr, r, c, 0, d_csrRowPtr, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
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
    cusparseDestroyDnMat(dnMatDescr);
    cudaFree(d_mat);
  };
  MatrixCUSPARSE() : MatrixGPU(){};                                                                                                // Default Constr.
  MatrixCUSPARSE(const MatrixCUSPARSE &m) : MatrixGPU(m.h_rows, m.h_columns, m.h_nnz, m.d_csrVal, m.d_csrColInd, m.d_csrRowPtr){}; // Copy Constr.
  MatrixCUSPARSE(MatrixCUSPARSE &&m) noexcept : MatrixGPU(std::move(m)) {                                                          // Move Constr.
    // free old resources
    cusparseErrCheck(cusparseDestroySpMat(m.spMatDescr));
    // placeholder
    cusparseErrCheck(cusparseCreateCsr(&m.spMatDescr, m.h_rows, m.h_columns, m.h_nnz, m.d_csrRowPtr, m.d_csrColInd, m.d_csrVal, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  };

  /** Destructor */
  ~MatrixCUSPARSE() { cusparseErrCheck(cusparseDestroySpMat(spMatDescr)); };

  /** Assignments */
  MatrixCUSPARSE &operator=(const MatrixCUSPARSE &m) { // Copy assignment operator
    // free + memory allocation
    if (h_rows != m.h_rows) {
      h_rows = m.h_rows;
      cudaErrCheck(cudaMemcpy(d_rows, m.d_rows, sizeof(unsigned), cudaMemcpyDeviceToDevice));
      cudaErrCheck(cudaFree(d_csrRowPtr));
      cudaErrCheck(cudaMalloc((void **)&d_csrRowPtr, sizeof(int) * (m.h_rows + 1)));
    }
    if (h_nnz != m.h_nnz) {
      h_nnz = m.h_nnz;
      cudaErrCheck(cudaMemcpy(d_nnz, m.d_nnz, sizeof(unsigned), cudaMemcpyDeviceToDevice));
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
  MatrixCUSPARSE &operator=(MatrixCUSPARSE &&m) noexcept { // Move assignment operator
    printf("MatrixCUSPARSE Move Assignment called\n");
    // call copy assignment
    *this = m;
    // free old resources
    cudaErrCheck(cudaFree(m.d_csrVal));
    cudaErrCheck(cudaFree(m.d_csrRowPtr));
    cudaErrCheck(cudaFree(m.d_csrColInd));
    cusparseErrCheck(cusparseDestroySpMat(m.spMatDescr));
    m.h_rows = ZERO;
    m.h_nnz = ZERO;
    m.h_columns = ZERO;
    cudaErrCheck(cudaMalloc((void **)&m.d_csrVal, h_nnz * sizeof(double)));
    cudaErrCheck(cudaMalloc((void **)&m.d_csrColInd, h_nnz * sizeof(int)));
    cudaErrCheck(cudaMalloc((void **)&m.d_csrRowPtr, (h_rows + 1) * sizeof(int)));
    cudaErrCheck(cudaMemcpy(m.d_rows, &m.h_rows, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(m.d_columns, &m.h_columns, sizeof(unsigned), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemset(m.d_csrVal, ZERO, m.h_nnz * sizeof(double)));
    cudaErrCheck(cudaMemset(m.d_csrColInd, ZERO, m.h_nnz * sizeof(int)));
    cudaErrCheck(cudaMemset(m.d_csrRowPtr, ZERO, (m.h_rows + 1) * sizeof(int)));
    cusparseErrCheck(cusparseCreateCsr(&m.spMatDescr, m.h_rows, m.h_columns, m.h_nnz, m.d_csrRowPtr, m.d_csrColInd, m.d_csrVal, CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
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