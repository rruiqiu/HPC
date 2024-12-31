// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#ifndef LAB2_SPARSE_MATMUL_SPMM_H
#define LAB2_SPARSE_MATMUL_SPMM_H

// please do not change this file

#include "def.h"

namespace swiftware::hpp {

  // please do not change below
  /// \brief Sparse Matrix-matrix multiplication
  /// \param m Number of rows of A
  /// \param n Number of columns of B
  /// \param k Number of columns of A
  /// \param Ap Pointer to the start of the row pointer array
  /// \param Ai Pointer to the column index array
  /// \param Ax Pointer to the value array
  /// \param B Matrix B
  /// \param C Matrix C
  /// \param Sp Schedule parameters
  void spmmCSR(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp);
  void spmmCSROptimized(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp);
  void spmmCSROptimized_tilled(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp);
  /// \brief Sparse Matrix-matrix multiplication with skipping
  /// \param m Number of rows of A
  /// \param n Number of columns of B
  /// \param k Number of columns of A
  /// \param A Matrix A
  /// \param B Matrix B
  /// \param C Matrix C
  /// \param Sp Schedule parameters
  void spmmSkipping(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp);
  void spmmCSR_cache(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp);

}

#endif //LAB1_DENSE_MATMUL_SPMM_H
