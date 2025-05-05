// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#ifndef LAB2_SPMV_H
#define LAB2_SPMV_H
#include "def.h"

namespace swiftware::hpp {

  // please do not change below
  /// \brief Sparse Matrix-vector multiplication
  /// \param m Number of rows of A
  /// \param n Number of columns of A
  /// \param Ap Pointer to the start of the row pointer array
  /// \param Ai Pointer to the column index array
  /// \param Ax Pointer to the value array
  /// \param b Vector b
  /// \param c Vector c
  /// \param Sp Schedule parameters
  void spmvCSR(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp);
  void spmvCSRTiled(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp);
  void spmvCSRVectorized(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp);
  void spmvCSROptimized(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp);

  /// \brief Sparse Matrix-vector multiplication with skipping
  /// \param m Number of rows of A
  /// \param n Number of columns of A
  /// \param A Matrix A
  /// \param b Vector b
  /// \param c Vector c
  /// \param Sp Schedule parameters
  void spmvSkipping(int m, int n, const float *A, const float *b, float *c, ScheduleParams Sp);

}

#endif //LAB2_SPMV_H
