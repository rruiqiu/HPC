// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#ifndef LAB2_GEMV_H
#define LAB2_GEMV_H
// please do not change this file

#include "def.h"
namespace swiftware::hpp {
  // please do not change below
  /// \brief Matrix-vector multiplication
  /// \param m Number of rows of A
  /// \param n Number of columns of A
  /// \param A Matrix A
  /// \param x Vector x
  /// \param y Vector y
  void gemvVec(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp);
}
#endif //LAB2_GEMV_H
