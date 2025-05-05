// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#ifndef LAB2_SPAARSE_MATMUL_SPARSE_NN_H
#define LAB2_SPAARSE_MATMUL_SPARSE_NN_H
#include "def.h"
#include "spmm.h"


namespace swiftware::hpp {
  //TODO add necessary includes

  // please do not change below
  DenseMatrix *sparseNNSpmm(DenseMatrix *InData, CSR *W1, CSR *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);

  DenseMatrix *denseNNGemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);

  DenseMatrix *denseNNGemmSkipping(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);


}
#endif //LAB1_DENSE_MATMUL_SPARSE_NN_H
