// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#ifndef LAB1_DENSE_MATMUL_DENSE_NN_H
#define LAB1_DENSE_MATMUL_DENSE_NN_H
#include "def.h"
#include "gemm.h"
#include "gemv.h"

namespace swiftware::hpp {
 //TODO add necessary includes

 // please do not change below
 DenseMatrix *dense_nn_gemm(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);

 DenseMatrix *dense_nn_gemv(DenseMatrix *InData, DenseMatrix *W1, DenseMatrix *W2, DenseMatrix *B1, DenseMatrix *B2, ScheduleParams Sp);


}
#endif //LAB1_DENSE_MATMUL_DENSE_NN_H
