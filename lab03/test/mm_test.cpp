// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <gtest/gtest.h>
#include "gemm.h"

namespace swiftware::hpp {
  TEST(MMTest, SmallTest) {
    int m = 2;
    int n = 2;
    int k = 2;
    // TODO generate random sparse matrices
    float A[4] = {1, 2, 3, 4};
    float B[4] = {1, 2, 3, 4};
    float C[4] = {0, 0, 0, 0};
    swiftware::hpp::gemmEfficientParallel(m, n, k, A, B, C, swiftware::hpp::ScheduleParams(32, 32, 8, 1));
    float expected[4] = {7, 10, 15, 22};
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        EXPECT_EQ(C[i * n + j], expected[i * n + j]);
      }
    }
  }

  TEST(MMTest,TestGemmLarge) {
        int m = 80;
        int n = 1000;
        int k = 10;
        // TODO replace below with DenseMatrix

        auto *A = new swiftware::hpp::DenseMatrix(m, k);
        auto *B = new swiftware::hpp::DenseMatrix(k, n);
        auto *C = new swiftware::hpp::DenseMatrix(m, n);
        auto *expected = new swiftware::hpp::DenseMatrix(m, n);

        float to_add = 1.0;
        for(int i=0;i<m*k;i++){
            A->data[i] = to_add;
            to_add++;
        }
        to_add = 2.0;
        for(int i=0;i<k*n;i++){
            B->data[i] = to_add;
            to_add++;
        }
        swiftware::hpp::gemmEfficientSequential(m, n, k, A->data.data(), B->data.data(), expected->data.data(), swiftware::hpp::ScheduleParams(-1, -1, 8 ,1)); //use the first function to construct the expected array

        swiftware::hpp::gemmEfficientParallel(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(32, 32, 8, 1));

        // printMatrix(expected,m,n);
        // printMatrix(C,m,n);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ASSERT_EQ(C -> data[i*n + j], expected->data[i*n + j]);
                // ASSERT_EQ(T2 -> data[i*n + j], expected->data[i*n + j]);
            }
        }
        delete A;
        delete B;
        delete C;
        delete expected;
  }

}