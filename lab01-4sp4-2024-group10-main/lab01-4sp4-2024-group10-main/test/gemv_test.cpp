// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include "gemv.h"
#include <iostream>
#include <gtest/gtest.h>
namespace swiftware::hpp {
 // TODO
  void printMatrix(const swiftware::hpp::DenseMatrix* matrix, int rows, int cols) {
    std::cout << "start" <<std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix->data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "end"<<std::endl;
  }

  TEST(GEMVTest, TestAll) {
    //to Ta, large m n could cause float point rounding issues
    int m = 128;
    int n = 128;
    // TODO replace below with DenseMatrix
    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *x = new swiftware::hpp::DenseMatrix(n, 1);
    auto *y = new swiftware::hpp::DenseMatrix(m, 1);

    auto *test = new swiftware::hpp::DenseMatrix(m, 1);
    auto *test_vec = new swiftware::hpp::DenseMatrix(m, 1);
    //initialize matrix
    float to_add = 1.0;
    for(int i=0;i<m*n;i++){
      A->data[i] = to_add;
      to_add++;
    }
    to_add = 1.0;
    for(int i=0;i<n;i++){
      x->data[i] = to_add;
      to_add++;
    }

    // swiftware::hpp::gemv(m, n, A->data.data(), x->data.data(), y->data.data(), swiftware::hpp::ScheduleParams(-1, -1));

    //test T1
    swiftware::hpp::gemvT1(m, n, A->data.data(), x->data.data(), test->data.data(), swiftware::hpp::ScheduleParams(1024, -1));

    swiftware::hpp::gemvVec(m, n, A->data.data(), x->data.data(), test_vec->data.data(), swiftware::hpp::ScheduleParams(2048, -1));

    // printMatrix(A,m,n);
    // printMatrix(x,n,1);
    // printMatrix(y,m,1);
    // printMatrix(test_vec,m,1);

    for (int i = 0; i < m; ++i) {
      // ASSERT_EQ(test->data[i], y->data[i]);
      std::cout<<i<<std::endl;
      ASSERT_FLOAT_EQ(test_vec->data[i], test->data[i]);

    }
    delete A;
    delete x;
    delete y;
    delete test;
    delete test_vec;

 }
}