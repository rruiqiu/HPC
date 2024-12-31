// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <gtest/gtest.h>
#include "spmv.h"
#include "gemv.h"
#include "utils.h"

namespace swiftware::hpp {

  void printMatrix(const swiftware::hpp::DenseMatrix* matrix, int rows, int cols) {
    std::cout << "start" <<std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << matrix->data[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "end"<<std::endl;
  }
  

  TEST(SPMVTest, SmallTest) {
    int m = 2;
    int n = 2;
    int k = 1;
    // TODO generate random sparse matrices
    float A[4] = {1, 0, 3, 4};
    float B[2] = {0, 3};
    float C[2] = {0, 0};
    swiftware::hpp::spmvSkipping(m, n, A, B, C, swiftware::hpp::ScheduleParams(32, 32));
    float expected[2] = {0,12};
    for (int j = 0; j < m; ++j) {
        EXPECT_EQ(C[j], expected[j]);
      
    }
  }
  /*

  TEST(SPMVTest, BigTest) {
    int m = 5;
    int n = 5;
    int k = 1;
    // TODO generate random sparse matrices
    int srPercentage = 50;

    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *b = new swiftware::hpp::DenseMatrix(n, 1);
    auto *c = new swiftware::hpp::DenseMatrix(m, 1);
    auto *c_vec = new swiftware::hpp::DenseMatrix(m, 1);
    for(int i=0;i<m*n;i++){
      A->data[i] = 1.0 + i;
    }

    for(int i=0;i<n;i++){
      b->data[i] =  1.0 + i;
    }
    b->data[n-1] = 0;

    float samplingRate = (float)srPercentage / 100.; //sampling rate contrls how much of matrix A remains non=zero
    //so if smpaleing rate ie 1 (100%), no elments are set to zero and matrix A remains fully dense
    auto* sampledA = swiftware::hpp::samplingDense(A,samplingRate);
    swiftware::hpp::spmvSkipping(m, n, sampledA->data.data(), b->data.data(), c->data.data(), swiftware::hpp::ScheduleParams(32, 32));
    swiftware::hpp::gemvVec(m, n, sampledA->data.data(), b->data.data(), c_vec->data.data(), swiftware::hpp::ScheduleParams(32, 32));
    for (int j = 0; j < m; ++j) {
        EXPECT_EQ(c->data[j], c_vec->data[j]);
        
    }
    delete sampledA;
    delete b;
    delete A;
    delete c;
    delete c_vec;
    //printMatrix(sampledA, m,n);
    //printMatrix(b, m,1);
    //printMatrix(c_vec, m,1);
    //printMatrix(c, m,1);
  }
  

  TEST(SPMVTest, spmmCSROptimized) {
    int m = 80;
    int n = 128; //the nnz elmenets play a role in the eror im getting
    int k = 1;
    // TODO generate random sparse matrices
    int srPercentage = 50;

    auto *A = new swiftware::hpp::DenseMatrix(m, n);
    auto *b = new swiftware::hpp::DenseMatrix(n, 1);
    auto *c = new swiftware::hpp::DenseMatrix(m, 1);
    auto *c_vec = new swiftware::hpp::DenseMatrix(m, 1);
    auto *tiles = new swiftware::hpp::ScheduleParams(1,1);
    for(int i=0;i<m*n;i++){
      A->data[i] = 1.0 + i;
    }

    for(int i=0;i<n;i++){
      b->data[i] =  1.0 + i;
    }

    float samplingRate = (float)srPercentage / 100.; //sampling rate contrls how much of matrix A remains non=zero
    //so if smpaleing rate is 1 (100%), no elments are set to zero and matrix A remains fully dense
    auto* sampledA = swiftware::hpp::samplingDense(A,samplingRate);
    // Convert A to CSR
    auto *ACSR = swiftware::hpp::dense2CSR(sampledA);
    //printMatrix(sampledA, m,n);
    //printMatrix(b, m,1);
    swiftware::hpp::spmvCSROptimized(m, n,  ACSR->rowPtr.data(), ACSR->colIdx.data(), ACSR->data.data(), b->data.data(), c->data.data(), *tiles);
    swiftware::hpp::gemvVec(m, n, sampledA->data.data(), b->data.data(), c_vec->data.data(), swiftware::hpp::ScheduleParams(32, 32));
    //swiftware::hpp::spmvCSR(m, n,  ACSR->rowPtr.data(), ACSR->colIdx.data(), ACSR->data.data(), b->data.data(), c_vec->data.data(), *tiles);
    for (int j = 0; j < m; ++j) {
        EXPECT_FLOAT_EQ(c->data[j], c_vec->data[j]);
        
    }
    // printMatrix(sampledA, m,n);
    //printMatrix(b, m,1);
    //printMatrix(c_vec, m,1);
    //printMatrix(c, m,1);
    delete sampledA;
    delete b;
    delete A;
    delete c;
    delete c_vec;
  }
  */

}
