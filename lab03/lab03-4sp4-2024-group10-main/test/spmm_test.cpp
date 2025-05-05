// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <gtest/gtest.h>
#include "spmm.h"
#include "def.h"
#include "gemm.h"
#include "utils.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
namespace swiftware::hpp {
  void printRandomNumbers(const std::vector<int>& vec, const std::string& vectorName) {
    std::cout << vectorName << ": ";
    for (int num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
  }
  TEST(SPMMTest, TestAll) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    //each srPercentage represent the sparsity of the matrix, loop through to test the final results
    // std::vector<int> srPercentages = {1,2,3,4,5,10,15,18,20,30,50,60};

    std::vector<int> srPercentages = {1,5,10,30};//small for quick testing

    std::vector<int> m_values;
    std::vector<int> n_values;
    std::vector<int> k_values;
    auto generateRandomNumbers = [](std::vector<int>& vec) {
        for (int i = 0; i < 10; ++i) {
            int randomNum = std::rand() % 1049;  // Random number between 0 and 99
            vec.push_back(randomNum);
        }
    };
    //m, n, k, has an array of same length 10 with random number at each index for catching up corner cases
    generateRandomNumbers(m_values);
    generateRandomNumbers(n_values);
    generateRandomNumbers(k_values);

    printRandomNumbers(m_values,"m");
    printRandomNumbers(n_values,"n");
    printRandomNumbers(k_values,"k");
    for(int srPercentage:srPercentages){
      for(size_t i=0; i< m_values.size();i++){

        int m = m_values[i];
        int n = n_values[i];
        int k = k_values[i];
        std::cout << srPercentage <<"  Start\n";
        std::cout << "m:" << m;
        std::cout << "n:" << n;
        std::cout << "k:" << k;
        auto *A = new swiftware::hpp::DenseMatrix(m, k);
        auto *B = new swiftware::hpp::DenseMatrix(k, n);
        auto *C = new swiftware::hpp::DenseMatrix(m, n);
        auto *Test_1 = new swiftware::hpp::DenseMatrix(m, n);
        // auto *Test_2 = new swiftware::hpp::DenseMatrix(m, n);
        // auto *gemm_rest = new swiftware::hpp::DenseMatrix(m, n);
        for (int i = 0; i < m * k; ++i) {
          A->data[i] = (float)i+1;
        }
        for (int i = 0; i < k * n; ++i) {
          B->data[i] = (float)i+1;
        }

        float samplingRate = (float)srPercentage / 100.;

        // Sample Dense Matrix
        auto* sampledA = swiftware::hpp::samplingDense(A,samplingRate);

        // Convert A to CSR
        auto *ACSR = swiftware::hpp::dense2CSR(sampledA);

        swiftware::hpp::spmmCSREfficientSequential(m, n, k, ACSR->rowPtr.data(), ACSR->colIdx.data(), ACSR->data.data(),  B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(32, 32,-1,-1));

        swiftware::hpp::spmmCSREfficientParallel(m, n, k, ACSR->rowPtr.data(), ACSR->colIdx.data(), ACSR->data.data(),  B->data.data(), Test_1->data.data(), swiftware::hpp::ScheduleParams(32, -1,8,32));

        // swiftware::hpp::spmmSkipping(m,n,k,sampledA->data.data(),B->data.data(),Test_2->data.data(), swiftware::hpp::ScheduleParams(-1, -1));

        // printMatrix(sampledA,m,k);
        // printMatrix(B,m,k);
        // printMatrix(C,m,n);
        // printMatrix(Test_1,m,n);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ASSERT_EQ(C -> data[i*n + j], Test_1->data[i*n + j]);
            }
        }
        std::cout<<"\n"<< srPercentage <<"  Complete\n\n";
        delete A;
        delete B;
        delete C;
        delete Test_1;
        // delete Test_2;
        delete sampledA;
        // delete gemm_rest;
      }
    }

    
  }

}