// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <gtest/gtest.h>
#include "gemm.h"
#include "def.h"
#include "utils.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
extern float matmul_single_row_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k);
extern float matmul_mutiple_rows_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k, int tiling_size);
extern float matmul_mutiple_rowA_colB_Wrapper(float* h_A, float* h_B, float* h_C, int m, int n, int k, int tiling_size_row_A,int tiling_size_col_B);
namespace swiftware::hpp {
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
  void printRandomNumbers(const std::vector<int>& vec, const std::string& vectorName) {
    std::cout << vectorName << ": ";
    for (int num : vec) {
        std::cout << num << " ";
    }
    std::cout << std::endl;
  }


  // TEST(MMTest, OpenMP_RandomTest) {
  //   std::srand(static_cast<unsigned>(std::time(nullptr)));

  //   std::vector<int> m_values;
  //   std::vector<int> n_values;
  //   std::vector<int> k_values;
  //   auto generateRandomNumbers = [](std::vector<int>& vec) {
  //       for (int i = 0; i < 10; ++i) {
  //           int randomNum = std::rand() % 1024;  // Random number between 0 and 99
  //           vec.push_back(randomNum);
  //       }
  //   };
  //   //m, n, k, has an array of same length 10 with random number at each index for catching up corner cases
  //   generateRandomNumbers(m_values);
  //   generateRandomNumbers(n_values);
  //   generateRandomNumbers(k_values);

  //   printRandomNumbers(m_values,"m");
  //   printRandomNumbers(n_values,"n");
  //   printRandomNumbers(k_values,"k");

  //   for(size_t i=0; i< m_values.size();i++){

  //       int m = m_values[i];
  //       int n = n_values[i];
  //       int k = k_values[i];
  //       std::cout << "m: " << m;
  //       std::cout << " n: " << n;
  //       std::cout << " k: " << k <<std::endl;
  //       auto *A = new swiftware::hpp::DenseMatrix(m, k);
  //       auto *B = new swiftware::hpp::DenseMatrix(k, n);
  //       auto *C = new swiftware::hpp::DenseMatrix(m, n);
  //       auto *Test_1 = new swiftware::hpp::DenseMatrix(m, n);
  //       auto *Test_2 = new swiftware::hpp::DenseMatrix(m, n);
  //       // auto *gemm_rest = new swiftware::hpp::DenseMatrix(m, n);
  //       for (int i = 0; i < m * k; ++i) {
  //         A->data[i] = (float)i+1;
  //       }
  //       for (int i = 0; i < k * n; ++i) {
  //         B->data[i] = (float)i+1;
  //       }

  //       swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(),swiftware::hpp::ScheduleParams(-1, -1,-1, -1));

  //       swiftware::hpp::gemmEfficientParallel(m, n, k, A->data.data(), B->data.data(), Test_1->data.data(),swiftware::hpp::ScheduleParams(512, -1,8, 256));

  //       // swiftware::hpp::spmmSkipping(m,n,k,sampledA->data.data(),B->data.data(),Test_2->data.data(), swiftware::hpp::ScheduleParams(-1, -1));

  //       // printMatrix(sampledA,m,k);
  //       // printMatrix(B,m,k);
  //       // printMatrix(C,m,n);
  //       // printMatrix(Test_1,m,n);
  //       // printRandomNumbers(m_values,"m");
  //       // printRandomNumbers(n_values,"n");
  //       // printRandomNumbers(k_values,"k");
  //       for (int i = 0; i < m; ++i) {
  //           for (int j = 0; j < n; ++j) {
  //               ASSERT_NEAR(C -> data[i*n + j], Test_1->data[i*n + j],1e-5);
  //               // ASSERT_EQ(C -> data[i*n + j], Test_1->data[i*n + j]);
  //           }
  //       }
  //       std::cout << "pass" <<std::endl;
  //       delete A;
  //       delete B;
  //       delete C;
  //       delete Test_1;
  //       delete Test_2;
  //   }



  // }

// TEST(gemm_single_row_Test, SmallCase) {
//     // Small fixed test case for easy validation
//     int m =1024, n = 1024, k = 1024;

//     // Allocate matrices
//     auto *A = new swiftware::hpp::DenseMatrix(m, k);
//     auto *B = new swiftware::hpp::DenseMatrix(k, n);
//     auto *C = new swiftware::hpp::DenseMatrix(m, n);       // Reference result
//     auto *Test_1 = new swiftware::hpp::DenseMatrix(m, n);  // CUDA result

//     // Initialize A and B with known values
//     // A = [1, 2; 3, 4]
//     for (int i = 0; i < m * k; ++i) {
//         A->data[i] = static_cast<float>(i + 1);
//     }
//     // B = [5, 6; 7, 8]
//     for (int i = 0; i < k * n; ++i) {
//         B->data[i] = static_cast<float>(i + 5);
//     }

//     // Expected result C:
//     // C = A * B = [19, 22; 43, 50]

//     // Run the reference CPU GEMM implementation
//     swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(),
//                          swiftware::hpp::ScheduleParams(-1, -1, -1, -1));

//     // Run the CUDA implementation
//     matmul_single_row_Wrapper(A->data.data(), B->data.data(), Test_1->data.data(), m, n, k);

//     // Print the matrices for debugging
//     // std::cout << "Matrix A:\n";
//     // printMatrix(A, m, k);
//     // std::cout << "Matrix B:\n";
//     // printMatrix(B, k, n);
//     // std::cout << "Matrix C (Expected):\n";
//     // printMatrix(C, m, n);
//     // std::cout << "Matrix Test_1 (CUDA):\n";
//     // printMatrix(Test_1, m, n);

//     // Compare results
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < n; ++j) {
//              EXPECT_LT(abs((C->data[i * n + j]-Test_1->data[i * n + j])/Test_1->data[i * n + j]),1e-3)
//                 << "Mismatch at (i=" << i << ", j=" << j << "): "
//                 << "expected=" << C->data[i * n + j] << ", actual=" << Test_1->data[i * n + j];
//         }
//     }

//     // Clean up
//     delete A;
//     delete B;
//     delete C;
//     delete Test_1;
// }


  // TODO add more tests for GEMM OpenCL
  // TEST(gemm_single_row_Test, Cuda_1) {
  //   std::srand(static_cast<unsigned>(std::time(nullptr)));

  //   std::vector<int> m_values;
  //   std::vector<int> n_values;
  //   std::vector<int> k_values;
  //   auto generateRandomNumbers = [](std::vector<int>& vec) {
  //       for (int i = 0; i < 10; ++i) {
  //           int randomNum = std::rand() % 1024;  // Random number between 0 and 99
  //           vec.push_back(randomNum);
  //       }
  //   };
  //   //m, n, k, has an array of same length 10 with random number at each index for catching up corner cases
  //   generateRandomNumbers(m_values);
  //   generateRandomNumbers(n_values);
  //   generateRandomNumbers(k_values);

  //   printRandomNumbers(m_values,"m");
  //   printRandomNumbers(n_values,"n");
  //   printRandomNumbers(k_values,"k");

  //   for(size_t i=0; i< m_values.size();i++){

  //       int m = m_values[i];
  //       int n = n_values[i];
  //       int k = k_values[i];
  //       std::cout << "m: " << m;
  //       std::cout << " n: " << n;
  //       std::cout << " k: " << k <<std::endl;
  //       auto *A = new swiftware::hpp::DenseMatrix(m, k);
  //       auto *B = new swiftware::hpp::DenseMatrix(k, n);
  //       auto *C = new swiftware::hpp::DenseMatrix(m, n);
  //       auto *Test_1 = new swiftware::hpp::DenseMatrix(m, n);
  //       auto *Test_2 = new swiftware::hpp::DenseMatrix(m, n);
  //       // auto *gemm_rest = new swiftware::hpp::DenseMatrix(m, n);
  //       for (int i = 0; i < m * k; ++i) {
  //         A->data[i] = (float)i+1;
  //       }
  //       for (int i = 0; i < k * n; ++i) {
  //         B->data[i] = (float)i+1;
  //       }

  //       swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(),swiftware::hpp::ScheduleParams(-1, -1,-1, -1));

  //       matmul_single_row_Wrapper(A->data.data(), B->data.data(), Test_1->data.data(), m, n, k);


  //       // printMatrix(sampledA,m,k);
  //       // printMatrix(B,m,k);
  //       // printMatrix(C,m,n);
  //       // printMatrix(Test_1,m,n);
  //       // printRandomNumbers(m_values,"m");
  //       // printRandomNumbers(n_values,"n");
  //       // printRandomNumbers(k_values,"k");
  //       for (int i = 0; i < m; ++i) {
  //           for (int j = 0; j < n; ++j) {
  //             EXPECT_LT(abs((C->data[i * n + j]-Test_1->data[i * n + j])/Test_1->data[i * n + j]),1e-3)
  //                 << "Mismatch at (i=" << i << ", j=" << j << "): "
  //                 << "expected=" << C->data[i * n + j] << ", actual=" << Test_1->data[i * n + j];
  //           }
  //       }
  //       std::cout << "pass" <<std::endl;
  //       delete A;
  //       delete B;
  //       delete C;
  //       delete Test_1;
  //       delete Test_2;
  //   }
  // }


  // TEST(gemm_mutiple_rows_Test, Cuda_2) {
  //   std::srand(static_cast<unsigned>(std::time(nullptr)));

  //   std::vector<int> m_values;
  //   std::vector<int> n_values;
  //   std::vector<int> k_values;
  //   auto generateRandomNumbers = [](std::vector<int>& vec) {
  //       for (int i = 0; i < 10; ++i) {
  //           int randomNum = std::rand() % 1024;  // Random number between 0 and 99
  //           vec.push_back(randomNum);
  //       }
  //   };
  //   //m, n, k, has an array of same length 10 with random number at each index for catching up corner cases
  //   generateRandomNumbers(m_values);
  //   generateRandomNumbers(n_values);
  //   generateRandomNumbers(k_values);

  //   printRandomNumbers(m_values,"m");
  //   printRandomNumbers(n_values,"n");
  //   printRandomNumbers(k_values,"k");

  //   for(size_t i=0; i< m_values.size();i++){

  //       int m = m_values[i];
  //       int n = n_values[i];
  //       int k = k_values[i];
  //       std::cout << "m: " << m;
  //       std::cout << " n: " << n;
  //       std::cout << " k: " << k <<std::endl;
  //       auto *A = new swiftware::hpp::DenseMatrix(m, k);
  //       auto *B = new swiftware::hpp::DenseMatrix(k, n);
  //       auto *C = new swiftware::hpp::DenseMatrix(m, n);
  //       auto *Test_1 = new swiftware::hpp::DenseMatrix(m, n);
  //       auto *Test_2 = new swiftware::hpp::DenseMatrix(m, n);
  //       // auto *gemm_rest = new swiftware::hpp::DenseMatrix(m, n);
  //       for (int i = 0; i < m * k; ++i) {
  //         A->data[i] = (float)i+1;
  //       }
  //       for (int i = 0; i < k * n; ++i) {
  //         B->data[i] = (float)i+1;
  //       }

  //       swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(),swiftware::hpp::ScheduleParams(-1, -1,-1, -1));

  //       matmul_mutiple_rows_Wrapper(A->data.data(), B->data.data(), Test_1->data.data(), m, n, k,4);


  //       // printMatrix(sampledA,m,k);
  //       // printMatrix(B,m,k);
  //       // printMatrix(C,m,n);
  //       // printMatrix(Test_1,m,n);
  //       // printRandomNumbers(m_values,"m");
  //       // printRandomNumbers(n_values,"n");
  //       // printRandomNumbers(k_values,"k");
  //       for (int i = 0; i < m; ++i) {
  //           for (int j = 0; j < n; ++j) {
  //             EXPECT_LT(abs((C->data[i * n + j]-Test_1->data[i * n + j])/Test_1->data[i * n + j]),1e-3)
  //                 << "Mismatch at (i=" << i << ", j=" << j << "): "
  //                 << "expected=" << C->data[i * n + j] << ", actual=" << Test_1->data[i * n + j];
  //           }
  //       }
  //       std::cout << "pass" <<std::endl;
  //       delete A;
  //       delete B;
  //       delete C;
  //       delete Test_1;
  //       delete Test_2;
  //   }
  // }

  TEST(gemm_tile_Row_col_Test, Cuda_3) {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    std::vector<int> m_values;
    std::vector<int> n_values;
    std::vector<int> k_values;
    auto generateRandomNumbers = [](std::vector<int>& vec) {
        for (int i = 0; i < 10; ++i) {
            int randomNum = std::rand() % 1024;  // Random number between 0 and 99
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

    for(size_t i=0; i< m_values.size();i++){

        int m = m_values[i];
        int n = n_values[i];
        int k = k_values[i];
        std::cout << "m: " << m;
        std::cout << " n: " << n;
        std::cout << " k: " << k <<std::endl;
        auto *A = new swiftware::hpp::DenseMatrix(m, k);
        auto *B = new swiftware::hpp::DenseMatrix(k, n);
        auto *C = new swiftware::hpp::DenseMatrix(m, n);
        auto *Test_1 = new swiftware::hpp::DenseMatrix(m, n);
        auto *Test_2 = new swiftware::hpp::DenseMatrix(m, n);
        // auto *gemm_rest = new swiftware::hpp::DenseMatrix(m, n);
        for (int i = 0; i < m * k; ++i) {
          A->data[i] = (float)i+1;
        }
        for (int i = 0; i < k * n; ++i) {
          B->data[i] = (float)i+1;
        }

        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(),swiftware::hpp::ScheduleParams(-1, -1,-1, -1));

        matmul_mutiple_rowA_colB_Wrapper(A->data.data(), B->data.data(), Test_1->data.data(), m, n, k,4,4);


        // printMatrix(sampledA,m,k);
        // printMatrix(B,m,k);
        // printMatrix(C,m,n);
        // printMatrix(Test_1,m,n);
        // printRandomNumbers(m_values,"m");
        // printRandomNumbers(n_values,"n");
        // printRandomNumbers(k_values,"k");
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              ASSERT_LT(abs((C->data[i * n + j]-Test_1->data[i * n + j])/Test_1->data[i * n + j]),1e-3)
                  << "Mismatch at (i=" << i << ", j=" << j << "): "
                  << "expected=" << C->data[i * n + j] << ", actual=" << Test_1->data[i * n + j];
            }
        }
        std::cout << "pass" <<std::endl;
        delete A;
        delete B;
        delete C;
        delete Test_1;
        delete Test_2;
    }
  }

  TEST(gemm_mutiple_rows_Test, SmallCase) {
      // Small fixed test case for easy validation
      int m =4096, n = 4096, k = 4096;

      // Allocate matrices
      auto *A = new swiftware::hpp::DenseMatrix(m, k);
      auto *B = new swiftware::hpp::DenseMatrix(k, n);
      auto *C = new swiftware::hpp::DenseMatrix(m, n);       // Reference result
      auto *Test_1 = new swiftware::hpp::DenseMatrix(m, n);  // CUDA result

      // Initialize A and B with known values
      // A = [1, 2; 3, 4]
      for (int i = 0; i < m * k; ++i) {
          A->data[i] = static_cast<float>(i + 1);
      }
      // B = [5, 6; 7, 8]
      for (int i = 0; i < k * n; ++i) {
          B->data[i] = static_cast<float>(i + 5);
      }

      // Expected result C:
      // C = A * B = [19, 22; 43, 50]

      // Run the reference CPU GEMM implementation
      swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(),
                          swiftware::hpp::ScheduleParams(-1, -1, -1, -1));

      // Run the CUDA implementation
      matmul_mutiple_rowA_colB_Wrapper(A->data.data(), B->data.data(), Test_1->data.data(), m, n, k,1,4096);

      // // Print the matrices for debugging
      // std::cout << "Matrix A:\n";
      // printMatrix(A, m, k);
      // std::cout << "Matrix B:\n";
      // printMatrix(B, k, n);
      // std::cout << "Matrix C (Expected):\n";
      // printMatrix(C, m, n);
      // std::cout << "Matrix Test_1 (CUDA):\n";
      // printMatrix(Test_1, m, n);

      // Compare results
      for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
              EXPECT_LT(abs((C->data[i * n + j]-Test_1->data[i * n + j])/Test_1->data[i * n + j]),1e-3)
                  << "Mismatch at (i=" << i << ", j=" << j << "): "
                  << "expected=" << C->data[i * n + j] << ", actual=" << Test_1->data[i * n + j];
          }
      }

      // Clean up
      delete A;
      delete B;
      delete C;
      delete Test_1;
  }


}