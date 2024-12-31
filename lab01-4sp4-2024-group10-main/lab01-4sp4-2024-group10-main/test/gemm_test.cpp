// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab


#include <gtest/gtest.h>
#include "gemm.h"
#include "utils.h"

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
    TEST(GEMMTestSample, SmallTest) {
        int m = 2;
        int n = 2;
        int k = 2;
        // TODO replace below with DenseMatrix

        auto *A = new swiftware::hpp::DenseMatrix(m, k);
        auto *B = new swiftware::hpp::DenseMatrix(k, n);
        auto *C = new swiftware::hpp::DenseMatrix(m, n);

     
        A->set(0, 0, 1); A->set(0, 1, 2);
        A->set(1, 0, 3); A->set(1, 1, 4);

        B->set(0, 0, 1); B->set(0, 1, 2);
        B->set(1, 0, 3); B->set(1, 1, 4);

        C->set(0, 0, 0); C->set(0, 1, 0);
        C->set(1, 0, 0); C->set(1, 1, 0);

        //float A[4] = {1, 2, 3, 4};
        //float B[4] = {1, 2, 3, 4};
        //float C[4] = {0, 0, 0, 0};

        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(32, 32));

        auto *expected = new swiftware::hpp::DenseMatrix(m, n);
        expected->set(0, 0, 7); 
        expected->set(0, 1, 10);
        expected->set(1, 0, 15); 
        expected->set(1, 1, 22);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                EXPECT_EQ(C->get(i,j), expected->get(i,j));
            }
        }
    }

    TEST(gemmT1,TestGemm512) {
        int m = 512;
        int n = 512;
        int k = 512;
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
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), expected->data.data(), swiftware::hpp::ScheduleParams(-1, -1)); //use the first function to construct the expected array

        swiftware::hpp::gemmT1(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(1024, -1));

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


    TEST(gemmVec,TestGemmVec) {
        int m = 512;
        int n = 512;
        int k = 512;
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
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), expected->data.data(), swiftware::hpp::ScheduleParams(-1, -1)); //use the first function to construct the expected array

        swiftware::hpp::gemmT2(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(1024, 512));

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



        TEST(gemmT2,TestGemm512) {
        int m = 512;
        int n = 512;
        int k = 512;
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
        to_add = 1.0;
        for(int i=0;i<k*n;i++){
            B->data[i] = to_add;
            to_add++;
        }
        swiftware::hpp::gemm(m, n, k, A->data.data(), B->data.data(), expected->data.data(), swiftware::hpp::ScheduleParams(-1, -1)); //use the first function to construct the expected array

        swiftware::hpp::gemmVectorized(m, n, k, A->data.data(), B->data.data(), C->data.data(), swiftware::hpp::ScheduleParams(1024, -1));

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
