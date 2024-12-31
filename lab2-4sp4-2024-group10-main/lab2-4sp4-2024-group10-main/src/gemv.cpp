// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include <numeric>
#include <iostream>
#include <limits>
#include <iomanip>
#include "gemv.h"
#include <immintrin.h> // For AVX intrinsics
namespace swiftware::hpp {
 void gemvVec(int m, int n, const float *A, const float *x, float *y, ScheduleParams Sp) {
        int tileSize = Sp.TileSize1;
        __m256 sum = _mm256_setzero_ps();
        float edge_to_sum = 0;
        int row_bnd = n - n%32;
        if(n < 32){
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    y[i] += A[i * n + j] * x[j];
                }
            }
            return;
        }

        // tileSize = 1024;
        for (int i = 0; i < m; i++) {


            //load next 4 rows, vectorize the row, block 32

            for (int j = 0; j < row_bnd; j += tileSize) {
                for(int jj = j;jj<j +tileSize && (jj < row_bnd);jj+=32){
                    
                    auto a_val = _mm256_loadu_ps(&A[i * n + jj]);
                    auto a_val_2 = _mm256_loadu_ps(&A[i * n + jj+8]);
                    auto a_val_3 = _mm256_loadu_ps(&A[i * n + jj+16]);
                    auto a_val_4 = _mm256_loadu_ps(&A[i * n + jj+24]);

                    auto x_val = _mm256_loadu_ps(&x[jj]); 
                    auto x_val_2 = _mm256_loadu_ps(&x[jj+8]);
                    auto x_val_3 = _mm256_loadu_ps(&x[jj+16]);
                    auto x_val_4 = _mm256_loadu_ps(&x[jj+24]);

                    auto prod   = _mm256_mul_ps(a_val,x_val);
                    auto prod_2 = _mm256_mul_ps(a_val_2, x_val_2);  
                    auto prod_3 = _mm256_mul_ps(a_val_3, x_val_3);  
                    auto prod_4 = _mm256_mul_ps(a_val_4, x_val_4);  

                    //
                    sum = _mm256_add_ps(prod,sum);
                    sum = _mm256_add_ps(prod_2,sum);
                    sum = _mm256_add_ps(prod_3,sum);
                    sum = _mm256_add_ps(prod_4,sum);

                    // cout << sum;
                }


            }
            for(int jj = row_bnd;jj<n;jj++){
                edge_to_sum += A[i * n + jj] * x[jj];
            }
            float sum_2[8];
            _mm256_storeu_ps(sum_2,sum);
            y[i] += std::accumulate(sum_2,sum_2+8,edge_to_sum);
            //proceed the next row
            sum = _mm256_setzero_ps();
            edge_to_sum = 0;

        }
    }


}