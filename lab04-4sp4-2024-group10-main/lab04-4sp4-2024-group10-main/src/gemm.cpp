// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab



#include "gemm.h"
#include <immintrin.h>
#include <algorithm>
namespace swiftware::hpp {
  void gemm(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
      for (int i = 0; i < m; ++i) { //row of A
          for (int l = 0; l < k; ++l) { //shared A and B
              for (int j = 0; j < n; ++j) { // col of B
                  C[i * n + j] += A[i * k + l] * B[l * n + j];
              }
          }
      }
  }

  void gemmEfficientParallel(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
    // TODO: add your parallel GEMM here
    int tileSize = Sp.TileSize1;
    int numThreads = Sp.NumThreads;
    int chunkSize = Sp.ChunkSize;

    //divide the matrix into a boundary can divide by row/4, column/8
    auto row_bnd = m - m % 4;   
    auto col_bnd = n - n % 8;   
    
    for (int i = 0; i < m; i += tileSize) {  // row A
            for (int l = 0; l < k; l += tileSize) {  // shared col A with row B
                #pragma omp parallel for num_threads(numThreads) schedule(static, chunkSize)
                for (int j = 0; j < n; j += tileSize) {  // col B
                    // Compute each tile's product
                    for (int ii = i; ii < i + tileSize && (ii+4) <= m; ii+=4) {
                        for (int ll = l; ll < l + tileSize && ll < k; ll++) {
                            auto a_val = _mm256_set1_ps(A[ii * k + ll]);
                            auto a_val_2 = _mm256_set1_ps(A[(ii+1) * k + ll]);
                            auto a_val_3 = _mm256_set1_ps(A[(ii+2) * k + ll]);
                            auto a_val_4 = _mm256_set1_ps(A[(ii+3) * k + ll]);
                            // auto a_val_5 = _mm256_set1_ps(A[(ii+4) * k + ll]);
                            for (int jj = j; jj < j + tileSize && (jj+8) <= n; jj+=8) {
                                //jj+8 : make sure next vectoriztion can hold all nums
                                //vectorize below operations
                                auto b_vec = _mm256_loadu_ps(&B[ll * n + jj]);
                                
                                auto c_vec = _mm256_loadu_ps(&C[ii * n + jj]); 
                                auto c_vec_2 = _mm256_loadu_ps(&C[(ii+1) * n + jj]); 
                                auto c_vec_3 = _mm256_loadu_ps(&C[(ii+2) * n + jj]); 
                                auto c_vec_4 = _mm256_loadu_ps(&C[(ii+3) * n + jj]); 
                                auto prod = _mm256_mul_ps(a_val, b_vec);  
                                auto prod_2 = _mm256_mul_ps(a_val_2, b_vec);  
                                auto prod_3 = _mm256_mul_ps(a_val_3, b_vec);  
                                auto prod_4 = _mm256_mul_ps(a_val_4, b_vec);  
                                prod = _mm256_add_ps(prod, c_vec);        
                                prod_2 = _mm256_add_ps(prod_2, c_vec_2);        
                                prod_3 = _mm256_add_ps(prod_3, c_vec_3); 
                                prod_4 = _mm256_add_ps(prod_4, c_vec_4); 
                                
                                _mm256_storeu_ps(&C[ii * n + jj], prod); 
                                _mm256_storeu_ps(&C[(ii+1) * n + jj], prod_2);
                                _mm256_storeu_ps(&C[(ii+2) * n + jj], prod_3);
                                _mm256_storeu_ps(&C[(ii+3) * n + jj], prod_4);
                            }
                            for(int jj = j + col_bnd; jj < n; jj++ ){
                                C[ii * n + jj] += A[ii * k + ll] * B[ll * n + jj];
                                C[(ii+1) * n + jj] += A[(ii+1) * k + ll] * B[ll * n + jj];
                                C[(ii+2) * n + jj] += A[(ii+2) * k + ll] * B[ll * n + jj];
                                C[(ii+3) * n + jj] += A[(ii+3) * k + ll] * B[ll * n + jj];
                            }
                        }
                    }
                }
            }
    }
    //remaining row column just handle directly
    for(int ii=row_bnd;ii < m;ii++){
        for(int ll=0;ll<k;ll++){
            for(int jj=0;jj<n;jj++){
                C[ii * n + jj] += A[ii * k + ll] * B[ll * n + jj];
            }
        }
    }
  }

}