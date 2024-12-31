// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab
#include <iostream>
#include <vector>
#include <cstdlib>
#include <math.h>
#include "gemm.h"
#include "def.h"
#include <algorithm>
#include <immintrin.h> 
#include <cstdio>
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

    void gemmT1(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
        //same code as gemmt1 but I optimized the loop order
        int tileSize = Sp.TileSize1;


        for (int i = 0; i < m; i += tileSize) {  // row A
            for (int l = 0; l < k; l += tileSize) {  // shared col A with row B
                for (int j = 0; j < n; j += tileSize) {  // col B
                    // Compute each tile's product
                    for (int ii = i; ii < i + tileSize && ii < m; ii++) {
                        for (int ll = l; ll < l + tileSize && ll < k; ll++) {
                            for (int jj = j; jj < j + tileSize && jj < n; jj++) {
                                C[ii * n + jj] += A[ii * k + ll] * B[ll * n + jj];
                            }
                        }
                    }
                }
            }
        }
    }

    void gemmT2(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
        // tileSizeL2: For outer tiling (L2 cache), tileSize2: For inner tiling (L1 cache)
        int tileSizeL2 = Sp.TileSize1;  // L2 cache tile size
        int tileSizeL1 = Sp.TileSize2;  // L1 cache tile size

        // Outer loop: tile for L2 cache
        for (int i2 = 0; i2 < m; i2 += tileSizeL2) {      // Tile rows for L2
            for (int l2 = 0; l2 < k; l2 += tileSizeL2) {  // Shared dimension K for L2
                for (int j2 = 0; j2 < n; j2 += tileSizeL2) {  // Tile columns for L2
                    // Inner loop: tile for L1 cache
                    for (int i1 = i2; i1 < i2 + tileSizeL2 && i1 < m; i1 += tileSizeL1) {  // Tile rows for L1
                        for (int l1 = l2; l1 < l2 + tileSizeL2 && l1 < k; l1 += tileSizeL1) {  // Shared dimension K for L1
                            for (int j1 = j2; j1 < j2 + tileSizeL2 && j1 < n; j1 += tileSizeL1) {  // Tile columns for L1
                                // Compute the inner-most product for each L1 tile
                                for (int i = i1; i < i1 + tileSizeL1 && i<i2 + tileSizeL2 && i < m; i++) {
                                    for (int l = l1; l < l1 + tileSizeL1 && l<l2 + tileSizeL2 && l < k; l++) {
                                        for (int j = j1; j < j1 + tileSizeL1 && j<j2 + tileSizeL2 && j < n; j++) {
                                            C[i * n + j] += A[i * k + l] * B[l * n + j];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


    void gemmVectorized(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp){
        int tileSize = 512; //this tileSize provides the best spatial locality and good usage of cache on large cache

        //cache tilling
        if(m < 4 || n < 8 || k<8){
        //run regular matrix calculation if matrix is not big, could improve optimize on either row or col
            for (int i = 0; i < m; ++i) {
                for (int l = 0; l < k; ++l) {
                    for (int j = 0; j < n; ++j) {
                        C[i * n + j] += A[i * k + l] * B[l * n + j];
                    }
                }
            }
            return;
        }
        //divide the matrix into a boundary can divide by row/4, column/8
        auto row_bnd = m - m % 4;   
        auto col_bnd = n - n % 8;   

        for (int i = 0; i < m; i += tileSize) {  // row A
                for (int l = 0; l < k; l += tileSize) {  // shared col A with row B
                    for (int j = 0; j < n; j += tileSize) {  // col B
                        // Compute each tile's product
                        for (int ii = i; ii < i + tileSize && (ii+4) <= m; ii+=4) {
                            for (int ll = l; ll < l + tileSize && ll < k; ll++) {
                                // unroll loops by 4 rows
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
                        //handle remaining rows
                        for(int ii = i + row_bnd;ii < m;ii++){
                            for(int ll=l;ll<k;ll++){
                                for(int jj=j;jj<n;jj++){
                                    C[ii * n + jj] += A[ii * k + ll] * B[ll * n + jj];
                                }
                            }
                        }
                    }
                }
        }

    }

}