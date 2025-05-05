// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include "spmv.h"
#include <iostream>
#include <immintrin.h> // For AVX intrinsics
#include <stdio.h>

void print_avx_vector(__m256 vec) {
    // Create an array to store the values
    float values[8];
    
    // Store the AVX vector into the array
    _mm256_storeu_ps(values, vec);
    
    // Print each value in the array
    printf("AVX vector values: [ ");
    for (int i = 0; i < 8; i++) {
        printf("%f ", values[i]);
    }
    printf("]\n");
}
namespace swiftware::hpp {
  void spmvCSR(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp){
    //Ap is the rwo ptr
    //Ai is the column indexes array
    //Ax is the non zero values array
    for (int i = 0; i < m; ++i) {
      c[i] = 0;
      for (int l = Ap[i]; l < Ap[i + 1]; ++l) {
        c[i] += Ax[l] * b[Ai[l]];
      }
    }
  }


  void spmvCSROptimized(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp) {
    int tileSize = Sp.TileSize1;
    for (int i = 0; i < m; i+=tileSize) {
      for (int ii = i; ii < i + tileSize && ii < m; ii++) {
        c[ii] = 0; 
        int row_start = Ap[ii];
        int row_end = Ap[ii + 1];
        int row_length = row_end - row_start;
        int row_bnd = (row_length - row_length % 32);
        __m256 sum_vec = _mm256_setzero_ps(); 
        //loop through the nnz elemnets in a row
        for (int j = row_start; j < (row_start + row_bnd); j += 32) {
      
            // store appropriate elements of b ( non- consective)
            __m256 b_vec1 = _mm256_set_ps(
                b[Ai[j + 7]], b[Ai[j + 6]], b[Ai[j + 5]], b[Ai[j + 4]], 
                b[Ai[j + 3]], b[Ai[j + 2]], b[Ai[j + 1]], b[Ai[j]]
            );

            __m256 b_vec2 = _mm256_set_ps(
                b[Ai[j + 15]], b[Ai[j + 14]], b[Ai[j + 13]], b[Ai[j + 12]], 
                b[Ai[j + 11]], b[Ai[j + 10]], b[Ai[j + 9]], b[Ai[j+8]]
            );

            __m256 b_vec3 = _mm256_set_ps(
                b[Ai[j + 23]], b[Ai[j + 22]], b[Ai[j + 21]], b[Ai[j + 20]], 
                b[Ai[j + 19]], b[Ai[j + 18]], b[Ai[j + 17]], b[Ai[j+16]]
            );

            __m256 b_vec4 = _mm256_set_ps(
                b[Ai[j + 31]], b[Ai[j + 30]], b[Ai[j + 29]], b[Ai[j + 28]], 
                b[Ai[j + 27]], b[Ai[j + 26]], b[Ai[j + 25]], b[Ai[j+24]]
            );
            // load val elemnts of Ax (consecutive)
            __m256 ax_vec1 = _mm256_loadu_ps(&Ax[j]);
            __m256 ax_vec2 = _mm256_loadu_ps(&Ax[j+8]);
            __m256 ax_vec3 = _mm256_loadu_ps(&Ax[j+16]);
            __m256 ax_vec4 = _mm256_loadu_ps(&Ax[j+24]);

            __m256 mul_vec1 = _mm256_mul_ps(ax_vec1, b_vec1);
            __m256 mul_vec2 = _mm256_mul_ps(ax_vec2, b_vec2);
            __m256 mul_vec3 = _mm256_mul_ps(ax_vec3, b_vec3);
            __m256 mul_vec4 = _mm256_mul_ps(ax_vec3, b_vec4);

            sum_vec = _mm256_add_ps(sum_vec, mul_vec1);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec2);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec3);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec4);

        }


        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);  

        //horiztonal add elements in sum array
        for (int k = 0; k < 8; ++k) {
            c[ii] += sum_array[k];
        }

        // sum up remaining elements
        for (int jj = (row_start + row_bnd); jj < row_end; ++jj) {
            c[ii] += Ax[jj] * b[Ai[jj]];
        }
      }

    }
  }



  //vecotrized code only
  void spmvCSRVectorized(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp) {

    for (int i = 0; i < m; ++i) {
        int row_start = Ap[i];
        int row_end = Ap[i + 1];
        int row_length = row_end - row_start;
        int row_bnd = (row_length - row_length % 32);
        __m256 sum_vec = _mm256_setzero_ps(); 
        //loop through the nnz elemnets in a row
        for (int j = row_start; j < (row_start + row_bnd); j += 32) {
      
            // store appropriate elements of b ( non- consective)
            __m256 b_vec1 = _mm256_set_ps(
                b[Ai[j + 7]], b[Ai[j + 6]], b[Ai[j + 5]], b[Ai[j + 4]], 
                b[Ai[j + 3]], b[Ai[j + 2]], b[Ai[j + 1]], b[Ai[j]]
            );

            __m256 b_vec2 = _mm256_set_ps(
                b[Ai[j + 15]], b[Ai[j + 14]], b[Ai[j + 13]], b[Ai[j + 12]], 
                b[Ai[j + 11]], b[Ai[j + 10]], b[Ai[j + 9]], b[Ai[j+8]]
            );

            __m256 b_vec3 = _mm256_set_ps(
                b[Ai[j + 23]], b[Ai[j + 22]], b[Ai[j + 21]], b[Ai[j + 20]], 
                b[Ai[j + 19]], b[Ai[j + 18]], b[Ai[j + 17]], b[Ai[j+16]]
            );

            __m256 b_vec4 = _mm256_set_ps(
                b[Ai[j + 31]], b[Ai[j + 30]], b[Ai[j + 29]], b[Ai[j + 28]], 
                b[Ai[j + 27]], b[Ai[j + 26]], b[Ai[j + 25]], b[Ai[j+24]]
            );
            // load val elemnts of Ax (consecutive)
            __m256 ax_vec1 = _mm256_loadu_ps(&Ax[j]);
            __m256 ax_vec2 = _mm256_loadu_ps(&Ax[j+8]);
            __m256 ax_vec3 = _mm256_loadu_ps(&Ax[j+16]);
            __m256 ax_vec4 = _mm256_loadu_ps(&Ax[j+24]);

            __m256 mul_vec1 = _mm256_mul_ps(ax_vec1, b_vec1);
            __m256 mul_vec2 = _mm256_mul_ps(ax_vec2, b_vec2);
            __m256 mul_vec3 = _mm256_mul_ps(ax_vec3, b_vec3);
            __m256 mul_vec4 = _mm256_mul_ps(ax_vec3, b_vec4);

            sum_vec = _mm256_add_ps(sum_vec, mul_vec1);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec2);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec3);
            sum_vec = _mm256_add_ps(sum_vec, mul_vec4);

        }


        float sum_array[8];
        _mm256_storeu_ps(sum_array, sum_vec);  

        //horiztonal add elements in sum array
        for (int k = 0; k < 8; ++k) {
            c[i] += sum_array[k];
        }

        // sum up remaining elements
        for (int jj = (row_start + row_bnd); jj < row_end; ++jj) {
            c[i] += Ax[jj] * b[Ai[jj]];
        }


    }
  }

  //tiled code only
  void spmvCSRTiled(int m, int n, const int *Ap, const int *Ai, const float *Ax, const float *b, float *c, ScheduleParams Sp) {
      int tileSize = Sp.TileSize1;
      
      for (int i = 0; i < m; i += tileSize) {
          for (int ii = i; ii < i + tileSize && ii < m; ii++) {
              c[ii] = 0;  

              for (int l = Ap[ii]; l < Ap[ii + 1]; ++l) {
                c[ii] += Ax[l] * b[Ai[l]];
              }

              
          }
      }
  }



  void spmvSkipping(int m, int n, const float *A, const float *b, float *c, ScheduleParams Sp){
    // TODO: implement and optimize skipping version of spmv ( this uses the dense nn version)
    for (int i = 0; i < m; ++i) {
      int in = i * n;
        for (int j = 0; j < n; ++j) {
            auto b_val = b[j];
            if (b_val!=0.0){
              auto a_val = A[in + j];
              if (a_val!=0.0){
                c[i] += a_val * b_val;
              }
            }
        }
    }

  }
}