// Created by SwiftWare Lab on 9/24.
// CE 4SP4 - High Performance Programming
// Copyright (c) 2024 SwiftWare Lab

#include "spmm.h"
#include <immintrin.h>
namespace swiftware::hpp {

  void spmmCSR(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        C[i * n + j] = 0;
        for (int l = Ap[i]; l < Ap[i + 1]; ++l) {
          C[i * n + j] += Ax[l] * B[Ai[l] * n + j];
        }
      }
    }
  }

  void spmmSkipping(int m, int n, int k, const float *A, const float *B, float *C, ScheduleParams Sp) {
    // TODO: Implement skipping version of spmm and optimize it
    for (int i = 0; i < m; ++i) {
      for(int l = 0; l < k; ++l){
        for (int j = 0; j < n; ++j) {
          if(A[i*k + l] !=0.0){
            C[i * n + j] += A[i * k + l] * B[l * n + j];
          }
        }
      }
    }
  }

  void spmmCSR_cache(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp){
    //apply tilling 
    int tileSize = Sp.TileSize1;

    for (int i = 0; i < m; i += tileSize) {
      for (int j = 0; j < n; j += tileSize) {
        for (int ii = i; ii < i + tileSize && ii < m; ii++){
          for (int jj = j; jj < j + tileSize && jj < n; jj++){
            C[ii * n + jj] = 0;
            for (int l = Ap[ii]; l < Ap[ii + 1]; ++l) {
              C[ii * n + jj] += Ax[l] * B[Ai[l] * n + jj];
            }
          }
        }
      }
    }
  }

  void spmmCSROptimized(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp) {
    // TODO: Implement optimized version of spmm
    for (int i = 0; i < m; ++i) {
      for (int j = Ap[i]; j < Ap[i + 1]; ++j){
        auto a_col_length = Ap[i+1] - Ap[i];
        auto min_length = std::min(a_col_length,n);
        auto col_bnd = min_length - min_length%32;
        auto a_val = _mm256_set1_ps(Ax[j]);

        for (int l = 0; l+32 <= (min_length); l+=32) {
          __m256 b_vec1 = _mm256_loadu_ps(&B[Ai[j]*n + l]);
          __m256 b_vec2 = _mm256_loadu_ps(&B[Ai[j]*n + l + 8]);
          __m256 b_vec3 = _mm256_loadu_ps(&B[Ai[j]*n + l + 16]);
          __m256 b_vec4 = _mm256_loadu_ps(&B[Ai[j]*n + l + 24]);

          __m256 mul_vec1 = _mm256_mul_ps(a_val, b_vec1);
          __m256 mul_vec2 = _mm256_mul_ps(a_val, b_vec2);
          __m256 mul_vec3 = _mm256_mul_ps(a_val, b_vec3);
          __m256 mul_vec4 = _mm256_mul_ps(a_val, b_vec4);

          __m256 c_vec1 = _mm256_loadu_ps(&C[i*n + l]);
          __m256 c_vec2 = _mm256_loadu_ps(&C[i*n + l + 8]);
          __m256 c_vec3 = _mm256_loadu_ps(&C[i*n + l + 16]);
          __m256 c_vec4 = _mm256_loadu_ps(&C[i*n + l + 24]);

          c_vec1 = _mm256_add_ps(c_vec1,mul_vec1);
          c_vec2 = _mm256_add_ps(c_vec2,mul_vec2);
          c_vec3 = _mm256_add_ps(c_vec3,mul_vec3);
          c_vec4 = _mm256_add_ps(c_vec4,mul_vec4);

          _mm256_storeu_ps(&C[i*n + l], c_vec1);
          _mm256_storeu_ps(&C[i*n + l + 8], c_vec2);
          _mm256_storeu_ps(&C[i*n + l + 16], c_vec3);
          _mm256_storeu_ps(&C[i*n + l + 24], c_vec4);
        }
        for(int jj = col_bnd; jj < n; jj++){
          C[i*n + jj] +=  Ax[j] * B[Ai[j]*n + jj];
        }
      }
    }
  }

  void spmmCSROptimized_tilled(int m, int n, int k, const int *Ap, const int *Ai, const float *Ax, const float *B, float *C, ScheduleParams Sp){
      // TODO: Implement optimized version of spmm
      //without tilling for now
      int tileSize = Sp.TileSize1;
      for (int i = 0; i < m; i+=tileSize) {
        for(int ii=i;ii<i+tileSize && ii < m;ii++){
            for (int j = Ap[ii]; j < Ap[ii + 1]; ++j){
              auto a_col_length = Ap[i+1] - Ap[i];
              auto min_length = std::min(a_col_length,n);
              auto col_bnd = min_length - min_length%32;
              auto a_val = _mm256_set1_ps(Ax[j]);
              for (int l = 0; l+32 <= min_length; l+=32) {
                __m256 b_vec1 = _mm256_loadu_ps(&B[Ai[j]*n + l]);
                __m256 b_vec2 = _mm256_loadu_ps(&B[Ai[j]*n + l + 8]);
                __m256 b_vec3 = _mm256_loadu_ps(&B[Ai[j]*n + l + 16]);
                __m256 b_vec4 = _mm256_loadu_ps(&B[Ai[j]*n + l + 24]);

                __m256 mul_vec1 = _mm256_mul_ps(a_val, b_vec1);
                __m256 mul_vec2 = _mm256_mul_ps(a_val, b_vec2);
                __m256 mul_vec3 = _mm256_mul_ps(a_val, b_vec3);
                __m256 mul_vec4 = _mm256_mul_ps(a_val, b_vec4);

                __m256 c_vec1 = _mm256_loadu_ps(&C[ii*n + l]);
                __m256 c_vec2 = _mm256_loadu_ps(&C[ii*n + l + 8]);
                __m256 c_vec3 = _mm256_loadu_ps(&C[ii*n + l + 16]);
                __m256 c_vec4 = _mm256_loadu_ps(&C[ii*n + l + 24]);

                c_vec1 = _mm256_add_ps(c_vec1,mul_vec1);
                c_vec2 = _mm256_add_ps(c_vec2,mul_vec2);
                c_vec3 = _mm256_add_ps(c_vec3,mul_vec3);
                c_vec4 = _mm256_add_ps(c_vec4,mul_vec4);

                _mm256_storeu_ps(&C[ii*n + l], c_vec1);
                _mm256_storeu_ps(&C[ii*n + l + 8], c_vec2);
                _mm256_storeu_ps(&C[ii*n + l + 16], c_vec3);
                _mm256_storeu_ps(&C[ii*n + l + 24], c_vec4);
              }
              for(int jj = col_bnd; jj < n; jj++){
                C[ii*n + jj] +=  Ax[j] * B[Ai[j]*n + jj];
              }
            }
        }
      }
  }

}